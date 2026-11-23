from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_manifest import publishable
from instance_generator import prepare_runtime_instance, sha256_file
from main import build_initial_user_message, public_task_configuration
from scoring import _dominant_attractive, _parse_formal_log, _parse_sapt_log, score_rules
from stage_validation import coordinate_hash, validate_stage_A, validate_stage_B, validate_stage_C
from tool_executor import run_python_tool, safety_check, timeout_floor_for_script
from utils import load_json, load_yaml

TASK = ROOT / "tasks" / "oligothiophene_smd_sapt_v3"
FORMAL_SYSTEMS = ["gas", "n-hexane", "1,4-dioxane", "chloroform", "water"]


def detect_anomalies(workspace: Path, public: dict) -> list[tuple[str, str]]:
    from rdkit import Chem
    found: list[tuple[str, str]] = []
    metadata = json.loads((workspace / "seed_evidence" / "structure_metadata.json").read_text())
    sdf_path = workspace / "seed_evidence" / "provided_structure.sdf"
    if metadata["sha256"] != sha256_file(sdf_path):
        found.append(("seed_evidence/structure_metadata.json", "stale_hash"))
    mol = next(m for m in Chem.SDMolSupplier(str(sdf_path), removeHs=False) if m is not None)
    if int(metadata["explicit_atom_count"]) != mol.GetNumAtoms():
        found.append(("seed_evidence/structure_metadata.json", "atom_count_mismatch"))
    probe = json.loads((workspace / "seed_evidence" / "solvent_probe.json").read_text())
    water = public["solvents"]["water"]
    if probe["smiles"] == water["smiles"] and (probe["solvent_name"] != "water" or probe["smd_name"] != water["smd_name"]):
        found.append(("seed_evidence/solvent_probe.json", "solvent_label_mismatch"))
    if probe["smiles"] == water["smiles"] and probe["solvent_name"] == "water" and probe["smd_name"] == water["smd_name"] and probe["xtb_alpb_name"] != water["xtb_alpb_name"]:
        found.append(("seed_evidence/solvent_probe.json", "solvent_alias_mismatch"))
    return found


def write_formal_log(path: Path, solvent: str, energy: float, geometry: str = "structures/R_g.xyz") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "BENCHMARK_PROGRAM: PySCF\n"
        "BENCHMARK_FUNCTIONAL: M05-2X\n"
        "BENCHMARK_BASIS: 6-31G*\n"
        "BENCHMARK_REFERENCE: RKS\n"
        f"BENCHMARK_SOLVENT: {solvent}\n"
        "BENCHMARK_CHARGE: 0\n"
        "BENCHMARK_MULTIPLICITY: 1\n"
        f"BENCHMARK_GEOMETRY_FILE: {geometry}\n"
        f"BENCHMARK_TOTAL_ENERGY_HARTREE: {energy:.12f}\n"
        "BENCHMARK_CONVERGED: true\n"
        "BENCHMARK_HOMO_HARTREE: -0.220000000000\n"
        "BENCHMARK_LUMO_HARTREE: -0.080000000000\n"
        "BENCHMARK_DIPOLE_DEBYE_XYZ: 0.1000000000,0.2000000000,0.3000000000\n"
        f"converged SCF energy = {energy:.12f}\n",
        encoding="utf-8",
    )


def write_sapt_log(path: Path, endpoint: str, complex_rel: str, a_rel: str, b_rel: str, values: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "BENCHMARK_PROGRAM: Psi4\n"
        "BENCHMARK_METHOD: SAPT0\n"
        "BENCHMARK_BASIS: jun-cc-pVDZ\n"
        f"BENCHMARK_SOLVENT: {endpoint}\n"
        f"BENCHMARK_GEOMETRY_FILE: {complex_rel}\n"
        f"BENCHMARK_FRAGMENT_A_GEOMETRY: {a_rel}\n"
        f"BENCHMARK_FRAGMENT_B_GEOMETRY: {b_rel}\n"
        "BENCHMARK_FRAGMENT_A_ATOMS: 2\n"
        "BENCHMARK_FRAGMENT_B_ATOMS: 1\n"
        f"BENCHMARK_SAPT_ELST_KCAL_MOL: {values['electrostatics']}\n"
        f"BENCHMARK_SAPT_EXCH_KCAL_MOL: {values['exchange']}\n"
        f"BENCHMARK_SAPT_IND_KCAL_MOL: {values['induction']}\n"
        f"BENCHMARK_SAPT_DISP_KCAL_MOL: {values['dispersion']}\n"
        f"BENCHMARK_SAPT_TOTAL_KCAL_MOL: {values['total']}\n",
        encoding="utf-8",
    )


def copy_contracts(workspace: Path) -> None:
    for name in ("stage_A_schema.json", "stage_B_schema.json", "stage_C_schema.json", "result_schema.json"):
        shutil.copy2(TASK / name, workspace / name)


def build_stage_a(workspace: Path, public: dict, truth: dict) -> None:
    (workspace / "structures").mkdir(parents=True, exist_ok=True)
    (workspace / "analysis").mkdir(parents=True, exist_ok=True)
    shutil.copy2(workspace / "seed_evidence" / "provided_structure.sdf", workspace / "structures" / "solute_validated.sdf")
    shutil.copy2(workspace / "seed_evidence" / "provided_structure.xyz", workspace / "structures" / "solute_validated.xyz")
    from rdkit import Chem
    mol = next(m for m in Chem.SDMolSupplier(str(workspace / "structures" / "solute_validated.sdf"), removeHs=False) if m is not None)
    sulfur = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
    carbons = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
    # The exact chemistry mapping is tested by scorer; these indices are in range and chemically typed.
    mapping = {"inter_repeat_bond": carbons[2:4], "inter_repeat_dihedral": [sulfur[0], carbons[2], carbons[3], sulfur[1]], "sulfur_atoms": sulfur[:2], "terminal_carbons": [carbons[0], carbons[-1]]}
    (workspace / "analysis" / "atom_mapping.json").write_text(json.dumps(mapping, indent=2))
    (workspace / "analysis" / "accepted_input_hashes.json").write_text(json.dumps({"structure_sha256":sha256_file(workspace / "structures" / "solute_validated.sdf")}, indent=2))
    findings = [{"artifact":x["artifact"],"anomaly_type":x["anomaly_type"],"evidence":x["anomaly_reason"],"action":"exclude and regenerate metadata; do not use downstream"} for x in truth["anomalies"]]
    audit = {
      "stage":"A",
      "molecular_model":{"structure_sdf":"structures/solute_validated.sdf","structure_xyz":"structures/solute_validated.xyz","formula":truth["expected_formula"],"canonical_smiles":truth["expected_canonical_smiles"],"charge":0,"multiplicity":1,"atom_mapping_file":"analysis/atom_mapping.json"},
      "solvent_identity_audit":{name:{"smiles":cfg["smiles"],"smd_name":cfg["smd_name"],"xtb_alpb_name":cfg["xtb_alpb_name"]} for name,cfg in public["solvents"].items()},
      "anomaly_audit":{"findings":findings,"accepted_inputs":["structures/solute_validated.sdf"]},
      "microtests":{"xtb":{"passed":True},"pyscf":{"passed":True},"psi4":{"passed":True}},
      "provenance":["SHA256 input hashes","protocol version 3.0"]
    }
    (workspace / "stage_A_audit.json").write_text(json.dumps(audit, indent=2))
    (workspace / "plan.json").write_text(json.dumps({"resources":"threads memory parallel timeout scratch separate directory failure isolation","budget":"allocate remaining calls by uncertainty"}, indent=2))


def build_stage_b(workspace: Path, public: dict, reference: dict) -> dict:
    shutil.copy2(workspace / "seed_evidence" / "provided_structure.xyz", workspace / "structures" / "R_g.xyz")
    digest = coordinate_hash(workspace / "structures" / "R_g.xyz", public["coordinate_hash_spec"]["round_digits"])
    (workspace / "analysis" / "formal_coordinate_hashes.json").write_text(json.dumps({"geometry_file":"structures/R_g.xyz","coordinate_sha256":digest,"systems":{n:digest for n in FORMAL_SYSTEMS}}, indent=2))
    energies = reference["smd"]["energy_hartree"]
    logs = {}
    for name in FORMAL_SYSTEMS:
        rel=f"logs/formal/{name}.log"; write_formal_log(workspace/rel,name,energies[name]); logs[name]=rel
    sens="logs/formal/sensitivity.log"; write_formal_log(workspace/sens,"chloroform",energies["chloroform"])
    seed_files=[]
    for idx,angle in enumerate((-165,-95,-20,55,135,165)):
        rel=f"structures/seeds/seed_{idx}.xyz"; (workspace/rel).parent.mkdir(parents=True,exist_ok=True); shutil.copy2(workspace/"structures/R_g.xyz",workspace/rel); seed_files.append((rel,angle))
    stage_b={
      "stage":"B",
      "gas_conformer_search":{"attempts":[{"seed":i,"status":"success","starting_geometry_file":rel,"initial_torsion_deg":angle} for i,(rel,angle) in enumerate(seed_files)],"retained_conformers":[{"geometry_file":"structures/R_g.xyz","relative_energy_kcal_mol":0.0},{"geometry_file":seed_files[1][0],"relative_energy_kcal_mol":0.5}],"selected_R_g":{"geometry_file":"structures/R_g.xyz","coordinate_sha256":digest}},
      "formal_calculations":{"method":public["methods"]["formal_single_point"],"geometry_file":"structures/R_g.xyz","energies_hartree":energies,"log_files":logs,"solvation_energies_kcal_mol":reference["smd"]["delta_kcal_mol"]},
      "sensitivity_calculation":{"type":"second_retained_conformer","rationale":"conformer uncertainty and solvent ranking stability","log_files":[sens],"ranking_preserved":True},
      "ranking":{"lowest_solvation_free_energy_solvent":"chloroform","ranking_robust_under_sensitivity":True,"near_tie":False},
      "provenance":["protocol version 3.0","SHA256 formal coordinate hash"]
    }
    (workspace / "stage_B_bulk.json").write_text(json.dumps(stage_b, indent=2))
    return stage_b


def build_stage_c(workspace: Path) -> dict:
    endpoints={}
    for endpoint,values,reported in (
      ("n-hexane",{"electrostatics":-2.0,"exchange":7.0,"induction":-0.5,"dispersion":-8.0,"total":-3.5},"Dispersion"),
      ("water",{"electrostatics":-7.0,"exchange":8.0,"induction":-1.2,"dispersion":-2.0,"total":-2.2},"Electrostatics"),
    ):
        base=f"stage_c/{endpoint}"; selected=f"{base}/selected.xyz"; opt=f"{base}/opt.log"
        (workspace/selected).parent.mkdir(parents=True,exist_ok=True); shutil.copy2(workspace/"structures/R_g.xyz",workspace/selected); (workspace/opt).write_text("xTB optimization success")
        a=f"{base}/solute.xyz"; b=f"{base}/solvent.xyz"; complex_rel=f"{base}/complex.xyz"
        shutil.copy2(workspace/selected,workspace/a); (workspace/b).write_text("1\nsolvent\nO 0 0 0\n"); shutil.copy2(workspace/selected,workspace/complex_rel)
        complexes=[]
        for i in range(3):
            rel=f"{base}/arrangement_{i}.xyz"; data=(workspace/selected).read_text().splitlines(); parts=data[2].split(); parts[1]=str(float(parts[1])+0.01*i); data[2]=' '.join(parts); (workspace/rel).write_text('\n'.join(data)+'\n')
            complexes.append({"geometry_file":rel,"source_endpoint_geometry":selected,"intact":True,"frozen_interaction_energy_kcal_mol":-1.0-i,"deformation_energy_kcal_mol":0.1+i*0.01})
        sapt_rel=f"logs/sapt/{endpoint}.log"; write_sapt_log(workspace/sapt_rel,endpoint,complex_rel,a,b,values)
        endpoints[endpoint]={"selected_structure_file":selected,"optimization_log":opt,"geometry_metrics":{},"explicit_complexes":complexes,"representative":{"complex_geometry_file":complex_rel,"monomer_a_xyz":a,"monomer_b_xyz":b,"source_endpoint_geometry":selected,"selection_rationale":"lowest intact diverse arrangement"},"sapt":{"raw_log":sapt_rel,"dominant_attractive_component":reported}}
    stage_c={"stage":"C","endpoints":endpoints,"endpoint_comparison":{"heavy_atom_rmsd_angstrom":0.0,"twist_difference_deg":0.0,"end_to_end_difference_angstrom":0.0},"provenance":["protocol version 3.0","geometry hashes"]}
    (workspace/"stage_C_local.json").write_text(json.dumps(stage_c,indent=2))
    return stage_c


def assert_rejected(script: str, expected: str) -> None:
    try: safety_check(script,max_script_size_bytes=300000)
    except ValueError as exc: assert expected in str(exc),(expected,str(exc))
    else: raise AssertionError(f"script was not rejected: {script}")


def main() -> None:
    cfg=load_yaml(TASK/"task_config.yaml")
    # Release 1.3: the public Agent runner never reads evaluator-private truth.
    # This optional variable is used only by this evaluator-side self-test.
    private_dir = os.environ.get("CHEMBENCH_TEST_PRIVATE_SCORING_DIR")
    has_private_reference = bool(private_dir)
    if private_dir:
        reference = load_json(Path(private_dir) / "scoring_reference.json")
    else:
        # Mock reference for structural tests (no real numeric secrets).
        # Must provide gas energy for build_stage_b to succeed.
        reference = {
            "smd": {
                "energy_hartree": {k: -1000.0 for k in FORMAL_SYSTEMS},
                # Gas is needed to write formal logs, but is not a valid
                # solvation-energy schema key.
                "delta_kcal_mol": {k: 0.0 for k in FORMAL_SYSTEMS if k != "gas"},
            },
            "sapt": {},
        }
    assert sum(cfg["scoring"]["sections"].values())==100
    assert sum(v for k,v in cfg["scoring"]["sections"].items() if k!="scientific_scope_and_uncertainty")==90
    assert cfg["scoring"]["sections"]["input_audit_and_identity"]==16
    assert cfg["tool_call_budget_enabled"] is False
    assert cfg["preferred_tool_call_budget"] is None
    assert cfg["fixed_tool_call_budget"] is None
    assert cfg["tool_call_penalties"]=={}
    assert "tolerance" not in json.dumps(reference["sapt"]).lower()
    visible_cfg=public_task_configuration(cfg)
    visible_blob=json.dumps(visible_cfg).lower()
    assert "instance_generation" not in visible_blob
    assert "hidden_anomaly_count" not in visible_blob
    assert "structure_anomaly_types" not in visible_blob
    assert "solvent_anomaly_types" not in visible_blob
    assert "scoring_reference" not in visible_blob
    prompt_blob=build_initial_user_message("QUESTION",cfg,{"instance_id":"public"}).lower()
    assert "hidden_anomaly_count" not in prompt_blob and "scoring_reference" not in prompt_blob
    for builtin in ("exec", "eval", "compile"):
        assert_rejected(f"{builtin}('print(1)')", "dynamic code execution is forbidden")
        assert_rejected(f"builtins.{builtin}('print(1)')", "dynamic code execution is forbidden")

    # Verify the retained control has the same release version and no public reference.
    CONTROL_TASK = ROOT / "tasks" / "oligothiophene_smd_sapt_v3_unbounded_control"
    control_cfg = load_yaml(CONTROL_TASK / "task_config.yaml")
    assert control_cfg["scoring"]["benchmark_version"] == "1.3", f"control task version: {control_cfg['scoring'].get('benchmark_version')}"
    assert control_cfg["requirements"]["scoring_reference"] is None, "control scoring_reference must be null"
    assert "unbounded_control" in str(CONTROL_TASK)
    # Verify control task public config hides scoring_reference
    control_visible = public_task_configuration(control_cfg)
    control_blob = json.dumps(control_visible).lower()
    assert "scoring_reference" not in control_blob

    for seed in (20260723,20260724,20260725):
        with tempfile.TemporaryDirectory() as tmp:
            ws=Path(tmp); public,truth=prepare_runtime_instance(cfg,ws,seed_override=seed)
            anomalies=detect_anomalies(ws,public)
            assert len(anomalies)==2,(seed,anomalies)
            assert len({a for a,_ in anomalies})==2
            assert sorted(anomalies)==sorted((x["artifact"],x["anomaly_type"]) for x in truth["anomalies"])
            # Same seed must reproduce the same hidden pair.
            with tempfile.TemporaryDirectory() as tmp2:
                _,truth2=prepare_runtime_instance(cfg,Path(tmp2),seed_override=seed)
                assert truth2["anomalies"]==truth["anomalies"]

    with tempfile.TemporaryDirectory() as tmp:
        ws=Path(tmp); public,truth=prepare_runtime_instance(cfg,ws,seed_override=20260723); truth["scoring_reference"]=reference; copy_contracts(ws)
        build_stage_a(ws,public,truth)
        assert not validate_stage_A(ws),validate_stage_A(ws)
        build_stage_b(ws,public,reference)
        assert not validate_stage_B(ws/"stage_B_bulk.json",ws),validate_stage_B(ws/"stage_B_bulk.json",ws)
        build_stage_c(ws)
        assert not validate_stage_C(ws),validate_stage_C(ws)
        (ws/"revisions.json").write_text("[]\n")
        score=score_rules(ws/"run",ws,{},instance_truth=truth,task_cfg=cfg)
        # Numerical SMD/ranking expectations require the evaluator-private
        # reference. The public-mode mock deliberately contains no such
        # values, while structural/schema/safety checks remain runnable.
        if has_private_reference:
            assert score["breakdown"]["fixed_geometry_smd"]==20.0,score["breakdown"]
        assert score["breakdown"]["sapt_evidence_and_dominance"]==12.0,score["details"]["sapt_recompute"]
        assert score["breakdown"]["multi_configuration_sapt"]==0.0
        assert score["breakdown"]["cross_layer_consistency"]==0.0
        if has_private_reference:
            solvent_detail=score["details"]["solvent_identity"]
            assert all(abs(x["score"]-0.5)<1e-9 for x in solvent_detail.values()),solvent_detail
        # SAPT values deliberately differ from any reference vector; only dominant-interaction analysis is scored. Evidence lineage is an eligibility gate.
        assert score["details"]["sapt_recompute"]["n-hexane"]["raw_dominance"]=="dispersion"
        assert score["details"]["sapt_recompute"]["water"]["raw_dominance"]=="electrostatics"
        assert score["details"]["sapt_recompute"]["n-hexane"]["dominance_score"]==7.0
        assert score["details"]["sapt_recompute"]["water"]["dominance_score"]==7.0
        # Evidence gates carry no partial points: breaking one endpoint's lineage removes that endpoint's full 7-point dominance item.
        broken=json.loads((ws/"stage_C_local.json").read_text())
        broken_fragment="stage_c/water/nonmatching_solvent.xyz"
        (ws/broken_fragment).write_text("1\nsolvent\nO 1 0 0\n")
        broken["endpoints"]["water"]["representative"]["monomer_b_xyz"]=broken_fragment
        (ws/"stage_C_local.json").write_text(json.dumps(broken,indent=2))
        gated=score_rules(ws/"run",ws,{},instance_truth=truth,task_cfg=cfg)
        assert gated["breakdown"]["sapt_evidence_and_dominance"]==6.0,gated["details"]["sapt_recompute"]

    assert_rejected("from pyscf import gto","direct chemistry import")
    assert_rejected("import os\nos.remove('x')","forbidden call")
    assert_rejected("import subprocess\nsubprocess.run(['python','x.py'])","candidate subprocess calls are forbidden")
    safety_check("from benchmark_tools import run_xtb_single_point\nprint('ok')",max_script_size_bytes=300000)
    policy=load_yaml(ROOT/"configs/default.yaml")["execution"]["timeout_floor_seconds"]
    assert timeout_floor_for_script("from benchmark_tools import run_xtb_single_point\nrun_xtb_single_point('a','b')",policy)==("xtb_production",1800)

    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/"sapt.log"; p.write_text("BENCHMARK_SAPT_ELST_KCAL_MOL: -4\nBENCHMARK_SAPT_EXCH_KCAL_MOL: 3\nBENCHMARK_SAPT_IND_KCAL_MOL: -1\nBENCHMARK_SAPT_DISP_KCAL_MOL: -2\nBENCHMARK_SAPT_TOTAL_KCAL_MOL: -4\n")
        parsed=_parse_sapt_log(p); assert _dominant_attractive(parsed)=="electrostatics"

    with tempfile.TemporaryDirectory() as tmp:
        project=Path(tmp); workspace=project/"runs"/"test"/"workspace"; workspace.mkdir(parents=True)
        private=project/"private_reference.json"; private.write_text("secret")
        guarded=run_python_tool(script=f"from pathlib import Path\nprint(Path({str(private)!r}).read_text())",timeout_seconds=30,workspace=workspace,project_root=project,python_executable=sys.executable,max_script_runtime_seconds=30,max_script_size_bytes=300000,stdout_stderr_tail_chars=10000,tool_index=1)
        assert guarded["returncode"] != 0 and "outside workspace" in guarded["stderr"], guarded

    manifest=json.loads((ROOT/"MANIFEST.json").read_text(encoding="utf-8"))
    listed=set(manifest["files"])
    actual = set()
    for p in ROOT.rglob("*"):
        if not p.is_file() or not publishable(p):
            continue
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        actual.add(rel)
    assert listed==actual,(sorted(actual-listed),sorted(listed-actual))
    for rel, expected in manifest.get("sha256", {}).items():
        assert sha256_file(ROOT/rel)==expected, (rel, sha256_file(ROOT/rel), expected)
    print("Release 1.3 package self-test passed: hidden multi-anomaly generation, solvent identities, private SMD checks, SAPT scoring gates, stage schemas, safety, and manifest digests are valid.")


if __name__=="__main__": main()
