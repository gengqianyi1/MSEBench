"""Designer/reviewer check: everything stated in more than one place must agree, nothing on the
public side may disclose the answer, and there must be exactly one scorer.

History this file exists to prevent. The first release of this problem shipped a README that still
described a retired 25+75 rubric while the shipped config was 0+100, and the consistency script
reported 20/20 green because it only ever read `question.md`. The second-round review then asked for
two more invariants (田双燕, priority 2): that `rescore_run.py` really is the shipped scorer rather
than a second copy of the grading logic, and that README, question.md, task_config.yaml and the
scorer all state the same rubric. Both are asserted below, and both are FUNCTIONAL assertions --
the scorer is executed on synthetic inputs rather than grepped.

Sections:
  1.  the starting molecule parses and satisfies every hard constraint
  2.  the grader's reference equals the enumeration's own output
  3.  the gate values equal the enumeration's own output, and each gate still passes
  4.  numeric claims in question.md match the shipped config
  5.  the scoring block is internally consistent (bands monotone, totals add up)
  6.  ONE RUBRIC IN FOUR PLACES: README, question.md, task_config.yaml and the executed scorer
  7.  ONE SCORER: rescore_run.py is score_run.py, and only the grader defines the bands
  8.  EVALUATION ISOLATION: nothing under public/ discloses or implies the reference or the bands
  9.  public and grader agree about the task they describe
  10. the regression suites pass
  11. the shipped trajectories are complete, and their stored scores recompute exactly
  12. every document that quotes a model score agrees with the trajectories
  13. the surrogate-property positioning is stated consistently everywhere
  14. no run output contains anything that looks like a live API key

Every traversal assertion carries a COUNT GUARD: if the loop finds nothing it fails instead of
passing vacuously, and it prints the count it actually found.

Run:
    python grader/scripts/check_consistency.py
Exit status is non-zero if any assertion fails, so this is usable in a pre-delivery hook.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
# Run directories used to live at the repository root while the harness config said `../runs`,
# which is correct only while `public/` sits inside this repository. In a standalone participant
# package there is no parent to write into, so the second-round packages set `run_dir: runs` and the
# directory lives inside the package. Both layouts are accepted here so that first-round run
# directories stay gradeable.
RUNS = ROOT / "public" / "runs" if (ROOT / "public" / "runs").is_dir() else ROOT / "runs"
sys.path.insert(0, str(ROOT / "public" / "src"))
sys.path.insert(0, str(ROOT / "grader" / "src"))
sys.path.insert(0, str(ROOT / "grader" / "scripts"))

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from evaluator import MoleculeEvaluation, evaluate_smiles, check_constraints  # noqa: E402
from scoring import final_score  # noqa: E402
from utils import load_yaml  # noqa: E402

PUBLIC = ROOT / "public"
GRADER = ROOT / "grader"
TASK_CFG = PUBLIC / "tasks" / "q1_multiobjective" / "task_config.yaml"
QUESTION = PUBLIC / "tasks" / "q1_multiobjective" / "question.md"
PUBLIC_README = PUBLIC / "README.md"
REFERENCE_CFG = GRADER / "reference" / "reference_target.yaml"
ENUM = GRADER / "answers" / "reference_enumeration.json"
GRADER_NOTES = GRADER / "审核说明.md"
GRADER_README = GRADER / "README.md"
ROOT_README = ROOT / "README.md"
REMEDIATION = ROOT / "整改说明2.md"
SCORING_SRC = GRADER / "src" / "scoring.py"
RESCORE_SRC = GRADER / "scripts" / "rescore_run.py"
SCORE_RUN_SRC = GRADER / "scripts" / "score_run.py"
PYTHON = sys.executable

failures: list[str] = []
checks = 0


def check(condition: bool, message: str) -> None:
    global checks
    checks += 1
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def approx(a, b, tol=1e-4) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def public_files() -> list[Path]:
    out = []
    for p in PUBLIC.rglob("*"):
        if not p.is_file():
            continue
        if any(part in {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"} for part in p.parts):
            continue
        if p.suffix == ".pyc":
            continue
        out.append(p)
    return out


def _synthetic(**over) -> MoleculeEvaluation:
    base = dict(smiles="synthetic", valid=True, canonical_smiles="synthetic",
                molecular_weight=None, logp=None, tpsa=None, qed=None, sa_score=None,
                similarity_to_start=None, n_alerts=0, alerts=[])
    base.update(over)
    return MoleculeEvaluation(**base)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-trajectories", action="store_true",
                    help="DEV ONLY: skip sections 11-12. A delivery run must not use this.")
    ap.add_argument("--probe-only", action="store_true",
                    help="the trajectories present are a PARTIAL calibration, not the full "
                         "3 models x 2 replicates. The run still asserts everything that can be "
                         "asserted about the trajectories that ARE present, and prints exactly "
                         "what is missing. A delivery run must not use this.")
    args = ap.parse_args()

    cfg = load_yaml(TASK_CFG)
    constraints = cfg["constraints"]
    sim_cfg = constraints["similarity"]
    sim_floor = float(sim_cfg["min"])
    ref_file = load_yaml(REFERENCE_CFG)
    ref = ref_file["reference"]
    gates = ref_file["gates"]
    scoring = ref_file["scoring"]

    print("1. starting molecule")
    start = evaluate_smiles(cfg["start_smiles"], start_smiles=cfg["start_smiles"],
                            similarity_cfg=sim_cfg)
    check(start.valid, "start SMILES parses and is admissible")
    flags = check_constraints(start, constraints)
    check(len(flags) >= 6, f"the constraint checker reports {len(flags)} flags (expected >= 6)")
    for name, satisfied in flags.items():
        check(satisfied, f"start satisfies {name}")

    enum = None
    print("\n2. grader reference matches the enumeration")
    if not ENUM.exists():
        check(False, f"{ENUM.relative_to(ROOT)} exists "
                     f"(run grader/scripts/enumerate_reference_best.py)")
    else:
        enum = json.loads(ENUM.read_text(encoding="utf-8"))
        check(approx(ref["best_objective_in_window"], enum["best_objective_in_window"]),
              "reference.best_objective_in_window == enumeration")
        check(ref["best_smiles_in_window"] == enum["best_smiles_in_window"],
              "reference.best_smiles_in_window == enumeration")
        check(approx(ref.get("start_objective"), enum["start_qed"]),
              "reference.start_objective == enumeration start QED")
        check(approx(ref.get("start_objective"), start.qed),
              "reference.start_objective == freshly computed start QED")
        check(ref.get("n_feasible_solutions_found") == enum["n_feasible_solutions_found"],
              "reference.n_feasible_solutions_found == enumeration")
        check(ref.get("n_slots_changed_vs_start") == enum["n_slots_changed_vs_start"],
              "reference.n_slots_changed_vs_start == enumeration")
        check(approx(ref.get("reference_similarity_to_start"),
                     enum["reference_properties"]["similarity"], tol=1e-3),
              "reference similarity to start == enumeration")
        check(approx(ref.get("objective_span"), enum["objective_span"]),
              "reference.objective_span == enumeration")

        print("\n3. gates match the enumeration, and each gate passes")
        for key in ["gate_g1_one_slot_ceiling", "gate_g1_thick_tinker_ceiling",
                    "gate_g2_safe_similarity_ceiling", "gate_g2_safe_margin", "gate_g3_n_feasible"]:
            check(key in enum and approx(gates.get(key), enum[key], tol=1e-3),
                  f"{key} == enumeration ({gates.get(key)} vs {enum.get(key)})")
        check(enum["gate_g1_one_slot_ceiling"] < 50, "G1 one-slot ceiling < 50")
        check(enum["gate_g1_thick_tinker_ceiling"] < 50, "G1 one+two-slot ceiling < 50")
        check(enum["gate_g2_safe_similarity_ceiling"] < 50, "G2 safe-similarity ceiling < 50")
        check(enum["gate_g2_safe_similarity_ceiling"] != 100,
              "G2 is not 100 (the edit-budget half of the mechanism actually binds)")
        check(approx(enum["gate_g2_safe_margin"], 0.05),
              "G2 uses the same 0.05 margin the start-molecule screen used")
        check(enum["gate_g3_n_feasible"] >= 50,
              f"G3 >= 50 feasible improving designs (fairness): {enum['gate_g3_n_feasible']}")
        # P-graded: gates bound the ceiling, these bound whether a competent design can land.
        # Introduced after three instances passed G1/G2/G3 and still scored 0 on every trajectory.
        # P1 is gated on the WINDOWS flavour. Gating the all-constraints flavour at 5% is
        # arithmetically incompatible with G2 < 50: any high-QED design at similarity >=
        # floor+0.05 would score >= 63 and break G2, so that bucket is forced to zero. The
        # all-constraints number is still recorded and asserted equal to the enumeration.
        check(enum["p_graded_p1_windows_fraction"] >= 0.05,
              f"P1 >= 5%: of the {enum['p_graded_n_high_qed']} designs at QED >= start+0.6*span, "
              f"{enum['p_graded_high_qed_in_windows']} fit the property windows "
              f"({100 * enum['p_graded_p1_windows_fraction']:.1f}%)")
        check(enum["p_graded_p2_n_comfortable"] >= 10,
              f"P2 >= 10 feasible improving designs with margin on every constraint at once "
              f"(found {enum['p_graded_p2_n_comfortable']})")
        check(enum["n_feasible_improving_one_slot_edits"] >= 1,
              f"at least one single-slot edit is feasible AND improving "
              f"(found {enum['n_feasible_improving_one_slot_edits']})")
        for key in ["p_graded_p1_windows_fraction", "p_graded_p1_all_fraction",
                    "p_graded_p2_n_comfortable", "n_feasible_improving_one_slot_edits"]:
            check(key in gates and approx(gates[key], enum[key], tol=1e-3),
                  f"{key} == enumeration ({gates.get(key)} vs {enum.get(key)})")
        # The score surface must be a ramp, not a wall. Two properties are asserted, both of them
        # falsifiable: enough distinct bands are reachable that partial credit is real, and no
        # single band swallows the distribution (which would make the score effectively binary).
        dist = enum["score_distribution"]
        band_points = sorted({int(b["points"]) for b in scoring["optimality_bands"]})
        populated = [p for p in band_points if str(p) in dist and dist[str(p)] > 0]
        check(len(populated) >= 4,
              f"at least 4 of the {len(band_points)} score bands are reachable "
              f"({len(populated)} populated: {dist})")
        total_designs = sum(dist.values())
        biggest = max(dist.values())
        check(total_designs >= 50 and biggest / total_designs <= 0.80,
              f"no single band holds more than 80% of the {total_designs} feasible designs "
              f"(largest is {biggest}, {100 * biggest / total_designs:.0f}%)")
        check(str(max(band_points)) in dist and dist[str(max(band_points))] < total_designs / 2,
              "the top band is reachable but is not where most designs land")

        # ------------------------------------------------------------------------------------
        # THE PROSE NEXT TO THE DATA MUST AGREE WITH THE DATA.
        #
        # reference_target.yaml carried a sentence from a previous instance -- "over those 59
        # designs. Four of the seven bands are populated" -- while the dict beside it summed to
        # 331 across all seven. Nothing asserted the sentence, only the numbers, which is the
        # same failure shape as the README that documented a retired rubric while the consistency
        # script reported green. The sentence is now written in a machine-checkable form and
        # parsed back out.
        # ------------------------------------------------------------------------------------
        ref_text = REFERENCE_CFG.read_text(encoding="utf-8")
        m_designs = re.search(r"Score distribution over those (\d+) designs", ref_text)
        # Search for the band phrase only AFTER the design-count phrase, so that the file's own
        # explanation of the bug it once had ("it said 'over those 59 designs...'") can never
        # shadow the live sentence.
        m_bands = (re.search(r"(\d+) of the (\d+) bands are populated", ref_text[m_designs.end():])
                   if m_designs else None)
        check(bool(m_designs),
              "reference_target.yaml states the design count in the parseable form "
              "'Score distribution over those N designs'")
        check(bool(m_bands),
              "reference_target.yaml states the band count in the parseable form "
              "'N of the M bands are populated'")
        if m_designs:
            check(int(m_designs.group(1)) == total_designs,
                  f"the prose in reference_target.yaml says {m_designs.group(1)} designs; "
                  f"score_distribution sums to {total_designs}")
        if m_bands:
            check(int(m_bands.group(1)) == len(populated),
                  f"the prose says {m_bands.group(1)} bands are populated; "
                  f"score_distribution has {len(populated)} non-empty bands")
            check(int(m_bands.group(2)) == len(band_points),
                  f"the prose says there are {m_bands.group(2)} bands in total; "
                  f"the rubric defines {len(band_points)}")
        # The same file must not still be describing a superseded instance.
        check(str(enum["gate_g3_n_feasible"]) in ref_text,
              f"reference_target.yaml mentions the current feasible count "
              f"({enum['gate_g3_n_feasible']})")
        check(enum["objective_span"] >= 0.05,
              f"the objective span is chemically meaningful, not noise "
              f"({enum['objective_span']:.4f} QED units)")

        print("\n4. numbers quoted in question.md")
        text = QUESTION.read_text(encoding="utf-8") if QUESTION.exists() else ""
        check(bool(text), "question.md exists and is non-empty")
        quoted = 0
        if text:
            check(cfg["start_smiles"] not in text,
                  "question.md does not hardcode the start SMILES (main.py injects it)")
            for name, key in (("molecular_weight", "mw"), ("logp", "logp"), ("tpsa", "tpsa")):
                lo, hi = enum["windows"][key]
                lo_s = f"{lo:g}" if key != "logp" else f"{lo}"
                hi_s = f"{hi:g}" if key != "logp" else f"{hi}"
                found = lo_s in text and hi_s in text
                quoted += 1
                check(found, f"question.md quotes the {name} window {lo_s}-{hi_s}")
            check(f"{enum['windows']['sa_max']}" in text, "question.md quotes the SA cap")
            check(f"{enum['windows']['similarity_min']}" in text,
                  "question.md quotes the similarity floor")
            quoted += 2
        check(quoted >= 5, f"at least 5 numeric window claims were cross-checked (found {quoted})")

    print("\n4b. prompt identity")
    import hashlib as _hashlib
    q_sha = _hashlib.sha256(QUESTION.read_bytes()).hexdigest()
    check(bool(cfg.get("prompt_version")),
          f"task_config declares a prompt_version ({cfg.get('prompt_version')})")
    check(cfg.get("question_sha256") == q_sha,
          f"task_config.question_sha256 matches the actual question.md digest "
          f"({str(cfg.get('question_sha256'))[:16]}... vs {q_sha[:16]}...)")

    print("\n5. scoring block is internally consistent")
    check(scoring["constraint_points"] == 0,
          "constraint_points == 0 (feasibility is a gate, not credit)")
    check(scoring["optimality_points"] == scoring["total_points"],
          "optimality_points == total_points")
    bands = scoring["optimality_bands"]
    mins = [float(b["min_achievement"]) for b in bands]
    ptss = [int(b["points"]) for b in bands]
    check(len(bands) >= 5, f"the rubric defines {len(bands)} bands (expected >= 5)")
    check(mins == sorted(mins, reverse=True), "band thresholds are descending")
    check(ptss == sorted(ptss, reverse=True), "band points are descending")
    check(max(ptss) == scoring["total_points"], "top band awards the full total")
    check(cfg["scoring"]["total_points"] == scoring["total_points"],
          "public and grader agree on the total point count")
    check(cfg["scoring"]["constraint_points"] == scoring["constraint_points"],
          "public and grader agree that constraints carry no points")
    check(cfg["scoring"]["optimality_points"] == scoring["optimality_points"],
          "public and grader agree on the optimality point count")

    # ---------------------------------------------------------------------------------------------
    print("\n6. ONE RUBRIC IN FOUR PLACES: README, question.md, task_config.yaml, scorer")
    rubric_docs = {"question.md": QUESTION, "public README.md": PUBLIC_README,
                   "grader README.md": GRADER_README, "root README.md": ROOT_README}
    checked_docs = 0
    for label, path in rubric_docs.items():
        if not path.exists():
            check(False, f"{label} exists")
            continue
        body = path.read_text(encoding="utf-8")
        checked_docs += 1
        check("all 100 come from the objective" in body,
              f"{label} states that all 100 points come from the objective")
        check("gate, not a scoring category" in body,
              f"{label} states that constraint satisfaction is a gate")
        # The retired rubric, in every spelling it was ever written in.
        for retired in ["Constraint satisfaction: 25 points", "Optimality: 75 points",
                        "constraint_points: 25", "25 + 75", "25+75"]:
            check(retired not in body, f"{label} does not carry the retired rubric ({retired!r})")
    check(checked_docs >= 4,
          f"at least 4 rubric documents were cross-checked (found {checked_docs})")

    # FUNCTIONAL check of the scorer: run it, do not grep it. The rubric is "feasibility is a gate
    # worth zero, all points come from the objective", so three probes pin it exactly.
    probe_constraints = constraints
    start_probe = _synthetic(molecular_weight=start.molecular_weight, logp=start.logp,
                             tpsa=start.tpsa, qed=start.qed, sa_score=start.sa_score,
                             similarity_to_start=1.0, n_alerts=0)
    feasible_no_gain = _synthetic(molecular_weight=start.molecular_weight, logp=start.logp,
                                  tpsa=start.tpsa, qed=start.qed, sa_score=start.sa_score,
                                  similarity_to_start=1.0, n_alerts=0)
    infeasible_big_gain = _synthetic(molecular_weight=1000.0, logp=start.logp, tpsa=start.tpsa,
                                     qed=float(ref["best_objective_in_window"]),
                                     sa_score=start.sa_score, similarity_to_start=1.0, n_alerts=0)
    perfect = _synthetic(molecular_weight=start.molecular_weight, logp=start.logp, tpsa=start.tpsa,
                         qed=float(ref["best_objective_in_window"]), sa_score=start.sa_score,
                         similarity_to_start=1.0, n_alerts=0)

    def total(evs):
        return final_score(start_eval=start_probe, round_evaluations=evs,
                           constraints=probe_constraints, scoring_cfg=scoring,
                           reference_cfg=ref, target_cfg=cfg["optimization_target"])

    s_none = total([feasible_no_gain])
    s_infeasible = total([infeasible_big_gain])
    s_perfect = total([perfect])
    check(s_none["total_score"] == 0,
          f"scorer: a feasible design with no gain scores 0 (got {s_none['total_score']})")
    check(s_none["constraint_points"] == 0,
          "scorer: satisfying every constraint is worth 0 points on its own")
    check(s_infeasible["total_score"] == 0,
          f"scorer: an infeasible design scores 0 however good the objective "
          f"(got {s_infeasible['total_score']})")
    check(s_perfect["total_score"] == scoring["total_points"],
          f"scorer: reaching the reference scores the full {scoring['total_points']} "
          f"(got {s_perfect['total_score']})")
    check(s_perfect["total_score"] == s_perfect["optimality_score"],
          "scorer: total_score is exactly the optimality score, with nothing added")

    print("\n7. ONE SCORER: rescore_run.py is score_run.py")
    rescore_src = RESCORE_SRC.read_text(encoding="utf-8")
    score_run_src = SCORE_RUN_SRC.read_text(encoding="utf-8")
    check("from score_run import main" in rescore_src,
          "rescore_run.py delegates to score_run.main")
    for forbidden in ["def grade(", "final_score(", "optimality_bands", "achievement"]:
        check(forbidden not in rescore_src,
              f"rescore_run.py contains no scoring logic of its own ({forbidden!r})")
    import rescore_run  # noqa: E402
    import score_run  # noqa: E402
    check(rescore_run.main is score_run.main,
          "rescore_run.main IS score_run.main (same function object, not a copy)")
    check("from scoring import final_score" in score_run_src,
          "score_run.py uses the shipped scoring module")
    # The band VALUES must be defined in exactly one place. Other files may read the key; only one
    # file may state a threshold/points pair.
    band_literal = re.compile(r"min_achievement:\s*0\.\d+")
    band_definers = [p for p in list(PUBLIC.rglob("*")) + list(GRADER.rglob("*"))
                     if p.is_file() and p.suffix in {".py", ".yaml", ".yml", ".json", ".md"}
                     and "__pycache__" not in p.parts
                     and band_literal.search(p.read_text(encoding="utf-8", errors="ignore"))]
    check(len(band_definers) == 1 and band_definers[0] == REFERENCE_CFG,
          f"the band values are written down in exactly one file "
          f"({[str(p.relative_to(ROOT)) for p in band_definers]})")

    # ---------------------------------------------------------------------------------------------
    # A provider switch declared in a config but ignored by the client is worse than not declaring
    # it: it looks configured and silently is not. Measured on this package -- Qwen3.5 without
    # enable_thinking=false returns finish_reason='length' with EMPTY content, i.e. a lost round.
    client_src = (PUBLIC / "src" / "llm_client.py").read_text(encoding="utf-8")
    check("extra_body" in client_src and 'create_kwargs["extra_body"]' in client_src,
          "llm_client.py passes extra_body through to the provider")

    print("\n8. EVALUATION ISOLATION: the public side must not disclose the answer")
    files = public_files()
    check(len(files) >= 15, f"found {len(files)} public files to scan (expected >= 15)")

    ref_smiles = ref["best_smiles_in_window"]
    ref_value = float(ref["best_objective_in_window"])
    ref_mol = Chem.MolFromSmiles(ref_smiles)
    ref_forms = {ref_smiles}
    if ref_mol is not None:
        ref_forms.add(Chem.MolToSmiles(ref_mol))
        ref_forms.add(Chem.MolToSmiles(ref_mol, kekuleSmiles=True, canonical=False))

    value_needles = [f"{ref_value:.6f}", f"{ref_value:.5f}", f"{ref_value:.4f}", f"{ref_value:.3f}"]
    band_needles = [str(int(b["points"])) for b in bands if int(b["points"]) not in (0, 100)]

    offenders_struct, offenders_value, offenders_band = [], [], []
    scanned = 0
    for p in files:
        try:
            body = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        scanned += 1
        rel = str(p.relative_to(ROOT))
        if any(form in body for form in ref_forms):
            offenders_struct.append(rel)
        if any(nd in body for nd in value_needles):
            offenders_value.append(rel)
        # Band anchors leak in the form "N / 100". A bare number appears in unrelated contexts, so
        # only the scored form is searched.
        if re.search(r"\b(" + "|".join(band_needles) + r")\s*/\s*100\b", body):
            offenders_band.append(rel)
    check(scanned >= 15, f"the isolation scan actually read files (scanned {scanned})")
    check(not offenders_struct,
          f"no public file contains the reference structure (offenders: {offenders_struct})")
    check(not offenders_value,
          f"no public file contains the reference value (offenders: {offenders_value})")
    check(not offenders_band,
          f"no public file quotes a band anchor as 'N / 100' (offenders: {offenders_band})")

    check("reference" not in cfg, "public task_config.yaml has no `reference:` block")
    check("optimality_bands" not in (cfg.get("scoring") or {}),
          "public task_config.yaml has no `optimality_bands`")
    check(str(cfg["scoring"].get("performed_by", "")).startswith("grader/"),
          "public task_config.yaml points scoring at the grader side")

    main_src = (PUBLIC / "src" / "main.py").read_text(encoding="utf-8")
    check("from scoring import" not in main_src and "\nimport scoring" not in main_src,
          "public main.py does not import the scorer")
    check("reference_target" not in main_src, "public main.py never mentions the reference file")
    check("final_report_path" not in main_src and "write_json(final_report" not in main_src
          and 'run_summary_filename' in main_src,
          "public main.py writes run_summary.json, never a final_report.json")
    check(not (PUBLIC / "src" / "scoring.py").exists(), "there is no scorer on the public side")
    check(not any("answers" in p.parts for p in files),
          "the enumeration answers directory is not on the public side")
    check(not any(p.name == "reference_target.yaml" for p in files),
          "the grader reference file is not on the public side")

    print("\n9. public and grader describe the same task")
    check(ref_file["task_id"] == cfg["task_id"], f"task_id matches ({cfg['task_id']})")
    if enum:
        check(enum["start_smiles"] == start.canonical_smiles,
              "the enumeration started from the configured molecule")
        for key, window in (("mw", "molecular_weight"), ("logp", "logp"), ("tpsa", "tpsa")):
            lo, hi = enum["windows"][key]
            check(approx(lo, constraints[window]["min"]) and approx(hi, constraints[window]["max"]),
                  f"the enumeration used the shipped {window} window")
        check(approx(enum["windows"]["similarity_min"], sim_floor),
              "the enumeration used the shipped similarity floor")

    print("\n10. regression suites")
    for suite, min_assertions in (("tests/test_input_validation.py", 80),
                                  ("tests/test_harness_robustness.py", 45)):
        proc = subprocess.run([PYTHON, suite], cwd=PUBLIC, capture_output=True, text=True)
        lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
        check(proc.returncode == 0, f"{suite} passes ({lines[-1] if lines else '(no output)'})")
        m = re.search(r"(\d+)/(\d+) assertions passed", proc.stdout)
        check(bool(m) and m.group(1) == m.group(2) and int(m.group(2)) >= min_assertions,
              f"{suite} ran a non-trivial number of assertions "
              f"({m.group(0) if m else 'none found'}; expected >= {min_assertions})")

    reports: list[tuple[Path, dict]] = []
    if args.skip_trajectories:
        print("\n11-12. SKIPPED (--skip-trajectories). This is not a valid delivery run.")
    else:
        print("\n11. shipped trajectories")
        qtext_current = QUESTION.read_text(encoding="utf-8")
        runs_dir = RUNS
        run_dirs = sorted(p for p in runs_dir.iterdir() if p.is_dir()) if runs_dir.is_dir() else []
        for run in run_dirs:
            rp = run / "final_report.json"
            if rp.is_file():
                reports.append((run, json.loads(rp.read_text(encoding="utf-8"))))
        # Guarded loop: a traversal assertion that finds nothing must FAIL, not silently pass.
        need = 2 if args.probe_only else 5
        check(len(reports) >= need,
              f"at least {need} graded trajectories exist "
              f"(found {len(reports)} in {len(run_dirs)} run dirs)")
        models = []
        # A run in which the API itself failed is not calibration evidence. It is kept in the
        # package (it is the only end-to-end proof that a mid-run provider outage costs rounds and
        # not the run), but it is excluded from every difficulty conclusion, here and in the docs.
        infra_failed = []
        for run, rep in reports:
            api_used = run / "api_config_used.json"
            model = (json.loads(api_used.read_text(encoding="utf-8")).get("model")
                     if api_used.is_file() else None)
            models.append(model)
            if (rep.get("checks", {}).get("n_llm_call_failures") or 0) > 0:
                infra_failed.append(run.name)
            score = rep.get("score") or {}
            check(isinstance(score.get("total_score"), int),
                  f"{run.name}: total_score is an integer ({score.get('total_score')})")
            for artefact in ["trajectory.jsonl", "conversation.md", "run_summary.json",
                             "llm_api_debug.jsonl", "question.md", "task_config.yaml"]:
                check((run / artefact).is_file(), f"{run.name}: {artefact} present")
            check(rep.get("checks", {}).get("n_iterations_completed") == cfg["iteration"]["max_rounds"],
                  f"{run.name}: all {cfg['iteration']['max_rounds']} rounds completed")
            # A trajectory is only calibration evidence for the CURRENT task. Editing question.md
            # and keeping older runs silently mixes two different tasks into one calibration table.
            shipped_question = run / "question.md"
            check(shipped_question.is_file()
                  and shipped_question.read_text(encoding="utf-8") == qtext_current,
                  f"{run.name}: was run against the CURRENT question.md")
            rs = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
            check(rs.get("question_sha256") == q_sha,
                  f"{run.name}: recorded question digest matches the current question.md")
            check(rs.get("prompt_version") == cfg.get("prompt_version"),
                  f"{run.name}: prompt_version {rs.get('prompt_version')!r} == "
                  f"{cfg.get('prompt_version')!r}")
            check("continuous_diagnostics" in score,
                  f"{run.name}: the report carries continuous diagnostics")
            # The trajectory must record WHY each round failed and by how much. Before this was
            # added a reviewer saw only `rejection_reasons`, which is empty for an admissible
            # molecule that merely missed a window.
            traj_lines = [json.loads(l) for l in
                          (run / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
                          if l.strip()]
            check(len(traj_lines) == cfg["iteration"]["max_rounds"],
                  f"{run.name}: trajectory has {len(traj_lines)} records")
            check(all("constraint_check" in t for t in traj_lines),
                  f"{run.name}: every trajectory record carries constraint_check margins")
            failed = [t for t in traj_lines
                      if not (t.get("constraint_check") or {}).get("satisfies_all", True)]
            # A failed round must explain itself: either a numeric distance to the bound it
            # missed, or -- for input that never parsed -- a machine-readable rejection code.
            def explained(t):
                pcs = (t["constraint_check"]["per_constraint"] or {}).values()
                return any(pc.get("shortfall") not in (None, 0.0) for pc in pcs) or \
                       any(pc.get("rejection_reasons") for pc in pcs)
            check(all(explained(t) for t in failed),
                  f"{run.name}: every failed round records a shortfall or a rejection code "
                  f"({len(failed)} failed rounds)")
        calib = ref_file.get("calibration") or {}
        required_models = list(calib.get("models_required") or [])
        n_rep = int(calib.get("replicates_required", 2))
        check(len(required_models) >= 3,
              f"the reference declares >= 3 calibration models ({required_models})")
        present = sorted({m for m in models if m})
        check(len(present) >= (1 if args.probe_only else 3),
              f"at least {1 if args.probe_only else 3} distinct model(s) ({present})")
        if args.probe_only:
            # A partial calibration still has to demonstrate the thing the instance was selected
            # for: that a non-zero score is reachable. An all-zero set is a stop signal, not a pass.
            totals = [rep["score"]["total_score"] for _, rep in reports]
            check(max(totals) > 0,
                  f"at least one run scored above zero (scores: {sorted(totals)})")
            # And it must say out loud what delivery is still waiting on.
            missing = sorted(m for m in required_models
                             if sum(1 for x in models if x == m) < n_rep)
            print(f"\n     MISSING FOR DELIVERY: {missing or 'nothing'} "
                  f"(each declared model needs {n_rep} infrastructure-clean replicates)")
            check(all(any(m in p.read_text(encoding="utf-8")
                          for p in [GRADER_NOTES, REMEDIATION] if p.exists())
                      for m in missing),
                  f"the reviewer documents name every model still missing replicates ({missing})")
        if not args.probe_only:
            for model in required_models:
                n = sum(1 for m in models if m == model)
                check(n >= n_rep, f"{model} has >= {n_rep} replicates (found {n})")
        # deepseek: the provider account ran out of balance part-way through the final calibration
        # (HTTP 402), so only one replicate is infrastructure-clean. That is a documented shortfall
        # against the n=2 plan, not a silent one -- see 整改说明2.md section 6.
        clean_ds = [run.name for run, rep in reports
                    if "deepseek" in (json.loads((run / "api_config_used.json")
                                                 .read_text(encoding="utf-8")).get("model") or "").lower()
                    and (rep.get("checks", {}).get("n_llm_call_failures") or 0) == 0]
        check(len(clean_ds) >= 1,
              f"at least one infrastructure-clean deepseek replicate exists ({len(clean_ds)})")
        for name in infra_failed:
            check(any(name in p.read_text(encoding="utf-8")
                      for p in [GRADER_NOTES, REMEDIATION] if p.exists()),
                  f"{name}: the infrastructure failure is disclosed in the reviewer documents")

        # The scores must recompute from the trajectories. This catches a final_report.json left
        # behind by an older rubric.
        from score_run import grade  # noqa: E402
        for run, rep in reports:
            fresh = grade(run, write=False, quiet=True)
            check(fresh["score"]["total_score"] == rep["score"]["total_score"],
                  f"{run.name}: stored score {rep['score']['total_score']} recomputes exactly "
                  f"(fresh {fresh['score']['total_score']})")

        # The sponsor's hard acceptance bar, asserted per replicate rather than per model.
        ds = []
        for run, rep in reports:
            api_used = run / "api_config_used.json"
            model = (json.loads(api_used.read_text(encoding="utf-8")).get("model") or ""
                     if api_used.is_file() else "")
            if "deepseek" in model.lower():
                ds.append((run, rep))
        check(len(ds) >= 1, f"at least 1 deepseek replicate is present (found {len(ds)})")
        for run, rep in ds:
            if (rep.get("checks", {}).get("n_llm_call_failures") or 0) > 0:
                continue  # provider outage, not a measurement
            check(rep["score"]["total_score"] < 50,
                  f"{run.name}: deepseek score {rep['score']['total_score']} < 50 "
                  f"(hard sponsor requirement, per replicate)")

        print("\n12. documents agree with the trajectories")
        actual = {}
        for run, rep in reports:
            actual[run.name] = rep["score"]["total_score"]
        doc_files = [p for p in [GRADER_NOTES, GRADER_README, PUBLIC_README, ROOT_README,
                                 REMEDIATION] if p.exists()]
        check(len(doc_files) >= 4, f"found {len(doc_files)} documents to cross-check")
        quoted_total = 0
        for doc in doc_files:
            body = doc.read_text(encoding="utf-8")
            for run_name, score_val in actual.items():
                for line in body.splitlines():
                    if run_name in line:
                        for m in re.finditer(r"\*\*(\d+)\s*/\s*100\*\*", line):
                            quoted_total += 1
                            check(int(m.group(1)) == score_val,
                                  f"{doc.name}: score quoted for {run_name} is {m.group(1)}, "
                                  f"actual {score_val}")
        check(quoted_total >= (2 if args.probe_only else 6),
              f"at least 6 per-run scores are quoted in the documents and were checked "
              f"(found {quoted_total})")

    print("\n13. surrogate-property positioning is stated consistently")
    positioning_docs = {
        "question.md": QUESTION,
        "public task_config.yaml": TASK_CFG,
        "public README.md": PUBLIC_README,
        "grader 审核说明.md": GRADER_NOTES,
        "root README.md": ROOT_README,
    }
    for label, path in positioning_docs.items():
        if not path.exists():
            check(False, f"{label} exists")
            continue
        body = path.read_text(encoding="utf-8").lower()
        check("surrogate" in body or "代理" in body,
              f"{label} states the surrogate-property positioning")
    for key in ["what_is_measured", "what_is_NOT_measured", "data_provenance",
                "scope_of_conclusions"]:
        check(key in (cfg.get("positioning") or {}),
              f"task_config positioning declares {key}")

    print("\n14. no leaked credentials in run outputs")
    key_like = re.compile(r"sk-[A-Za-z0-9_.\-]{16,}")
    scanned_runs = 0
    offenders = []
    for path in RUNS.rglob("*"):
        if not path.is_file():
            continue
        try:
            body = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        scanned_runs += 1
        if key_like.search(body):
            offenders.append(str(path.relative_to(ROOT)))
    check(not offenders, f"no key-shaped strings under runs/ (offenders: {offenders})")
    if not args.skip_trajectories:
        check(scanned_runs >= (10 if args.probe_only else 30),
              f"the credential scan actually read files (scanned {scanned_runs})")

    print(f"\n{checks - len(failures)}/{checks} checks passed")
    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    if args.skip_trajectories:
        print("all consistency checks passed -- BUT sections 11-12 were skipped (dev mode)")
        raise SystemExit(0)
    print("all consistency checks passed")


if __name__ == "__main__":
    main()
