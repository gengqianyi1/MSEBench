"""Designer/reviewer check: everything stated in more than one place must agree, and nothing on the
public side may disclose the answer.

Earlier problems in this series shipped with a number that had drifted: a re-scored figure left over
from a previous scoring split, a reference anchor a model had already beaten, a gate computed over an
incomplete vocabulary, and -- worst -- a README printing a retired rubric while the consistency script
reported 20/20 because it only ever read question.md. This script asserts the invariants instead, over
every file that prints a score, a band or a positioning claim.

Sections:
  1.  the starting molecule parses and satisfies every hard constraint
  2.  the grader's reference equals the enumeration's own output
  3.  the gate values equal the enumeration's own output, and each gate still passes
  4.  numeric claims in question.md match the enumeration
  5.  the scoring block is internally consistent (bands monotone, totals add up)
  6.  EVALUATION ISOLATION: nothing under public/ discloses or implies the reference or the bands
  7.  public and grader agree about the task they describe
  8.  the tool layer matches its declaration and cannot return the objective
  9.  the input-validation and tool regression suites pass
  10. the shipped trajectories are complete, and their stored scores recompute exactly
  11. every document that quotes a model score agrees with the trajectories
  12. the surrogate-property positioning is stated consistently everywhere
  13. no run output contains anything that looks like a live API key
  14. ruler_version / prompt_version are declared, agree across files, and match every shipped run
  15. the toolchain is pinned exactly, and the enumeration vocabulary declaration matches the code
  16. every report carries the fixed proxy_limitation field

Run:
    python grader/scripts/check_consistency.py
Exit status is non-zero if any assertion fails, so this is usable in a pre-delivery hook.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import unicodedata
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

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from evaluator import evaluate_smiles, check_constraints  # noqa: E402
from tools import TOOL_NAMES, TOOL_SPECS  # noqa: E402
from utils import load_yaml  # noqa: E402

PUBLIC = ROOT / "public"
GRADER = ROOT / "grader"
TASK_CFG = PUBLIC / "tasks" / "q1_lipophilic" / "task_config.yaml"
QUESTION = PUBLIC / "tasks" / "q1_lipophilic" / "question.md"
PUBLIC_README = PUBLIC / "README.md"
REFERENCE_CFG = GRADER / "reference" / "reference_target.yaml"
ENUM = GRADER / "answers" / "reference_enumeration.json"
GRADER_NOTES = GRADER / "审核说明.md"
GRADER_README = GRADER / "README.md"
REMEDIATION = ROOT / "整改说明.md"            # first review round; HISTORICAL after prompt_version 2
CURRENT_CALIBRATION = ROOT / "整改说明2.md"   # second review round; holds the live calibration table
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
        if any(part in {"__pycache__", ".ipynb_checkpoints"} for part in p.parts):
            continue
        if p.suffix == ".pyc":
            continue
        out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-trajectories", action="store_true",
                    help="DEV ONLY: skip sections 10-11. A delivery run must not use this.")
    args = ap.parse_args()

    cfg = load_yaml(TASK_CFG)
    constraints = cfg["constraints"]
    sim_cfg = constraints["similarity"]
    ref_file = load_yaml(REFERENCE_CFG)
    ref = ref_file["reference"]
    gates = ref_file["gates"]
    scoring = ref_file["scoring"]

    print("1. starting molecule")
    start = evaluate_smiles(cfg["start_smiles"], start_smiles=cfg["start_smiles"],
                            similarity_cfg=sim_cfg)
    check(start.valid, "start SMILES parses and is admissible")
    flags = check_constraints(start, constraints)
    for name, satisfied in flags.items():
        check(satisfied, f"start satisfies {name}")

    enum = None
    print("\n2. grader reference matches the enumeration")
    if not ENUM.exists():
        check(False, f"{ENUM.relative_to(ROOT)} exists (run grader/scripts/enumerate_reference_best.py)")
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

        print("\n3. gates match the enumeration, and each gate passes")
        for key in ["gate_g1_one_slot_ceiling", "gate_g1_thick_tinker_ceiling",
                    "gate_g2_safe_similarity_ceiling", "gate_g2_safe_margin", "gate_g3_n_feasible",
                    "gate_g4_checklist_only_expected_score", "gate_g4_checklist_only_p_at_least_50",
                    "best_ordinary_move_score"]:
            check(key in enum and approx(gates.get(key), enum[key], tol=1e-3),
                  f"{key} == enumeration ({gates.get(key)} vs {enum.get(key)})")
        check(enum["gate_g1_one_slot_ceiling"] < 50, "G1 one-slot ceiling < 50")
        check(enum["gate_g1_thick_tinker_ceiling"] < 50, "G1 one+two-slot ceiling < 50")
        check(enum["gate_g2_safe_similarity_ceiling"] < 50, "G2 safe-similarity ceiling < 50")
        check(enum["gate_g3_n_feasible"] >= 50, "G3 >= 50 feasible improving designs (fairness)")
        check(enum["best_ordinary_move_score"] < 50, "no ordinary medicinal-chemistry move reaches 50")
        check(enum["gate_g4_checklist_only_expected_score"] < 50,
              "G4 < 50: feasibility verification alone does not buy a pass")

        print("\n4. numbers quoted in question.md")
        text = QUESTION.read_text(encoding="utf-8") if QUESTION.exists() else ""
        check(bool(text), "question.md exists and is non-empty")
        if text:
            check(cfg["start_smiles"] in text, "question.md quotes the configured start SMILES")
            for name, key in (("molecular_weight", "mw"), ("logp", "logp")):
                lo, hi = enum["windows"][key]
                check(f"{lo}" in text and f"{hi}" in text,
                      f"question.md quotes the {name} window {lo}-{hi}")
            check(f"{enum['windows']['similarity_min']}" in text,
                  "question.md quotes the similarity floor")

            # -------------------------------------------------------------------------------------
            # The similarity-shell populations used to be printed in question.md and asserted here as
            # a required disclosure. Second-round review (15-P1 #1) ruled that class of number a
            # strategy leak: a table saying which shell holds how many feasible designs, and which
            # shell can reach the whole headroom, hands the model the answer to "how much of the edit
            # budget should I spend" before it has considered a single substituent. question.md now
            # states the shape qualitatively -- feasible designs exist across the whole range, the
            # good ones crowd against the floor -- and the counts stay grader-side. The assertion is
            # inverted accordingly: they must NOT appear.
            # -------------------------------------------------------------------------------------
            # Denying each count as a bare substring is too crude: shells hold as few as 3 and 12
            # designs, and "3" and "12" occur innocently all over the document ("12 Da above the MW
            # floor", "radius 2, nBits 2048"). The leak has a shape -- a shell range printed next to
            # its population -- so that is what is checked: no single line may carry both a band
            # boundary and that band's count. The distinctive counts are additionally denied
            # outright, since they cannot arise by coincidence.
            for band in enum["similarity_bands"]:
                n, lo, hi = str(band["n"]), str(band["lo"]), str(band["hi"])
                paired = [ln for ln in text.splitlines()
                          if (lo in ln or hi in ln) and re.search(rf"(?<![\d.]){n}(?![\d.])", ln)]
                check(not paired,
                      f"no line pairs the similarity shell [{band['lo']}, {band['hi']}) with its "
                      f"population {n} (offending lines: {paired[:2]})")
                if band["n"] >= 20:
                    check(re.search(rf"(?<![\d.]){n}(?![\d.])", text) is None,
                          f"question.md does not quote the feasible count {n} anywhere")

            # The whole shell table must be gone, not merely defanged row by row.
            check("| Similarity to the start | Feasible designs |" not in text,
                  "the similarity-shell population table has been removed from question.md")

            check(approx(enum["start_properties"]["qed"], start.qed),
                  "the start QED in the enumeration matches the evaluator")
            check(f"{start.qed:.4f}" in text, f"question.md quotes the start QED {start.qed:.4f}")

            # -------------------------------------------------------------------------------------
            # 4b. The published fourteen-move table, and the sentence that describes it.
            #
            # External review (2026-08, 田 item 2.5) found the sentence after this table --
            # "Not one of those moves earns anything." -- reading as "no single-position edit can
            # score", which contradicts the measured fact that exactly one single-slot edit is
            # feasible AND improving (it scores in the lowest band). The sentence was rescoped and
            # the exception is now stated outright. These assertions stop the old phrasing from
            # coming back and, more importantly, verify that the rescoped claim is TRUE: every row
            # of the published table really does score zero.
            # -------------------------------------------------------------------------------------
            # The table lost its QED and MW columns this round: printed together they are a
            # calibrated map of the local surface, which is the search the task measures. The
            # verification they enabled -- "every published row really does score zero" -- is kept,
            # by matching each row to its enumerated move on the move DESCRIPTION instead.
            header = "| Ordinary move | Outcome |"
            lines = text.splitlines()
            check(header in lines, "question.md still carries the ordinary-move table")
            check("| Ordinary move | QED | MW | Outcome |" not in lines,
                  "the table no longer prints per-move QED and MW")
            if header in lines:
                start_i = lines.index(header)
                rows = []
                for ln in lines[start_i + 2:]:
                    if not ln.startswith("|"):
                        break
                    rows.append([c.strip() for c in ln.split("|")[1:-1]])
                check(len(rows) == 14, f"the published table has 14 rows (found {len(rows)})")
                check("fourteen" in text.lower(),
                      "question.md refers to the table by its row count ('fourteen')")

                def move_key(s: str) -> str:
                    """Normalise a move description so the question's typography and the
                    enumeration's ASCII labels compare equal: unicode arrows and subscripts, a
                    parenthesised aside such as '(alert check)', and case all collapse away."""
                    s = unicodedata.normalize("NFKD", s).replace("→", "->").replace("‑", "-")
                    s = "".join(c for c in s if not unicodedata.combining(c))
                    s = re.sub(r"\(.*?\)", " ", s)
                    return " ".join(re.sub(r"[^a-z0-9>-]+", " ", s.lower()).split())

                by_label = {move_key(r["label"]): r for r in enum["out_of_chemotype_moves"]}
                mapped, nonzero, unmatched = 0, [], []
                for cells in rows:
                    move = by_label.get(move_key(cells[0]))
                    if move is None:
                        unmatched.append(cells[0])
                        continue
                    mapped += 1
                    if move["score"] != 0:
                        nonzero.append(f"{cells[0]} -> {move['score']}")
                check(mapped == 14,
                      f"all 14 published rows map onto an enumerated move by description "
                      f"(mapped {mapped}; unmatched {unmatched})")
                check(not nonzero,
                      f"every published row really does score zero (offenders: {nonzero})")

                # The point of dropping the columns is defeated if the same numbers survive in the
                # prose. Every enumerated move's QED and MW must be absent from the whole document.
                leaked = []
                for r in enum["out_of_chemotype_moves"] + enum["one_slot_moves"]:
                    for probe in (f"{r['qed']:.4f}", f"{r['mw']:.1f}"):
                        if probe in text and not approx(float(probe), start.qed):
                            leaked.append(f"{r['label']}: {probe}")
                check(not leaked,
                      f"no per-move QED/MW value survives anywhere in question.md "
                      f"(offenders: {leaked[:6]})")

            retired = [
                "Not one of those moves earns anything",
                "every ordinary edit makes things worse or is rejected",
                # Withdrawn this round: naming the exact multiplicity told the model that the
                # single-edit family is a dead end apart from one entry, which is search guidance
                # rather than chemistry.
                "**exactly one** is both feasible and an improvement",
            ]
            for phrase in retired:
                check(phrase not in text,
                      f"question.md no longer makes the over-broad claim {phrase!r}")
            check("Single-position edits are **not** uniformly worthless" in text,
                  "question.md states outright that single-position edits are not all worthless")
            check(enum["k_slot_ceilings"]["1"]["ceiling"] > 0,
                  f"...and that statement is true: the one-slot ceiling is "
                  f"{enum['k_slot_ceilings']['1']['ceiling']}, not 0")
            n_one_slot_scoring = sum(1 for r in enum["one_slot_moves"] if r["score"] > 0)
            check(n_one_slot_scoring >= 1,
                  f"question.md's 'some one-step change is feasible and an improvement' is true "
                  f"({n_one_slot_scoring} one-slot move(s) score above zero)")
            check(enum["best_ordinary_move_score"] == 0,
                  f"no out-of-chemotype ordinary move scores at all "
                  f"({enum['best_ordinary_move_score']})")

    print("\n5. scoring block is internally consistent")
    check(scoring["constraint_points"] == 0,
          "constraint_points == 0 (feasibility is a gate, not credit)")
    check(scoring["optimality_points"] == scoring["total_points"],
          "optimality_points == total_points")
    bands = scoring["optimality_bands"]
    mins = [float(b["min_achievement"]) for b in bands]
    ptss = [int(b["points"]) for b in bands]
    check(mins == sorted(mins, reverse=True), "band thresholds are descending")
    check(ptss == sorted(ptss, reverse=True), "band points are descending")
    check(max(ptss) == scoring["total_points"], "top band awards the full total")
    check(cfg["scoring"]["total_points"] == scoring["total_points"],
          "public and grader agree on the total point count")
    check(cfg["scoring"]["constraint_points"] == scoring["constraint_points"],
          "public and grader agree that constraints carry no points")

    # ---------------------------------------------------------------------------------------------
    print("\n6. EVALUATION ISOLATION: the public side must not disclose the answer")
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
    # The reference's SIMILARITY to the start is reference-derived too, and it is the one number that
    # would tell a model exactly where inside the 0.40-0.45 shell to aim. It was NOT scanned for
    # until 2026-08, and the omission bit immediately: the first draft of that revision put it into
    # three public files while every existing assertion still passed. Scanned now.
    ref_sim = float(ref["reference_similarity_to_start"])
    sim_needles = [f"{ref_sim:.3f}", f"{ref_sim:.4f}", f"{ref_sim:.5f}", f"{ref_sim:.6f}"]

    offenders_struct, offenders_value, offenders_band, offenders_sim = [], [], [], []
    for p in files:
        try:
            body = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = str(p.relative_to(ROOT))
        if any(form in body for form in ref_forms):
            offenders_struct.append(rel)
        if any(n in body for n in value_needles):
            offenders_value.append(rel)
        if any(n in body for n in sim_needles):
            offenders_sim.append(rel)
        # Band anchors leak in the form "N / 100". A bare number appears in unrelated contexts, so
        # only the scored form is searched.
        if re.search(r"\b(" + "|".join(band_needles) + r")\s*/\s*100\b", body):
            offenders_band.append(rel)

    check(not offenders_struct,
          f"no public file contains the reference structure (offenders: {offenders_struct})")
    check(not offenders_value,
          f"no public file contains the reference value (offenders: {offenders_value})")
    check(not offenders_sim,
          f"no public file contains the reference's similarity to the start "
          f"(offenders: {offenders_sim})")
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
    check(not (PUBLIC / "src" / "scoring.py").exists(), "there is no scorer on the public side")
    check(not any("answers" in p.parts for p in files),
          "the enumeration answers directory is not on the public side")
    check(not any(p.name == "reference_target.yaml" for p in files),
          "the grader reference file is not on the public side")

    print("\n7. public and grader describe the same task")
    check(ref_file["task_id"] == cfg["task_id"], f"task_id matches ({cfg['task_id']})")
    if enum:
        check(enum["start_smiles"] == start.canonical_smiles,
              "the enumeration started from the configured molecule")
        for key, window in (("mw", "molecular_weight"), ("logp", "logp")):
            lo, hi = enum["windows"][key]
            check(approx(lo, constraints[window]["min"]) and approx(hi, constraints[window]["max"]),
                  f"the enumeration used the shipped {window} window")

    print("\n8. the tool layer matches its declaration")
    tool_cfg = cfg["tools"]
    check(sorted(tool_cfg["available"]) == sorted(TOOL_NAMES),
          f"task_config tool list == tools.py ({sorted(TOOL_NAMES)})")
    check(len(TOOL_NAMES) >= 2, f"at least two tools are offered ({len(TOOL_NAMES)})")
    qtext = QUESTION.read_text(encoding="utf-8")
    for name in TOOL_NAMES:
        check(f"`{name}`" in qtext, f"question.md documents the tool {name}")
    for code in tool_cfg["failure_modes_returned_to_model"]:
        check(f"`{code}`" in qtext, f"question.md documents the failure code {code}")
    check(str(tool_cfg["max_calls_per_round"]) in qtext,
          f"question.md states the per-round call budget ({tool_cfg['max_calls_per_round']})")
    check(str(tool_cfg["max_tool_messages_per_round"]) in qtext,
          f"question.md states the per-round tool-message limit "
          f"({tool_cfg['max_tool_messages_per_round']})")
    # The hard rule: the tool layer must not touch the objective or the descriptors behind it.
    tools_src = (PUBLIC / "src" / "tools.py").read_text(encoding="utf-8")
    for forbidden in ["QED.qed", "ev.qed", "ev.molecular_weight", "ev.logp", "ev.tpsa",
                      "ev.sa_score", "Descriptors.MolWt", "Crippen.MolLogP", "CalcTPSA"]:
        check(forbidden not in tools_src, f"tools.py never reads {forbidden}")
    check(all("QED" not in spec["returns"] for spec in TOOL_SPECS),
          "no tool advertises returning QED")
    check("no tool returns QED" in qtext or "No tool returns QED" in qtext,
          "question.md states plainly that no tool returns QED")

    print("\n9. regression suites")
    for suite in ["tests/test_input_validation.py", "tests/test_tools.py"]:
        proc = subprocess.run([PYTHON, suite], cwd=PUBLIC, capture_output=True, text=True)
        lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
        check(proc.returncode == 0, f"{suite} passes ({lines[-1] if lines else '(no output)'})")
        m = re.search(r"(\d+)/(\d+) assertions passed", proc.stdout)
        check(bool(m) and m.group(1) == m.group(2) and int(m.group(2)) >= 50,
              f"{suite} ran a non-trivial number of assertions "
              f"({m.group(0) if m else 'none found'})")

    reports: list[tuple[Path, dict]] = []
    if args.skip_trajectories:
        print("\n10-11. SKIPPED (--skip-trajectories). This is not a valid delivery run.")
    else:
        print("\n10. shipped trajectories")
        sys.path.insert(0, str(GRADER / "scripts"))
        from score_run import PROXY_LIMITATION, grade  # noqa: E402
        qtext_current = QUESTION.read_text(encoding="utf-8")
        runs_dir = RUNS
        run_dirs = sorted(p for p in runs_dir.iterdir() if p.is_dir()) if runs_dir.is_dir() else []
        for run in run_dirs:
            rp = run / "final_report.json"
            if rp.is_file():
                reports.append((run, json.loads(rp.read_text(encoding="utf-8"))))
        # Guarded loop: a traversal assertion that finds nothing must FAIL, not silently pass.
        check(len(reports) >= 3,
              f"at least 3 graded trajectories exist (found {len(reports)} in "
              f"{len(run_dirs)} run dirs)")
        # No half-finished run directory may ship: every directory under runs/ must be graded, or
        # the delivery contains a trajectory nothing has checked.
        check(len(reports) == len(run_dirs),
              f"every run directory is graded ({len(reports)} reports for {len(run_dirs)} dirs; "
              f"ungraded: {[p.name for p in run_dirs if not (p / 'final_report.json').is_file()]})")
        models = []
        for run, rep in reports:
            api_used = run / "api_config_used.json"
            model = (json.loads(api_used.read_text(encoding="utf-8")).get("model")
                     if api_used.is_file() else None)
            models.append(model)
            score = rep.get("score") or {}
            check(isinstance(score.get("total_score"), int),
                  f"{run.name}: total_score is an integer ({score.get('total_score')})")
            for artefact in ["trajectory.jsonl", "conversation.md", "run_summary.json",
                             "llm_api_debug.jsonl", "question.md", "task_config.yaml"]:
                check((run / artefact).is_file(), f"{run.name}: {artefact} present")
            check(rep.get("checks", {}).get("n_iterations_completed") == cfg["iteration"]["max_rounds"],
                  f"{run.name}: all {cfg['iteration']['max_rounds']} rounds completed")
            # A trajectory is only calibration evidence for the CURRENT task. Editing question.md and
            # keeping older runs silently mixes two different tasks into one calibration table -- it
            # happened during this revision, so it is asserted rather than remembered.
            shipped_question = run / "question.md"
            check(shipped_question.is_file()
                  and shipped_question.read_text(encoding="utf-8") == qtext_current,
                  f"{run.name}: was run against the CURRENT question.md")
            tu = rep.get("tool_usage") or {}
            check(tu.get("total_tool_calls", 0) > 0,
                  f"{run.name}: the trajectory contains real tool calls "
                  f"({tu.get('total_tool_calls')})")
            check("continuous_diagnostics" in score,
                  f"{run.name}: the report carries continuous diagnostics")
            # Version stamps. The identical-question.md check above catches a stale prompt by its
            # text; this catches it by its number, which is the form a downstream consumer reads.
            check(rep.get("prompt_version") == cfg.get("prompt_version"),
                  f"{run.name}: prompt_version {rep.get('prompt_version')} == package "
                  f"{cfg.get('prompt_version')}")
            check(rep.get("ruler_version") == cfg.get("ruler_version"),
                  f"{run.name}: ruler_version {rep.get('ruler_version')!r} == package "
                  f"{cfg.get('ruler_version')!r}")
            check("prompt_version_warning" not in rep,
                  f"{run.name}: the grader raised no prompt-version warning")
            env = rep.get("environment") or {}
            check((env.get("run") or {}).get("rdkit") is not None,
                  f"{run.name}: the report records the RDKit version that ran the model "
                  f"({(env.get('run') or {}).get('rdkit')})")
            # F4: the fixed proxy-limitation field, verbatim.
            check(rep.get("proxy_limitation") == PROXY_LIMITATION,
                  f"{run.name}: proxy_limitation is present and identical to the constant")
        check(len({m for m in models if m}) >= 3,
              f"at least 3 distinct models ({sorted({m for m in models if m})})")

        # The scores must recompute from the trajectories. This catches a final_report.json left
        # behind by an older rubric.
        for run, rep in reports:
            fresh = grade(run, write=False, quiet=True)
            check(fresh["score"]["total_score"] == rep["score"]["total_score"],
                  f"{run.name}: stored score {rep['score']['total_score']} recomputes exactly "
                  f"(fresh {fresh['score']['total_score']})")

        # The sponsor's hard acceptance bar.
        ds = []
        for run, rep in reports:
            api_used = run / "api_config_used.json"
            model = (json.loads(api_used.read_text(encoding="utf-8")).get("model") or ""
                     if api_used.is_file() else "")
            if "deepseek" in model.lower():
                ds.append((run, rep))
        check(len(ds) >= 1, "a deepseek trajectory is present")
        for run, rep in ds:
            check(rep["score"]["total_score"] < 50,
                  f"{run.name}: deepseek score {rep['score']['total_score']} < 50 "
                  f"(hard sponsor requirement)")

        print("\n11. documents agree with the trajectories")
        # Indexed BY RUN, never by model. Keying this dict by model name silently discarded every
        # replicate but the last one -- it is the single most repeated bug in this series, and with
        # n>1 per model it produces a table that looks complete and is wrong.
        per_run: list[tuple[str, str, int]] = []
        for run, rep in reports:
            api_used = run / "api_config_used.json"
            model = (json.loads(api_used.read_text(encoding="utf-8")).get("model")
                     if api_used.is_file() else None)
            per_run.append((run.name, model or "?", rep["score"]["total_score"]))
        scores_by_model: dict[str, list[int]] = {}
        for _, model, sc in per_run:
            scores_by_model.setdefault(model, []).append(sc)
        for model, scs in sorted(scores_by_model.items()):
            print(f"       {model:18s} n={len(scs)}  scores={sorted(scs)}")

        # The current calibration document must carry ONE LINE PER RUN, with that run's own score.
        # This is what makes "no aggregation by model" mechanical instead of a promise.
        check(CURRENT_CALIBRATION.exists(),
              f"{CURRENT_CALIBRATION.name} exists (the current calibration table)")
        if CURRENT_CALIBRATION.exists():
            cal_lines = CURRENT_CALIBRATION.read_text(encoding="utf-8").splitlines()
            matched_runs = 0
            for run_name, model, sc in per_run:
                hits = [ln for ln in cal_lines if run_name in ln]
                ok_row = any(re.search(r"\*\*" + str(sc) + r"\s*/\s*100\*\*", ln) for ln in hits)
                check(bool(hits) and ok_row,
                      f"{CURRENT_CALIBRATION.name}: run {run_name} has its own row quoting "
                      f"**{sc} / 100** ({len(hits)} line(s) mention it)")
                matched_runs += bool(hits) and ok_row
            check(matched_runs == len(per_run),
                  f"every one of the {len(per_run)} runs is listed individually "
                  f"(matched {matched_runs})")

        # Any other current document that quotes a score on a model line must quote a score that a
        # replicate of that model actually got.
        doc_files = [p for p in [GRADER_NOTES, GRADER_README, PUBLIC_README, CURRENT_CALIBRATION]
                     if p.exists()]
        check(len(doc_files) >= 3, f"found {len(doc_files)} current documents to cross-check")
        quoted_total = 0
        for doc in doc_files:
            body = doc.read_text(encoding="utf-8")
            for model, scs in scores_by_model.items():
                for line in body.splitlines():
                    if model in line:
                        for m in re.finditer(r"\*\*(\d+)\s*/\s*100\*\*", line):
                            quoted_total += 1
                            check(int(m.group(1)) in scs,
                                  f"{doc.name}: score quoted for {model} is {m.group(1)}, "
                                  f"and the replicates actually scored {sorted(scs)}")
        check(quoted_total >= len(per_run),
              f"at least {len(per_run)} model scores are quoted in the current documents and were "
              f"checked (found {quoted_total})")

        # The first-round remediation report is HISTORICAL: its table describes trajectories that
        # were deleted when prompt_version went 1 -> 2. It is exempt from the equality check only
        # because it says so in its own text, and that is asserted rather than assumed.
        if REMEDIATION.exists():
            hist = REMEDIATION.read_text(encoding="utf-8")
            check("prompt_version = 1" in hist and "已作废" in hist,
                  f"{REMEDIATION.name} declares its score table retired (prompt_version = 1)")
            check(CURRENT_CALIBRATION.name in hist,
                  f"{REMEDIATION.name} points at {CURRENT_CALIBRATION.name} for the current table")

    print("\n12. surrogate-property positioning is stated consistently")
    positioning_docs = {
        "question.md": QUESTION,
        "public task_config.yaml": TASK_CFG,
        "public README.md": PUBLIC_README,
        "grader 审核说明.md": GRADER_NOTES,
        "整改说明.md": REMEDIATION,
        "整改说明2.md": CURRENT_CALIBRATION,
    }
    for label, path in positioning_docs.items():
        if not path.exists():
            check(False, f"{label} exists")
            continue
        body = path.read_text(encoding="utf-8").lower()
        check("surrogate" in body or "代理" in body,
              f"{label} states the surrogate-property positioning")
    qbody = QUESTION.read_text(encoding="utf-8").lower()
    for phrase in ["no potency", "none is simulated", "not a measure of candidate quality"]:
        check(phrase in qbody, f"question.md disclaims non-existent data ({phrase!r})")
    check("surrogate-property" in cfg["task_name"].lower(),
          f"task_name declares the positioning ({cfg['task_name'][:60]}...)")
    for key in ["what_is_measured", "what_is_NOT_measured", "data_provenance",
                "scope_of_conclusions"]:
        check(key in (cfg.get("positioning") or {}),
              f"task_config positioning declares {key}")

    print("\n13. no leaked credentials in run outputs")
    key_like = re.compile(r"sk-[A-Za-z0-9_.\-]{16,}")
    scanned = 0
    offenders = []
    for path in RUNS.rglob("*"):
        if not path.is_file():
            continue
        try:
            body = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        scanned += 1
        if key_like.search(body):
            offenders.append(str(path.relative_to(ROOT)))
    check(not offenders, f"no key-shaped strings under runs/ (offenders: {offenders})")
    if not args.skip_trajectories:
        check(scanned >= 15, f"the credential scan actually read files (scanned {scanned})")

    # ---------------------------------------------------------------------------------------------
    print("\n14. version discipline (ruler_version / prompt_version)")
    ruler = str(cfg.get("ruler_version", ""))
    check(bool(ruler), f"public task_config declares a ruler_version ({ruler!r})")
    check(str(cfg["scoring"].get("ruler_version", "")) == ruler,
          "public task_config.scoring.ruler_version == top-level ruler_version")
    check(str(ref_file.get("ruler_version", "")) == ruler,
          f"grader reference_target ruler_version == public ({ref_file.get('ruler_version')!r})")
    check(str(scoring.get("ruler_version", "")) == ruler,
          "grader reference_target.scoring.ruler_version == public")
    pv = cfg.get("prompt_version")
    check(isinstance(pv, int) and pv >= 1, f"public task_config declares an integer prompt_version ({pv})")
    check(bool(cfg.get("prompt_version_changed_on")),
          f"task_config records when the prompt last changed ({cfg.get('prompt_version_changed_on')})")
    main_src_v = (PUBLIC / "src" / "main.py").read_text(encoding="utf-8")
    check('"prompt_version"' in main_src_v and '"ruler_version"' in main_src_v,
          "the run harness stamps both versions into run_summary.json")

    # ---------------------------------------------------------------------------------------------
    print("\n15. frozen toolchain and enumeration vocabulary")
    req_root = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    req_public = (PUBLIC / "requirements.txt").read_text(encoding="utf-8")
    check(req_root == req_public,
          "requirements.txt at the package root and under public/ are identical")
    pins = dict(re.findall(r"^([A-Za-z0-9_.\-]+)==([0-9][^\s#]*)\s*$", req_root, re.M))
    for pkg in ["rdkit", "openai", "PyYAML", "numpy"]:
        check(pkg in pins, f"requirements.txt pins {pkg} exactly (== {pins.get(pkg)})")
    check(not re.search(r"^[A-Za-z0-9_.\-]+\s*>=", req_root, re.M),
          "no dependency is left on a floating >= range")

    def _norm(v: str) -> tuple:
        return tuple(int(x) for x in re.findall(r"\d+", str(v)))

    env_block = ref_file.get("environment") or {}
    check(_norm(pins.get("rdkit", "")) == _norm(env_block.get("rdkit", "x")),
          f"the pinned rdkit {pins.get('rdkit')} == the grader's recorded environment "
          f"{env_block.get('rdkit')}")
    repro = cfg.get("reproducibility") or {}
    check(_norm(repro.get("rdkit", "y")) == _norm(pins.get("rdkit", "")),
          f"the public reproducibility block records the same rdkit ({repro.get('rdkit')})")
    for key in ["standardization_protocol", "fingerprint_protocol", "determinism", "python"]:
        check(bool(repro.get(key)), f"public task_config.reproducibility declares {key}")
    fp_text = str(repro.get("fingerprint_protocol", ""))
    check(f"radius={sim_cfg['radius']}" in fp_text and f"fpSize={sim_cfg['n_bits']}" in fp_text,
          f"the documented fingerprint protocol matches the constraint "
          f"(radius {sim_cfg['radius']}, {sim_cfg['n_bits']} bits)")
    check("GetMorganGenerator" in (PUBLIC / "src" / "evaluator.py").read_text(encoding="utf-8"),
          "the evaluator really uses the documented Morgan generator")
    check("rdMolStandardize.Cleanup" in (PUBLIC / "src" / "validation.py").read_text(encoding="utf-8"),
          "the validator really uses the documented standardization call")

    # The vocabulary declaration must equal BOTH the enumeration output and the live module, so a
    # slot list cannot be edited without re-running the enumeration.
    vocab = ref_file.get("enumeration_vocabulary") or {}
    check(bool(vocab.get("version")), f"the grader declares an enumeration vocabulary version "
                                      f"({vocab.get('version')})")
    for key in ["chemotype", "covers", "does_not_cover", "consequence", "defined_in"]:
        check(bool(vocab.get(key)), f"enumeration_vocabulary declares {key}")
    if enum:
        evocab = enum.get("vocabulary") or {}
        check(bool(evocab),
              "the enumeration output carries a vocabulary block (re-run enumerate_reference_best.py)")
        for key in ["version", "cartesian_product", "n_skipped_nonsensical", "n_valid_designs",
                    "n_out_of_chemotype_moves", "n_rejected_by_admissibility"]:
            check(vocab.get(key) == evocab.get(key),
                  f"enumeration_vocabulary.{key}: grader {vocab.get(key)} == enumeration "
                  f"{evocab.get(key)}")
        check(dict(vocab.get("slot_sizes") or {}) == dict(evocab.get("slot_sizes") or {}),
              f"enumeration_vocabulary.slot_sizes matches the enumeration {evocab.get('slot_sizes')}")
    # ... and against the code itself, not just against a file the code wrote.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "enumerate_reference_best", GRADER / "scripts" / "enumerate_reference_best.py")
    enum_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enum_mod)
    live_sizes = {n: len(s) for n, s in zip(enum_mod.SLOT_NAMES, enum_mod.SLOTS)}
    check(dict(vocab.get("slot_sizes") or {}) == live_sizes,
          f"enumeration_vocabulary.slot_sizes == the slot lists in the script itself ({live_sizes})")
    check(vocab.get("version") == enum_mod.VOCABULARY_VERSION,
          f"enumeration_vocabulary.version == VOCABULARY_VERSION in the script "
          f"({enum_mod.VOCABULARY_VERSION!r})")
    check(vocab.get("n_out_of_chemotype_moves") == len(enum_mod.ORDINARY_MOVES),
          f"the declared out-of-chemotype move count == the script's list "
          f"({len(enum_mod.ORDINARY_MOVES)})")

    # ---------------------------------------------------------------------------------------------
    print("\n16. the fixed proxy-limitation field")
    sys.path.insert(0, str(GRADER / "scripts"))
    from score_run import PROXY_LIMITATION as PL  # noqa: E402
    for key in ["headline", "proxies", "must_not_be_interpreted_as", "forbidden_claim",
                "supported_claim", "data_provenance"]:
        check(key in PL, f"proxy_limitation declares {key}")
    forbidden = PL["forbidden_claim"].lower()
    for word in ["qed", "efficacy", "admet", "safety", "synthetic route"]:
        check(word in forbidden, f"the forbidden-claim sentence names {word!r}")
    for proxy in ["QED", "SA_Score", "Morgan/Tanimoto similarity"]:
        check(proxy in PL["proxies"], f"proxy_limitation.proxies covers {proxy}")
    check(len(PL["must_not_be_interpreted_as"]) >= 5,
          f"proxy_limitation lists at least five forbidden interpretations "
          f"({len(PL['must_not_be_interpreted_as'])})")
    score_src = (GRADER / "scripts" / "score_run.py").read_text(encoding="utf-8")
    check('"proxy_limitation": PROXY_LIMITATION' in score_src,
          "score_run.py writes the constant into every report rather than a per-run copy")

    print(f"\n{checks - len(failures)}/{checks} checks passed")
    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    if args.skip_trajectories:
        print("all consistency checks passed -- BUT sections 10-11 were skipped (dev mode)")
        raise SystemExit(0)
    print("all consistency checks passed")


if __name__ == "__main__":
    main()
