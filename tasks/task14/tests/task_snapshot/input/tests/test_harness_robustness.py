"""Regression tests for the run loop: abnormal input must cost a round, never the run.

Second-round review, 田双燕 §2.3, asked specifically whether abnormal values can crash the benchmark
or terminate it early ("对于异常值是否会引起 benchmark 的崩溃/提前终止"). That is a property of the
loop, not of the evaluator, so it is tested here by driving `main.run_benchmark` end to end against a
scripted LLM stub -- no API key, no network.

Groups:
  1. the parser rejects every malformed shape and always raises ParseError, never a bare exception;
  2. the feedback builder survives invalid, empty and duplicate candidates and never leaks a verdict;
  3. a full five-round run in which EVERY round is pathological still completes five rounds, writes a
     trajectory with five records, and writes a run summary with no score in it;
  4. a run in which the same candidate is submitted five times completes and is reported as one
     distinct molecule submitted five times;
  5. the public harness contains no scorer (evaluation isolation, asserted from the source).

Run standalone (no pytest needed):
    python tests/test_harness_robustness.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

import main as harness  # noqa: E402
from evaluator import evaluate_smiles  # noqa: E402
from feedback import generate_feedback  # noqa: E402
from parser import ParseError, parse_candidate  # noqa: E402
from utils import load_yaml  # noqa: E402

TASK_CFG_PATH = ROOT / "tasks" / "q1_multiobjective" / "task_config.yaml"
TASK_CFG = load_yaml(TASK_CFG_PATH)
START = TASK_CFG["start_smiles"]
CONSTRAINTS = TASK_CFG["constraints"]
SIM_CFG = CONSTRAINTS["similarity"]
MAX_ROUNDS = int(TASK_CFG["iteration"]["max_rounds"])

_failures: list[str] = []
_checks = 0


def check(condition: bool, message: str) -> None:
    global _checks
    _checks += 1
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


# =================================================================================================
def test_parser_rejects_every_malformed_shape() -> None:
    print("\n1. parser failure modes")
    bad = {
        "empty string": "",
        "whitespace only": "   ",
        "plain prose": "Sure! Here is my molecule: CCO",
        "truncated JSON": '{"candidate": {"name": "a", "smiles": "CCO"',
        "JSON array instead of object": '[{"candidate": {}}]',
        "candidate is not an object": '{"candidate": "CCO"}',
        "missing candidate key": '{"molecule": {"smiles": "CCO"}}',
        "missing smiles field": '{"candidate": {"name": "a", "rationale": "b"}}',
        "missing name field": '{"candidate": {"smiles": "CCO", "rationale": "b"}}',
        "missing rationale field": '{"candidate": {"name": "a", "smiles": "CCO"}}',
        "smiles is not a string": '{"candidate": {"name": "a", "smiles": 42, "rationale": "b"}}',
        "smiles is null": '{"candidate": {"name": "a", "smiles": null, "rationale": "b"}}',
        "smiles is a list": '{"candidate": {"name": "a", "smiles": ["CCO"], "rationale": "b"}}',
        "nothing but a fence": "```json\n```",
        "None instead of text": None,
    }
    for label, text in bad.items():
        try:
            parse_candidate(text)
            check(False, f"ParseError raised for {label}")
        except ParseError:
            check(True, f"ParseError raised for {label}")
        except Exception as exc:  # noqa: BLE001 - this is the thing being ruled out
            check(False, f"ParseError (not {type(exc).__name__}) raised for {label}")

    good = {
        "bare JSON object": '{"candidate": {"name": "a", "smiles": "CCO", "rationale": "b"}}',
        "fenced JSON": '```json\n{"candidate": {"name": "a", "smiles": "CCO", "rationale": "b"}}\n```',
        "JSON with prose around it":
            'Here you go:\n{"candidate": {"name": "a", "smiles": "CCO", "rationale": "b"}}\nThanks!',
        "extra unknown keys":
            '{"candidate": {"name": "a", "smiles": "CCO", "rationale": "b", "extra": 1}, "note": 2}',
        "whitespace inside smiles":
            '{"candidate": {"name": "a", "smiles": "  CCO  ", "rationale": "b"}}',
    }
    for label, text in good.items():
        try:
            got = parse_candidate(text)
            check(got["smiles"] == "CCO", f"parsed correctly: {label}")
        except ParseError as exc:
            check(False, f"parsed correctly: {label} (raised {exc})")


def test_feedback_survives_abnormal_evaluations() -> None:
    print("\n2. feedback builder on abnormal candidates")
    start_eval = evaluate_smiles(START, start_smiles=START, similarity_cfg=SIM_CFG)
    cases = {
        "unparseable SMILES": "not_a_smiles",
        "empty SMILES": "",
        "salt": f"{START}.Cl",
        "bare ion": "[Na+]",
        "duplicate of the start molecule": START,
    }
    for label, smi in cases.items():
        ev = evaluate_smiles(smi, start_smiles=START, similarity_cfg=SIM_CFG)
        fb = generate_feedback(evaluation=ev, start_eval=start_eval, constraints=CONSTRAINTS,
                               best_so_far=None)
        check(isinstance(fb, dict) and "status" in fb, f"feedback is a dict for {label}")
        check(json.dumps(fb) is not None, f"feedback is JSON-serialisable for {label}")
        # Reduced bandwidth: never a per-constraint verdict, never a success verdict.
        body = json.dumps(fb).lower()
        check("constraint_flags" not in body and "success" not in body and "pass" not in body,
              f"feedback leaks no verdict for {label}")


class ScriptedLLM:
    """Stand-in for `llm_client.call_llm`, replaying a fixed list of responses."""

    class Result:
        def __init__(self, content, ok=True, empty=False, truncated=False, error=None):
            self.content = content
            self.ok = ok
            self.empty_content = empty
            self.truncated = truncated
            self.error = error
            self.model = "scripted-stub"
            self.attempts = 1
            self.finish_reason = "stop"
            self.max_tokens_used = 0

        def to_dict(self):
            return {"content": self.content, "ok": self.ok, "model": self.model,
                    "empty_content": self.empty_content, "truncated": self.truncated,
                    "error": self.error, "finish_reason": self.finish_reason,
                    "attempts": self.attempts, "max_tokens_used": self.max_tokens_used}

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = 0

    def __call__(self, messages, llm_cfg):
        item = self.scripted[min(self.calls, len(self.scripted) - 1)]
        self.calls += 1
        if item is None:
            return self.Result("", ok=False, empty=True, error="scripted LLM failure")
        return self.Result(item)


def _run_with(scripted, tmp: Path):
    """Drive a whole benchmark run with a scripted LLM and return (run_dir, summary)."""
    cfg = {
        "llm": {"api_config_file": str(tmp / "api.json")},
        "runtime": {"run_dir": str(tmp / "runs"), "random_seed": 42},
        "task": {"task_dir": str(ROOT / "tasks" / "q1_multiobjective"),
                 "question_file": "question.md", "task_config_file": "task_config.yaml"},
        "parsing": {"allow_markdown_fence": True},
        "logging": {"trajectory_filename": "trajectory.jsonl",
                    "llm_debug_filename": "llm_api_debug.jsonl",
                    "conversation_markdown_filename": "conversation.md",
                    "run_summary_filename": "run_summary.json"},
    }
    import yaml
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (tmp / "api.json").write_text(json.dumps({
        "provider": "openai_compatible", "model": "scripted-stub",
        "api_key": "PUT_YOUR_KEY_HERE", "base_url": "http://localhost",
    }), encoding="utf-8")

    real = harness.call_llm
    harness.call_llm = ScriptedLLM(scripted)
    try:
        run_dir = harness.run_benchmark(str(cfg_path))
    finally:
        harness.call_llm = real
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    return run_dir, summary


def test_pathological_run_completes() -> None:
    print("\n3. a run in which every round is pathological still completes")
    tmp = Path(tempfile.mkdtemp(prefix="a_robustness_"))
    try:
        scripted = [
            "this is not JSON at all",                                     # parse failure
            '{"candidate": {"name": "x", "smiles": "", "rationale": "r"}}',  # empty SMILES
            '{"candidate": {"name": "x", "smiles": "Xx1ccccc1", "rationale": "r"}}',  # unparseable
            None,                                                          # LLM failure
            f'{{"candidate": {{"name": "x", "smiles": "{START}.Cl", "rationale": "r"}}}}',  # salt
        ]
        run_dir, summary = _run_with(scripted, tmp)
        check(summary["checks"]["n_iterations_completed"] == MAX_ROUNDS,
              f"all {MAX_ROUNDS} rounds ran ({summary['checks']['n_iterations_completed']})")
        traj = [json.loads(l) for l in (run_dir / "trajectory.jsonl").read_text(
            encoding="utf-8").splitlines() if l.strip()]
        check(len(traj) == MAX_ROUNDS, f"trajectory has {MAX_ROUNDS} records ({len(traj)})")
        check(all("evaluation" in r for r in traj),
              "every trajectory record carries its evaluation")
        check(summary["checks"]["n_valid"] == 0, "no pathological candidate was scored as valid")
        check(summary["checks"]["n_parse_failures"] >= 2,
              f"parse failures were counted ({summary['checks']['n_parse_failures']})")
        check(summary["checks"]["n_llm_call_failures"] >= 1,
              f"the LLM failure was counted ({summary['checks']['n_llm_call_failures']})")
        check("multiple_fragments" in summary["checks"]["rejection_reason_codes"],
              f"the salt was rejected with a reason code "
              f"({summary['checks']['rejection_reason_codes']})")
        # Evaluation isolation: the harness output must not contain a score.
        body = json.dumps(summary)
        check("total_score" not in body and "achievement" not in body
              and "optimality" not in body,
              "run_summary.json contains no score, achievement or optimality field")
        check((run_dir / "final_report.json").exists() is False,
              "the harness did not write a final_report.json (only the grader may)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_duplicate_candidates_are_harmless() -> None:
    print("\n4. five identical submissions still complete five rounds")
    tmp = Path(tempfile.mkdtemp(prefix="a_dup_"))
    try:
        one = f'{{"candidate": {{"name": "same", "smiles": "{START}", "rationale": "r"}}}}'
        run_dir, summary = _run_with([one] * MAX_ROUNDS, tmp)
        check(summary["checks"]["n_iterations_completed"] == MAX_ROUNDS,
              f"all {MAX_ROUNDS} rounds ran with a repeated candidate")
        check(summary["checks"]["n_valid"] == MAX_ROUNDS,
              "each repeat was evaluated rather than skipped")
        smis = {c["canonical_smiles"] for c in summary["candidates"]}
        check(len(smis) == 1, f"all repeats canonicalise to one molecule ({len(smis)})")
        check(summary["checks"]["n_parse_failures"] == 0, "no parse failure on the repeat run")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_public_side_has_no_scorer() -> None:
    print("\n5. evaluation isolation on the public side")
    src = ROOT / "src"
    check(not (src / "scoring.py").exists(), "there is no scoring.py under public/src")
    main_src = (src / "main.py").read_text(encoding="utf-8")
    check("from scoring import" not in main_src and "\nimport scoring" not in main_src,
          "main.py does not import a scorer")
    check("reference_target" not in main_src, "main.py never mentions the grader reference file")
    task_cfg_text = TASK_CFG_PATH.read_text(encoding="utf-8")
    check("reference" not in TASK_CFG, "the public task config has no reference block")
    check("optimality_bands" not in task_cfg_text,
          "the public task config publishes no score bands")
    check("best_smiles_in_window" not in task_cfg_text,
          "the public task config carries no reference structure")


TESTS = [
    test_parser_rejects_every_malformed_shape,
    test_feedback_survives_abnormal_evaluations,
    test_pathological_run_completes,
    test_duplicate_candidates_are_harmless,
    test_public_side_has_no_scorer,
]


def main() -> None:
    for fn in TESTS:
        fn()
    print(f"\n{_checks - len(_failures)}/{_checks} assertions passed "
          f"across {len(TESTS)} test groups")
    if _checks < 45:
        print(f"FAILED: only {_checks} assertions ran; the suite is expected to run at least 45")
        raise SystemExit(1)
    if _failures:
        print("FAILED:")
        for f in _failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("all harness-robustness regression tests passed")


if __name__ == "__main__":
    main()
