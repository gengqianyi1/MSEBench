"""End-to-end self-test of the tool protocol, including every failure and recovery branch.

Why this file exists, stated plainly because it is an honest gap in the evidence:

Audit finding #5 asks that a complete trajectory show "tool selection, parsing, exception handling
and re-planning". The three shipped model trajectories show selection, parsing and re-planning --
deepseek-v4-pro, glm-4.6 and qwen-plus made 5, 5 and 7 tool calls of their own accord across 2, 4 and
3 rounds respectively -- but **none of them made a malformed call**, so none of them exercised the
error branch. Waiting for a model to make a mistake is not evidence, and provoking one by rigging the
tools would be worse.

So the error and recovery branches are demonstrated here instead, with a SCRIPTED client driving the
real harness (src/main.py, unmodified) through every advertised failure mode and out the other side.
This is a harness self-test, not a model trajectory, and it is labelled as such everywhere it appears.

What it proves, by running the real loop:
  1. all six advertised failure modes are reachable through the protocol the model actually speaks;
  2. a malformed call does not kill the round -- the harness returns a structured error and the
     client can fix the call and succeed in the SAME round (the recovery branch);
  3. spending the per-round call budget yields quota_exceeded and the round can still be submitted;
  4. ignoring "you must submit now" costs the round, is recorded as a protocol failure, and the run
     continues to the next round;
  5. the tool-layer counters in final_report.json (checks.tool_layer) add up to what happened.

Run:
    $PY scripts/test_tool_protocol.py            # exits non-zero on any failure
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import main as harness                                     # noqa: E402
from formulation import START                              # noqa: E402

OUT = ROOT / "answers" / "tool_protocol_selftest"
FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print("   [%s] %s%s" % ("ok " if ok else "FAIL", label, ("  -- " + detail) if detail else ""))
    if not ok:
        FAILURES.append(label)


class ScriptedResult:
    """The shape src/main.py expects back from llm_client.call_llm."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.ok = True
        self.empty_content = False
        self.truncated = False
        self.error = None
        self.model = "scripted-protocol-selftest"
        self.attempts = 1
        self.finish_reason = "stop"

    def to_dict(self) -> dict:
        return {"ok": True, "model": self.model, "attempts": 1, "finish_reason": "stop",
                "truncated": False, "content": self.content,
                "note": "scripted client, no API call was made"}


def tool(name, **args) -> dict:
    return {"tool": name, "arguments": args}


def submit(name: str, **overrides) -> str:
    form = dict(START)
    form.update(overrides)
    return json.dumps({"candidate": {"name": name, "rationale": "protocol self-test",
                                     "formulation": form}})


# ------------------------------------------------------------------------------------------------
# THE SCRIPT. One entry per LLM turn, in order. Every advertised failure mode appears, and every
# failure is followed by a corrected call in the SAME round wherever the budget allows.
# ------------------------------------------------------------------------------------------------
SCRIPT = [
    # --- round 1: unknown_tool + bad_arguments, then recover and submit -------------------------
    json.dumps({"tool_calls": [{"tool": "predict_cqa"},                       # unknown_tool
                               tool("validate_formulation")]}),               # bad_arguments
    json.dumps({"tool_calls": [tool("validate_formulation", formulation=dict(START))]}),  # recovery
    submit("r1_after_recovery", T2_mcc_pct=46.0),

    # --- round 2: missing_slots + off_grid, then recover and submit -----------------------------
    json.dumps({"tool_calls": [tool("validate_formulation",
                                    formulation={"T1_mcc_grade": "PH102"}),   # missing_slots
                               tool("composition_overlap",
                                    formulation={**START, "T2_mcc_pct": 33.0})]}),  # off_grid
    json.dumps({"tool_calls": [tool("composition_overlap",
                                    formulation={**START, "T8_mgst_pct": 0.4})]}),  # recovery
    submit("r2_after_recovery", T2_mcc_pct=46.0, T8_mgst_pct=0.4),

    # --- round 3: unknown_excipient, then recover and submit ------------------------------------
    json.dumps({"tool_calls": [tool("excipient_note", code="UNOBTAINIUM")]}),  # unknown_excipient
    json.dumps({"tool_calls": [tool("excipient_note", code="mgst")]}),         # recovery, lowercase
    submit("r3_after_recovery", T1_mcc_grade="PH200", T2_mcc_pct=46.0,
           T7_dis_pct=4.0, T8_mgst_pct=0.4),

    # --- round 4: spend the whole budget in one turn -> quota_exceeded, then submit anyway ------
    json.dumps({"tool_calls": [tool("excipient_note", code="CCS"),
                               tool("excipient_note", code="SSG"),
                               tool("excipient_note", code="XPVP"),
                               tool("excipient_note", code="LHPC"),
                               tool("excipient_note", code="MEG")]}),         # 5 calls, budget 4
    submit("r4_after_quota", T12_man_pct=6.0),

    # --- round 5: ignore "you must submit now" -> the round is lost, recorded as a protocol fail -
    json.dumps({"tool_calls": [tool("excipient_note", code="CIT")]}),
    json.dumps({"tool_calls": [tool("excipient_note", code="TAR")]}),
    json.dumps({"tool_calls": [tool("excipient_note", code="FUM")]}),         # turns exhausted
]


def main() -> None:
    print("=" * 98)
    print("driving the REAL harness with a scripted client -- no API call is made")
    it = iter(SCRIPT)
    n_calls = {"n": 0}

    def fake_call_llm(messages, cfg):
        n_calls["n"] += 1
        return ScriptedResult(next(it))

    # No credentials are needed and none must be required: this script has to run in a freshly
    # extracted delivery package, where configs/api_config_*.json is deliberately absent (only the
    # *.example.json templates ship). So the api-config load is stubbed too, not just call_llm.
    real_call, real_load = harness.call_llm, harness.load_json

    def fake_load_json(path):
        if "api_config" in str(path):
            return {"provider": "scripted", "model": "scripted-protocol-selftest",
                    "base_url": "<none: no API call is made>", "api_key": "<redacted>",
                    "temperature": 0.0, "max_tokens": 1}
        return real_load(path)

    harness.call_llm = fake_call_llm
    harness.load_json = fake_load_json
    try:
        run_dir = harness.run_benchmark(str(ROOT / "configs" / "default_glm.yaml"))
    finally:
        harness.call_llm, harness.load_json = real_call, real_load

    run_dir = Path(run_dir)
    report = json.loads((run_dir / "final_report.json").read_text(encoding="utf-8"))
    traj = [json.loads(ln) for ln in (run_dir / "trajectory.jsonl").read_text().splitlines() if ln]
    tl = report["checks"]["tool_layer"]

    # every failure mode seen, and where
    seen: dict[str, list[int]] = {}
    for rec in traj:
        for ti in rec.get("tool_interactions") or []:
            for res in ti.get("results") or []:
                if not res.get("ok"):
                    seen.setdefault(res["error"], []).append(rec["round"])

    print("\n1  every advertised failure mode is reachable through the real protocol")
    for kind in ("unknown_tool", "bad_arguments", "missing_slots", "off_grid",
                 "unknown_excipient", "quota_exceeded"):
        check(f"{kind} occurred", kind in seen, f"round(s) {seen.get(kind)}")

    print("\n2  a failed call does not kill the round: recovery inside the SAME round")
    for rnd in (1, 2, 3):
        rec = next(r for r in traj if r["round"] == rnd)
        results = [x for ti in rec["tool_interactions"] for x in (ti.get("results") or [])]
        had_err = any(not x.get("ok") for x in results)
        then_ok = any(x.get("ok") for x in results)
        submitted = rec["submitted_candidate"] is not None
        check(f"round {rnd}: error -> corrected call -> submission",
              had_err and then_ok and submitted,
              f"{sum(1 for x in results if not x.get('ok'))} error(s), "
              f"{sum(1 for x in results if x.get('ok'))} success(es), submitted={submitted}")
    check("the counter agrees that recovery happened",
          tl["n_rounds_recovered_after_tool_error"] == 3,
          f"n_rounds_recovered_after_tool_error = {tl['n_rounds_recovered_after_tool_error']}")

    print("\n3  spending the budget yields quota_exceeded and the round is still submittable")
    r4 = next(r for r in traj if r["round"] == 4)
    res4 = [x for ti in r4["tool_interactions"] for x in (ti.get("results") or [])]
    check("only the budgeted calls executed", sum(1 for x in res4 if x.get("ok")) == 4,
          f"{sum(1 for x in res4 if x.get('ok'))} executed of 5 requested")
    check("the 5th came back quota_exceeded and non-recoverable",
          any(x.get("error") == "quota_exceeded" and x.get("recoverable") is False for x in res4))
    check("round 4 still submitted a formulation", r4["submitted_candidate"] is not None)

    print("\n4  ignoring 'you must submit now' costs the round and the run continues")
    r5 = next(r for r in traj if r["round"] == 5)
    check("round 5 submitted nothing", r5["submitted_candidate"] is None)
    check("round 5 is recorded as a protocol failure",
          "tool budget" in (r5.get("parse_error") or ""), str(r5.get("parse_error"))[:80])
    check("the counter agrees", tl["n_tool_requests_after_budget_exhausted"] == 1,
          str(tl["n_tool_requests_after_budget_exhausted"]))
    check("the run still completed all five rounds", len(traj) == 5)
    check("per-round LLM calls are bounded by 1 + max_tool_turns = 3",
          max(r["n_llm_calls_this_round"] for r in traj) <= 3,
          str([r["n_llm_calls_this_round"] for r in traj]))

    print("\n5  the reported counters add up to what happened")
    executed = sum(len(ti.get("results") or []) for r in traj
                   for ti in (r.get("tool_interactions") or []))
    errors = sum(1 for r in traj for ti in (r.get("tool_interactions") or [])
                 for x in (ti.get("results") or []) if not x.get("ok"))
    check("n_tool_calls == results recorded in the trajectory",
          tl["n_tool_calls"] == executed, f"{tl['n_tool_calls']} vs {executed}")
    check("n_tool_errors == errors recorded in the trajectory",
          tl["n_tool_errors"] == errors, f"{tl['n_tool_errors']} vs {errors}")
    check("n_rounds_with_tool_use == 5", tl["n_rounds_with_tool_use"] == 5,
          str(tl["n_rounds_with_tool_use"]))
    check("no tool call returned a CQA, a desirability or a verdict",
          not [k for r in traj for ti in (r.get("tool_interactions") or [])
               for x in (ti.get("results") or []) for k in x
               if any(f in k.lower() for f in ("desirability", "cqa", "achievement",
                                               "in_spec", "q30"))])

    # ------------------------------------------------------------------ persist, then clean up
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("trajectory.jsonl", "final_report.json"):
        shutil.copy2(run_dir / name, OUT / name)
    (OUT / "README.md").write_text(
        "# Tool-protocol self-test (NOT a model trajectory)\n\n"
        "Produced by `scripts/test_tool_protocol.py`, which drives the unmodified harness with a\n"
        "SCRIPTED client. No API call was made and no language model was involved: `api_config_used`\n"
        "records the model as `scripted-protocol-selftest`.\n\n"
        "It exists because the three shipped model trajectories exercise tool *selection* and *use*\n"
        "but never produced a malformed call, so they do not exercise the error and recovery\n"
        "branches audit finding #5 asks about. Those branches are demonstrated here instead:\n\n"
        "| failure mode | round(s) |\n|---|---|\n"
        + "".join(f"| `{k}` | {sorted(set(v))} |\n" for k, v in sorted(seen.items()))
        + "\nRounds 1-3 each show an error followed by a corrected call and a submission in the SAME\n"
          "round. Round 4 spends the per-round budget and still submits. Round 5 ignores 'you must\n"
          "submit now', loses the round, and the run continues.\n\n"
          "This directory is grader-side only and is not one of the >= 3 required trajectories.\n",
        encoding="utf-8")
    summary = {"generated_by": "scripts/test_tool_protocol.py",
               "is_model_trajectory": False,
               "client": "scripted, no API call",
               "audit_finding": "#5 -- agent tool interaction coverage",
               "n_llm_turns_scripted": n_calls["n"],
               "failure_modes_observed": {k: sorted(set(v)) for k, v in seen.items()},
               "tool_layer_counters": tl,
               "per_round": [{"round": r["round"], "llm_calls": r["n_llm_calls_this_round"],
                              "tool_turns": len(r.get("tool_interactions") or []),
                              "submitted": r["submitted_candidate"] is not None,
                              "parse_error": r.get("parse_error")} for r in traj],
               "verdict": "FAIL" if FAILURES else "PASS"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    shutil.rmtree(run_dir)          # a scripted run must never be mistaken for a trajectory

    print("\n" + "=" * 98)
    print(f"wrote {OUT.relative_to(ROOT)}/ (summary.json, README.md, trajectory.jsonl, "
          f"final_report.json); the scripted run directory under runs/ was removed")
    if FAILURES:
        print("TOOL PROTOCOL SELF-TEST FAILED: %s" % "; ".join(FAILURES))
        raise SystemExit(1)
    print("TOOL PROTOCOL SELF-TEST PASSED -- all six failure modes reachable, recovery works in "
          "three separate rounds, quota and protocol violations behave as documented")


if __name__ == "__main__":
    main()
