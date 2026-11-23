#!/usr/bin/env python3
"""Budget-bypass and isolation tests for task 17 (review item 17-P1 #1).

FIRST, THE THREAT MODEL THE REVIEW ASSUMED, AND WHY IT DOES NOT APPLY HERE.

The review asked for "API 鉴权、五轮调用的原子计数、并发/重放、超时/断连……防止并发/批量请求绕过预算"
-- authentication, atomic counting of the five calls, concurrency and replay, guarding against a
parallel or batched request slipping past the budget. That is the right list for an architecture in
which the agent can *call* an evaluation endpoint. This task is not that architecture, and saying so
precisely matters more than writing tests against a threat that does not exist:

  * There is no evaluation API. The model never invokes the evaluator. It emits one JSON object per
    turn to the LLM endpoint; the loop in `src/main.py` parses it, and the loop -- not the model --
    decides whether that turn was a tool request or a submission, and calls the simulator itself.
  * The round counter is a Python `for round_idx in range(1, max_rounds + 1)` in a single-threaded
    process. There is nothing to count atomically and nothing to race: the model cannot issue two
    turns at once, because the loop does not read a second message until it has finished the first.
  * Replay is meaningless for the same reason. Re-sending an identical message does not re-enter the
    loop; it is simply the content of the next turn, which still consumes exactly one turn.
  * Authentication is the LLM endpoint's, and the credential reaches the process from the harness
    environment. The model never sees it.

What CAN be bypassed, and therefore what is tested here, is the **per-round tool phase**: the model
does control how many tool turns it asks for and what it puts in them. So the tests below drive the
tool layer directly and assert that the quota, the leak guard and the error contract hold under the
inputs a model could actually produce.

    python private_grader/test_budget_isolation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1]


def load_modules():
    private_src = TASK_DIR / "private_data" / "src"
    public_src = TASK_DIR / "input" / "src"
    for path in (private_src, public_src):
        if not path.is_dir():
            raise SystemExit(f"missing {path}")
    sys.path[:0] = [str(private_src), str(public_src)]
    import tools                                                          # noqa: E402
    import design_grid                                                    # noqa: E402
    return tools, design_grid


def main() -> int:
    tools, design_grid = load_modules()
    failures: list[str] = []

    def check(condition: bool, what: str, detail: str = "") -> None:
        print(f"  {'ok  ' if condition else 'FAIL'} {what}"
              + (f"\n         {detail}" if detail and not condition else ""))
        if not condition:
            failures.append(what)

    quota = tools.MAX_TOOL_CALLS_PER_TURN
    baseline = dict(design_grid.START) if hasattr(design_grid, "START") else None
    if baseline is None:
        raise SystemExit("design_grid.START not found; cannot build a legal probe process")

    print(f"\nA. per-turn call quota (declared: {quota})")

    legal = {"tool": "change_budget_audit", "arguments": {"process": baseline}}
    burst = tools.run_tool_calls([legal] * (quota + 5))
    executed = [r for r in burst.get("results", []) if "error" not in r]
    refused = [r for r in burst.get("results", []) if r.get("error") == "quota_exceeded"]
    check(len(executed) <= quota,
          f"a burst of {quota + 5} calls in ONE turn executes at most {quota}",
          f"executed {len(executed)}")
    check(len(refused) >= 1,
          "the calls past the quota come back as quota_exceeded rather than being silently dropped",
          f"refused {len(refused)}")
    check(len(burst.get("results", [])) == quota + 5,
          "every requested call gets a result object, so the model can see what was refused",
          f"{len(burst.get('results', []))} result(s) for {quota + 5} request(s)")

    print("\nB. the quota is per turn and does not accumulate a debt across turns")
    again = tools.run_tool_calls([legal])
    check(all("error" not in r or r.get("error") != "quota_exceeded"
              for r in again.get("results", [])),
          "a fresh turn starts with a fresh quota after a previous turn exhausted it")

    print("\nC. the leak guard: no tool result may carry a response, D, achievement or a score")
    probes = [
        {"tool": "change_budget_audit", "arguments": {"process": baseline}},
        {"tool": "design_space_check", "arguments": {"process": baseline}},
    ]
    for probe in probes:
        out = tools.run_tool_calls([probe])
        flat = repr(out).lower()
        for forbidden in sorted(tools.FORBIDDEN_RESULT_KEYS):
            check(f'"{forbidden}"' not in flat and f"'{forbidden}'" not in flat,
                  f"{probe['tool']} does not return {forbidden!r}")

    print("\nD. malformed and hostile calls are data, not crashes")
    hostile = [
        ({"tool": "no_such_tool", "arguments": {}}, "unknown_tool"),
        ({"not_a_tool_object": 1}, None),
        ({"tool": "change_budget_audit"}, None),
        ({"tool": "change_budget_audit", "arguments": {"process": {"S1_bridging_liquid": "NOPE"}}}, None),
        ({"tool": "change_budget_audit", "arguments": {"process": None}}, None),
    ]
    for call, expected in hostile:
        try:
            out = tools.run_tool_calls([call])
            results = out.get("results", [])
            ok = bool(results) and "error" in results[0]
            if expected:
                ok = ok and results[0]["error"] == expected
            detail = str(results[:1])
        except Exception as exc:                                          # noqa: BLE001
            ok, detail = False, f"raised {type(exc).__name__}: {exc}"
        check(ok, f"{str(call)[:58]:58s} -> structured error", detail)

    print("\nE. the architecture claim, asserted rather than asserted-in-prose")
    loop_source = (TASK_DIR / "input" / "src" / "main.py").read_text(encoding="utf-8")
    check("for round_idx in range(1, max_rounds + 1)" in loop_source,
          "the round budget is a single-threaded for-loop over max_rounds, not a served counter")
    for forbidden in ("threading", "asyncio", "multiprocessing", "concurrent.futures"):
        check(f"import {forbidden}" not in loop_source,
              f"the loop does not import {forbidden} -- there is no concurrency to race")
    check("Flask" not in loop_source and "FastAPI" not in loop_source
          and "socket" not in loop_source,
          "the loop exposes no network service the model could call directly")

    print(f"\n{'FAILED: ' + str(len(failures)) if failures else 'all budget-isolation tests passed'}")
    for item in failures:
        print(f"  - {item}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
