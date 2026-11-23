"""Regression tests for the diagnostic tool layer (review finding 4).

Two things have to hold and keep holding:

  1. **The tools must not turn the objective into a computable quantity.** No tool may return QED, and
     none may return the numeric value of any of QED's eight component descriptors. This is checked
     mechanically: every tool is run on several molecules, every number in every response is
     collected, and the run fails if any of them matches that molecule's MW, LogP, TPSA or QED. Key
     names for the count-valued components are checked too. This is the assertion that stops a future
     "just return the properties, it's more useful" edit from handing away the answer.

  2. **The failure branches are real.** Unknown tool name, malformed call object, missing or wrongly
     typed arguments, an inadmissible molecule and an exhausted quota each have to come back as a
     structured, recoverable error -- and the failures that are the model's own fault have to consume
     budget, so that mismanaging the quota has a cost.

Run standalone (no pytest needed):
    python tests/test_tools.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from evaluator import evaluate_smiles  # noqa: E402
from tools import (  # noqa: E402
    E_INVALID_ARGS,
    E_INVALID_MOLECULE,
    E_MALFORMED_CALL,
    E_QUOTA,
    E_UNKNOWN_TOOL,
    TOOL_NAMES,
    TOOL_SPECS,
    ToolSession,
)
from utils import load_yaml  # noqa: E402

TASK_CFG = load_yaml(ROOT / "tasks" / "q1_lipophilic" / "task_config.yaml")
START = TASK_CFG["start_smiles"]
CONSTRAINTS = TASK_CFG["constraints"]
SIM_CFG = CONSTRAINTS["similarity"]
TOOL_CFG = TASK_CFG["tools"]

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


def session(max_calls: int = 6) -> ToolSession:
    return ToolSession(start_smiles=START, constraints=CONSTRAINTS,
                       similarity_cfg=SIM_CFG, max_calls=max_calls)


def _numeric_leaves(obj, key="<root>", out=None):
    """Every numeric leaf in a nested response, paired with the key it sits under.

    Booleans are excluded: `constraint_checklist` is booleans by design, and a boolean cannot be
    inverted into a property value.
    """
    if out is None:
        out = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append((str(key).lower(), float(obj)))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _numeric_leaves(v, k, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numeric_leaves(v, key, out)
    return out


# The ONLY keys a tool response may carry a number under. This is a whitelist rather than a
# blacklist on purpose: a future edit that starts returning `"molecular_weight": 357.45` fails here
# without anyone having to remember to add it to a list of forbidden names.
ALLOWED_NUMERIC_KEYS = {
    "similarity_to_start",   # the edit budget -- a declared, allowed diagnostic
    "n_alerts",              # structural-alert count -- a declared, allowed diagnostic
    "n_components",          # admissibility
    "net_formal_charge",     # admissibility
    "calls_used",            # protocol bookkeeping
    "calls_remaining",       # protocol bookkeeping
}


# =================================================================================================
def test_catalogue_matches_config() -> None:
    print("\n1. the tool catalogue agrees with the task config")
    check(sorted(TOOL_NAMES) == sorted(TOOL_CFG["available"]),
          f"tools.py and task_config.yaml list the same tools ({sorted(TOOL_NAMES)})")
    check(len(TOOL_SPECS) >= 2,
          f"at least two tools are offered (there are {len(TOOL_SPECS)})")
    check(TOOL_CFG["max_calls_per_round"] == 6 and TOOL_CFG["max_tool_messages_per_round"] == 2,
          "the configured budget is 6 calls over 2 messages")
    for spec in TOOL_SPECS:
        check("smiles" in spec["arguments"], f"{spec['name']} takes a smiles argument")


def test_no_tool_leaks_the_objective() -> None:
    """The load-bearing assertion: no tool response contains the objective or its components."""
    print("\n2. no tool returns QED or any QED component value")
    probes = [
        START,
        "CC(=O)Nc1ccc(C)c(S(=O)(=O)N(C)CCCN2CCCC2)c1",
        "CC(=O)Nc1ccc(Cl)c(S(=O)(=O)N(C)CCCN2CCCC2)c1",
        "Oc1ccc(F)c(S(=O)(=O)N(C)CCCN2CCCC2)c1",
        "c1ccccc1",
    ]
    for smi in probes:
        ev = evaluate_smiles(smi, start_smiles=START, similarity_cfg=SIM_CFG)
        if not ev.valid:
            continue
        # Degenerate descriptor values (0 and 1) are skipped in the value-collision check because
        # they legitimately collide with counts such as n_components=1 or n_alerts=0. The key
        # whitelist below is what actually pins the contract, so nothing is lost.
        forbidden_values = {
            "qed": ev.qed, "molecular_weight": ev.molecular_weight,
            "logp": ev.logp, "tpsa": ev.tpsa, "sa_score": ev.sa_score,
        }
        for name in TOOL_NAMES:
            s = session()
            resp = s.call({"tool": name, "arguments": {"smiles": smi}})
            leaves = _numeric_leaves(resp)
            nums = [v for _, v in leaves]

            hits = [f"{k}={v}" for k, v in forbidden_values.items()
                    if v is not None and float(v) not in (0.0, 1.0)
                    and any(abs(n - float(v)) < 1e-6 for n in nums)]
            check(not hits,
                  f"{name} on {smi[:26]}... returns no descriptor value (hits={hits})")

            stray = sorted({k for k, _ in leaves} - ALLOWED_NUMERIC_KEYS)
            check(not stray,
                  f"{name} on {smi[:26]}... carries numbers only under whitelisted keys "
                  f"(stray={stray})")

    # constraint_checklist is the closest call: prove it really is booleans only.
    s = session()
    resp = s.call({"tool": "constraint_checklist", "arguments": {"smiles": START}})
    result = resp["result"]
    non_bool = {k: v for k, v in result.items() if not isinstance(v, bool) and k != "note"}
    check(not non_bool, f"constraint_checklist returns booleans only (non-bool: {non_bool})")
    check("qed" not in json.dumps(resp).lower(),
          "the word 'qed' does not appear anywhere in a constraint_checklist response")


def test_tools_return_what_they_promise() -> None:
    print("\n3. each tool returns its documented diagnostic")
    s = session(max_calls=20)
    r = s.call({"tool": "validate_structure", "arguments": {"smiles": START}})
    check(r["ok"] and r["result"]["admissible"] is True, "validate_structure accepts the start")
    check(r["result"]["n_components"] == 1 and r["result"]["net_formal_charge"] == 0,
          "validate_structure reports one neutral component")

    r = s.call({"tool": "validate_structure", "arguments": {"smiles": f"{START}.Cl"}})
    check(r["ok"] and r["result"]["admissible"] is False, "validate_structure reports a salt as inadmissible")
    check("multiple_fragments" in r["result"]["rejection_reasons"],
          "validate_structure returns the reason code for a salt")

    r = s.call({"tool": "similarity_to_start", "arguments": {"smiles": START}})
    check(r["ok"] and abs(r["result"]["similarity_to_start"] - 1.0) < 1e-9,
          "similarity_to_start returns 1.0 for the start molecule itself")

    r = s.call({"tool": "similarity_to_start", "arguments": {"smiles": "c1ccccc1"}})
    check(r["ok"] and r["result"]["similarity_to_start"] < 0.2,
          f"similarity_to_start returns a low value for benzene ({r['result']['similarity_to_start']})")

    r = s.call({"tool": "structural_alerts", "arguments": {"smiles": START}})
    check(r["ok"] and r["result"]["n_alerts"] == 0, "structural_alerts finds none on the start")

    r = s.call({"tool": "structural_alerts", "arguments": {"smiles": "Nc1ccc(F)c(S(=O)(=O)N(C)CCCN2CCCC2)c1"}})
    check(r["ok"] and r["result"]["n_alerts"] >= 1,
          f"structural_alerts flags the free aniline ({r['result']['alerts']})")

    r = s.call({"tool": "constraint_checklist", "arguments": {"smiles": START}})
    check(r["ok"] and r["result"]["all_satisfied"] is True,
          "constraint_checklist passes the start molecule")

    r = s.call({"tool": "constraint_checklist", "arguments": {"smiles": "c1ccccc1"}})
    check(r["ok"] and r["result"]["all_satisfied"] is False,
          "constraint_checklist fails benzene")


def test_failure_branches() -> None:
    print("\n4. failure branches are real and structured")
    s = session()
    r = s.call({"tool": "compute_qed", "arguments": {"smiles": START}})
    check(not r["ok"] and r["error"]["code"] == E_UNKNOWN_TOOL,
          "asking for a non-existent tool returns unknown_tool")
    check(r["error"].get("available_tools") == TOOL_NAMES,
          "unknown_tool tells the model which tools exist")
    check(s.calls_used == 1, "a wrong tool name still costs one call")

    r = s.call({"tool": "constraint_checklist", "arguments": {}})
    check(not r["ok"] and r["error"]["code"] == E_INVALID_ARGS,
          "a missing smiles argument returns invalid_arguments")
    check(s.calls_used == 2, "bad arguments still cost one call")

    r = s.call({"tool": "constraint_checklist", "arguments": {"smiles": 42}})
    check(not r["ok"] and r["error"]["code"] == E_INVALID_ARGS,
          "a non-string smiles returns invalid_arguments")

    r = s.call({"tool": "constraint_checklist", "arguments": "CCO"})
    check(not r["ok"] and r["error"]["code"] == E_INVALID_ARGS,
          "a non-object arguments field returns invalid_arguments")

    r = s.call("constraint_checklist")
    check(not r["ok"] and r["error"]["code"] == E_MALFORMED_CALL,
          "a bare string instead of a call object returns malformed_tool_call")

    r = s.call({"arguments": {"smiles": START}})
    check(not r["ok"] and r["error"]["code"] == E_MALFORMED_CALL,
          "a call object with no tool name returns malformed_tool_call")

    s2 = session()
    r = s2.call({"tool": "constraint_checklist", "arguments": {"smiles": "CC(=O)[O-].[Na+]"}})
    check(not r["ok"] and r["error"]["code"] == E_INVALID_MOLECULE,
          "an inadmissible molecule returns invalid_molecule")
    check("multiple_fragments" in (r["error"].get("rejection_reasons") or []),
          "invalid_molecule carries the reason codes")
    check(r["error"].get("hint"), "invalid_molecule suggests a recovery step")

    # validate_structure is the exception: an inadmissible input is a normal answer for it, not an
    # error, so the model always has a way to diagnose a string without burning an error.
    r = s2.call({"tool": "validate_structure", "arguments": {"smiles": "CC(=O)[O-].[Na+]"}})
    check(r["ok"], "validate_structure answers normally for an inadmissible input")


def test_quota() -> None:
    print("\n5. the per-round quota binds")
    s = session(max_calls=3)
    for i in range(3):
        r = s.call({"tool": "similarity_to_start", "arguments": {"smiles": START}})
        check(r["ok"], f"call {i + 1} of 3 succeeds")
        check(r["calls_remaining"] == 3 - (i + 1),
              f"calls_remaining is reported correctly after call {i + 1} ({r['calls_remaining']})")
    r = s.call({"tool": "similarity_to_start", "arguments": {"smiles": START}})
    check(not r["ok"] and r["error"]["code"] == E_QUOTA, "the 4th call is refused with quota_exceeded")
    check(s.calls_used == 3, "a refused call does not consume more budget")
    r = s.call({"tool": "validate_structure", "arguments": {"smiles": START}})
    check(not r["ok"] and r["error"]["code"] == E_QUOTA,
          "every tool is refused once the budget is gone")

    print("\n6. batch calls share one budget")
    s = session(max_calls=2)
    responses = s.call_batch([
        {"tool": "similarity_to_start", "arguments": {"smiles": START}},
        {"tool": "structural_alerts", "arguments": {"smiles": START}},
        {"tool": "constraint_checklist", "arguments": {"smiles": START}},
    ])
    check(len(responses) == 3, "one response per requested call")
    check(responses[0]["ok"] and responses[1]["ok"], "the first two calls in the batch run")
    check(not responses[2]["ok"] and responses[2]["error"]["code"] == E_QUOTA,
          "the third call in the batch is refused by the shared budget")

    r = s.call_batch("not a list")
    check(not r[0]["ok"] and r[0]["error"]["code"] == E_MALFORMED_CALL,
          "a non-list tool_calls payload returns malformed_tool_call")
    r = s.call_batch([])
    check(not r[0]["ok"] and r[0]["error"]["code"] == E_MALFORMED_CALL,
          "an empty tool_calls list returns malformed_tool_call")


def test_log_is_auditable() -> None:
    print("\n7. the session log records what a reviewer needs")
    s = session()
    s.call({"tool": "similarity_to_start", "arguments": {"smiles": START}})
    s.call({"tool": "constraint_checklist", "arguments": {"smiles": "c1ccccc1"}})
    check(len(s.log) == 2, "each executed call is logged")
    check(all("tool" in e and "arguments" in e and "response" in e for e in s.log),
          "each log entry records the tool, the arguments and the response")


TESTS = [
    test_catalogue_matches_config,
    test_no_tool_leaks_the_objective,
    test_tools_return_what_they_promise,
    test_failure_branches,
    test_quota,
    test_log_is_auditable,
]


def main() -> None:
    for fn in TESTS:
        fn()
    print(f"\n{_checks - len(_failures)}/{_checks} assertions passed across {len(TESTS)} test groups")
    if _checks < 60:
        print(f"FAILED: only {_checks} assertions ran; the suite is expected to run at least 60")
        raise SystemExit(1)
    if _failures:
        print("FAILED:")
        for f in _failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("all tool-layer regression tests passed")


if __name__ == "__main__":
    main()
