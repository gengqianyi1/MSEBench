"""The diagnostic tool layer the model may call before committing a candidate.

Review finding (2026-07, issue 4): the earlier harness called one fixed evaluator for the model, so
a trajectory contained no tool selection, no argument construction, no failure and no recovery. The
model now drives a small JSON tool protocol itself: which tool, on which molecule, how many times,
and what to do when a call comes back as an error.

HARD DESIGN RULE -- the tools must not turn the objective into a computable quantity.
The optimisation target is RDKit QED, a fitted geometric mean of eight desirability functions over
molecular weight, LogP, TPSA, H-bond donors, H-bond acceptors, rotatable bonds, aromatic rings and a
structural-alert term. Therefore NO tool returns QED, and NO tool returns the numeric value of any of
those eight descriptors. What the tools return is diagnostic only:

  * whether a string is an admissible molecule at all, and why not;
  * how far a molecule has drifted from the starting point (the edit budget);
  * which structural alerts it triggers;
  * a pass/fail checklist against the published constraint windows -- booleans, never values.

A boolean "molecular weight is inside its window" cannot be inverted into a molecular weight, and
three booleans out of eight components cannot be assembled into QED. The objective still has to be
reasoned about; only feasibility bookkeeping is delegated.

Every call is quota-limited, so choosing which molecule to spend a probe on is itself part of the
task. Errors are real returns, not exceptions: an unknown tool name, a malformed argument object, an
inadmissible molecule and an exhausted quota each come back as a structured error the model has to
read and recover from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from evaluator import check_constraints, evaluate_smiles
from validation import REASON_TEXT, validate_smiles_input

# ---------------------------------------------------------------------------------------------
# Tool catalogue. The descriptions here are the single source of truth: question.md quotes them and
# scripts/check_consistency.py asserts that the two agree.
# ---------------------------------------------------------------------------------------------
TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "validate_structure",
        "purpose": "Check whether a SMILES string is an admissible candidate at all: parses, one "
                   "component, neutral, allowed elements, no isotope labels or atom maps.",
        "arguments": {"smiles": "string, required"},
        "returns": "ok flag, canonical SMILES, component count, net formal charge, element list, "
                   "and the rejection reason codes when it is not admissible",
    },
    {
        "name": "similarity_to_start",
        "purpose": "Morgan/Tanimoto similarity of a molecule to the starting molecule -- how much of "
                   "the edit budget the design has spent.",
        "arguments": {"smiles": "string, required"},
        "returns": "similarity (float, 0-1)",
    },
    {
        "name": "structural_alerts",
        "purpose": "Run the combined PAINS + Brenk + NIH structural-alert catalogs on a molecule.",
        "arguments": {"smiles": "string, required"},
        "returns": "n_alerts (int) and the alert names",
    },
    {
        "name": "constraint_checklist",
        "purpose": "Pass/fail against every published hard-constraint window for a molecule.",
        "arguments": {"smiles": "string, required"},
        # Phrased without naming the objective on purpose: check_consistency.py asserts that no
        # spec's `returns` text contains the objective's name at all, so that a future edit which
        # starts advertising it is caught mechanically rather than by reading.
        "returns": "one boolean per constraint plus all_satisfied. BOOLEANS ONLY -- no property "
                   "values, and nothing about the objective",
    },
]

TOOL_NAMES = [spec["name"] for spec in TOOL_SPECS]

# Error codes. Stable strings; the regression tests assert on them.
E_UNKNOWN_TOOL = "unknown_tool"
E_MALFORMED_CALL = "malformed_tool_call"
E_INVALID_ARGS = "invalid_arguments"
E_INVALID_MOLECULE = "invalid_molecule"
E_QUOTA = "quota_exceeded"


@dataclass
class ToolSession:
    """Per-round tool budget and dispatcher.

    `max_calls` is the number of individual tool calls allowed in one round, across all tools and
    across however many tool-call messages the model chooses to send. A call that fails because the
    model got the tool name or the arguments wrong still consumes budget -- a probe that was spent is
    spent. A call refused because the budget is already gone does not, since there is nothing left to
    consume.
    """

    start_smiles: str
    constraints: Dict[str, Any]
    similarity_cfg: Optional[Dict[str, Any]] = None
    max_calls: int = 6
    calls_used: int = 0
    log: List[Dict[str, Any]] = field(default_factory=list)

    # -- envelope helpers ----------------------------------------------------------------------
    def _envelope(self, tool: Optional[str], ok: bool, payload: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"tool": tool, "ok": ok}
        out.update(payload)
        out["calls_used"] = self.calls_used
        out["calls_remaining"] = max(0, self.max_calls - self.calls_used)
        return out

    def _error(self, tool: Optional[str], code: str, message: str, **extra) -> Dict[str, Any]:
        err: Dict[str, Any] = {"code": code, "message": message}
        err.update(extra)
        return self._envelope(tool, False, {"error": err})

    # -- dispatch ------------------------------------------------------------------------------
    def call(self, request: Any) -> Dict[str, Any]:
        """Execute one tool call object and return its envelope. Never raises."""
        if not isinstance(request, dict):
            return self._error(None, E_MALFORMED_CALL,
                               "each entry of tool_calls must be an object with 'tool' and "
                               "'arguments'", available_tools=TOOL_NAMES)

        name = request.get("tool") or request.get("name")
        if not isinstance(name, str) or not name.strip():
            return self._error(None, E_MALFORMED_CALL,
                               "missing string field 'tool'", available_tools=TOOL_NAMES)
        name = name.strip()

        if self.calls_used >= self.max_calls:
            return self._error(name, E_QUOTA,
                               f"the tool budget for this round ({self.max_calls} calls) is "
                               f"exhausted; submit a candidate instead")

        if name not in TOOL_NAMES:
            self.calls_used += 1
            return self._error(name, E_UNKNOWN_TOOL,
                               f"no tool named {name!r}", available_tools=TOOL_NAMES)

        args = request.get("arguments", request.get("args"))
        if args is None:
            args = {}
        if not isinstance(args, dict):
            self.calls_used += 1
            return self._error(name, E_INVALID_ARGS, "'arguments' must be an object",
                               expected={"smiles": "string"})

        unknown = sorted(k for k in args if k != "smiles")
        smiles = args.get("smiles")
        if not isinstance(smiles, str) or not smiles.strip():
            self.calls_used += 1
            return self._error(name, E_INVALID_ARGS,
                               "argument 'smiles' is required and must be a non-empty string",
                               expected={"smiles": "string"},
                               unexpected_arguments=unknown or None)

        self.calls_used += 1
        result = self._run(name, smiles.strip())
        self.log.append({
            "tool": name,
            "arguments": {"smiles": smiles.strip()},
            "ok": bool(result.get("ok")),
            "response": result,
        })
        return result

    def call_batch(self, requests: Any) -> List[Dict[str, Any]]:
        if not isinstance(requests, list):
            return [self._error(None, E_MALFORMED_CALL, "'tool_calls' must be a list",
                                available_tools=TOOL_NAMES)]
        if not requests:
            return [self._error(None, E_MALFORMED_CALL, "'tool_calls' was empty",
                                available_tools=TOOL_NAMES)]
        return [self.call(r) for r in requests]

    # -- individual tools ----------------------------------------------------------------------
    def _run(self, name: str, smiles: str) -> Dict[str, Any]:
        check = validate_smiles_input(smiles)

        if name == "validate_structure":
            if check.ok:
                return self._envelope(name, True, {"result": {
                    "admissible": True,
                    "canonical_smiles": check.canonical_smiles,
                    "n_components": check.n_fragments,
                    "net_formal_charge": check.net_charge,
                    "elements": check.elements,
                }})
            return self._envelope(name, True, {"result": {
                "admissible": False,
                "rejection_reasons": check.reasons,
                "explanations": [REASON_TEXT.get(r, r) for r in check.reasons],
                "n_components": check.n_fragments,
                "net_formal_charge": check.net_charge,
                "elements": check.elements,
            }})

        # Every other tool needs a real molecule; an inadmissible input is a genuine error return.
        if not check.ok:
            return self._error(name, E_INVALID_MOLECULE,
                               check.message() or "inadmissible molecule",
                               rejection_reasons=check.reasons,
                               hint="call validate_structure to see why before spending more probes")

        ev = evaluate_smiles(smiles, start_smiles=self.start_smiles,
                             similarity_cfg=self.similarity_cfg)
        if not ev.valid:  # pragma: no cover - validation already passed
            return self._error(name, E_INVALID_MOLECULE, ev.error or "evaluation failed")

        if name == "similarity_to_start":
            return self._envelope(name, True, {"result": {
                "similarity_to_start": ev.similarity_to_start,
                "fingerprint": "Morgan radius 2, 2048 bits, Tanimoto",
            }})

        if name == "structural_alerts":
            return self._envelope(name, True, {"result": {
                "n_alerts": ev.n_alerts,
                "alerts": ev.alerts,
                "catalogs": ["PAINS", "Brenk", "NIH"],
            }})

        if name == "constraint_checklist":
            flags = check_constraints(ev, self.constraints)
            return self._envelope(name, True, {"result": {
                **{k: bool(v) for k, v in flags.items()},
                "all_satisfied": all(flags.values()),
                # The starting molecule satisfies everything, so `all_satisfied` marks eligibility,
                # not merit. Saying so plainly is a restatement of the published rules, not extra
                # information -- and on the first calibration of the tool layer two models read
                # `all_satisfied: true` as success and resubmitted the starting molecule.
                "note": "booleans only; no property values and nothing about the objective. "
                        "all_satisfied means the design is ELIGIBLE to be scored, not that it "
                        "improved anything -- the starting molecule also satisfies everything",
            }})

        # unreachable: name was validated against TOOL_NAMES
        return self._error(name, E_UNKNOWN_TOOL, f"no tool named {name!r}",
                           available_tools=TOOL_NAMES)


def tool_catalogue_markdown() -> str:
    """Render the catalogue for the prompt so the doc and the code cannot drift apart."""
    lines = []
    for spec in TOOL_SPECS:
        args = ", ".join(f"`{k}` ({v})" for k, v in spec["arguments"].items())
        lines.append(f"- **`{spec['name']}`** — {spec['purpose']}\n"
                     f"  - arguments: {args}\n"
                     f"  - returns: {spec['returns']}")
    return "\n".join(lines)
