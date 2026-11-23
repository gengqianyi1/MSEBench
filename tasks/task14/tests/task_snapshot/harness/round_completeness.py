"""Was the multi-round protocol actually completed? One implementation, four tasks.

§16.6 requires a multi-round task to record, separately from the score, whether every planned round
produced a real candidate -- "最终推荐成功不能掩盖中间轮失败". The round-3 audit (刘丞军 P1) found
this suite deciding after the fact that "losing fewer than two rounds still counts as a complete
run". A degradation rule chosen once the scores are visible can always be placed where it flatters
them, so the rule now lives in each task's PUBLIC `task_config.yaml` under
`iteration.round_completeness`, is declared before any run, and is applied here identically for
every task and every model.

Why this is a module and not twenty lines in each of the four graders: the four tasks already ship
same-named modules and have drifted apart once before. A rule copied four times is a rule that will
eventually mean four things.

The verdict is deliberately NOT the score and does not touch it:

  * SCORE stays whatever the task's own scorer says. Best-of-N over a trajectory is non-decreasing
    in N, so a run that lost a round can only have scored the same or less than it would have with
    all of them -- its score is a valid LOWER BOUND, which is a measurement, not a discard reason.
    §10 forbids letting an infrastructure loss disappear into a dash or a zero.
  * COMPLETENESS is a separate verdict that travels with the score wherever the score goes, so a
    reader cannot pick up the number without the label.
  * RE-RUN is an operational trigger read from the same declared config, not a judgement made here.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

MISSING_CAUSE = "LLM_CALL_FAILURE_OR_UNPARSEABLE"


def assess(rounds_with_candidate: Iterable[Any],
           observed_round_numbers: Iterable[Any],
           iteration_cfg: Mapping[str, Any] | None,
           run_summary: Mapping[str, Any] | None = None) -> dict:
    """Build the completeness verdict.

    `rounds_with_candidate` is the round number of every round that carried something the scorer
    could read -- a SMILES, a formulation, a process design. What that candidate then scored is a
    scientific question and none of this function's business.

    `observed_round_numbers` is every round the trajectory holds a record for, so a round that is
    absent entirely can be told apart from a round that was recorded and came back empty.
    """
    cfg = dict((iteration_cfg or {}).get("round_completeness") or {})
    declared_raw = (iteration_cfg or {}).get("max_rounds")
    observed = [r for r in observed_round_numbers if r is not None]
    with_candidate = sorted({r for r in rounds_with_candidate if r is not None})
    declared = int(declared_raw) if declared_raw else (max(observed) if observed else len(with_candidate))

    missing: list[dict] = []
    seen = set(observed)
    for n in range(1, declared + 1):
        if n not in seen:
            missing.append({"round": n, "cause": "NO_ROUND_RECORD"})
        elif n not in with_candidate:
            missing.append({"round": n, "cause": MISSING_CAUSE})

    # The runner knows whether the provider never answered or answered with something unparseable;
    # the grader does not. Use its codes when it recorded them rather than guessing.
    codes = (run_summary or {}).get("round_failures") or []
    if isinstance(codes, list):
        for m in missing:
            for c in codes:
                if isinstance(c, Mapping) and c.get("round") == m["round"] and c.get("code"):
                    m["cause"] = str(c["code"])

    complete = not missing
    trigger = cfg.get("rerun_required_missing_rounds")
    return {
        "declared_rounds": declared,
        "rounds_with_a_candidate": len(with_candidate),
        "n_missing_rounds": len(missing),
        "missing_rounds": missing,
        "verdict": "COMPLETE" if complete else "INCOMPLETE_MULTI_ROUND",
        "score_interpretation": ("two-sided" if complete else
                                 "lower bound only -- fewer attempts can only lower best-of-N"),
        "rerun_required": (False if trigger is None else len(missing) >= int(trigger)),
        "rule_source": "public task_config.yaml :: iteration.round_completeness",
        "may_not_support": ([] if complete else
                            list(cfg.get("incomplete_run_may_not_support")
                                 or ["model ranking", "complete three-model difficulty evidence",
                                     "saturation or non-saturation of the instance"])),
    }


def failure_code(verdict: Mapping[str, Any] | None) -> list[str]:
    """The code to append to a run's failure list, or nothing when the run was complete."""
    if not verdict or verdict.get("verdict") == "COMPLETE":
        return []
    return [f"{verdict['verdict']}:{verdict.get('n_missing_rounds', '?')}"]
