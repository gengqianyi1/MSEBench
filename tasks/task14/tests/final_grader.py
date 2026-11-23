#!/usr/bin/env python3
"""Task 14 final verifier (M1.B).

Rebuilds the legacy submission surface (trajectory.jsonl + run_summary.json) from the five
Harbor step candidates, so the EXISTING legacy grader can re-derive every property from SMILES
and return the official score. Never trusts feedback.json / best_so_far.json / agent-reported
properties.
"""
from __future__ import annotations
import json, os
from pathlib import Path

WORK = Path(os.environ.get('CHEMBENCH_WORKSPACE', '/workspace'))
ROUNDS = WORK / 'rounds'


def _read_candidate_json(path):
    """Read one round's candidate file, treating a corrupt one as a round with no candidate.

    A truncated or malformed `candidate.json` is the single most common way a real model fails a
    round -- the archived legacy runs for this task contain `UNPARSEABLE_SUBMISSION` rounds. Letting
    `json.loads` raise here makes the whole verifier exit non-zero, which `run_legacy_grader.py`
    records as `verifier_error` with `reward 0.0`: an INFRASTRUCTURE failure. That silently converts
    a bad submission into "re-run this", which is exactly the misattribution Evaluation §16.5 and
    §10 forbid. Returning None instead lets the round be counted as missing, the score be computed
    as a genuine lower bound, and `protocol_completeness` report it as INCOMPLETE_MULTI_ROUND.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError, OSError):
        return None


def _candidate(n: int):
    p = ROUNDS / ('round-%d' % n) / 'candidate.json'
    if not p.is_file():
        return None
    d = _read_candidate_json(p)
    if d is None:
        return None
    c = d.get('candidate', d) if isinstance(d, dict) else {}
    if not isinstance(c, dict):
        return None
    return {
        'round': n,
        'submitted_candidate': {
            'name': str(c.get('name', '')).strip(),
            'smiles': str(c.get('smiles', '')).strip(),
            'rationale': str(c.get('rationale', '')).strip(),
        },
    }


def main() -> int:
    recs = [r for r in (_candidate(n) for n in range(1, 6)) if r and r.get('submitted_candidate', {}).get('smiles')]
    lines = [json.dumps(r, ensure_ascii=False) for r in recs]
    (WORK / 'trajectory.jsonl').write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    (WORK / 'run_summary.json').write_text(
        json.dumps({'reconstructed_from_harbor_steps': True, 'n_rounds_recorded': len(recs)}, indent=2),
        encoding='utf-8')
    # environment/public/rubric.json marks BOTH final_answer_present and
    # final_answer_declares_task CRITICAL, but this grader never wrote final_answer.json
    # at all until 2026-09-10 -- a stronger form of the gap a round-5 reviewer found on
    # task 17. Nothing caught it because no step executed the rubric; tests/rubric_check.py
    # does now. 'rounds' carries the submitted candidates so final_answer_records_rounds
    # (count_in_range 1..5) is satisfied by real content rather than a stub.
    (WORK / 'final_answer.json').write_text(
        json.dumps({'task_id': 'task_14_multiobjective_mol_design',
                    'rounds': [r['submitted_candidate'] for r in recs]}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    print(json.dumps({'reconstructed_rounds': len(recs)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
