#!/usr/bin/env python3
import json,os
from pathlib import Path
WORK=Path(os.environ.get("CHEMBENCH_WORKSPACE","/workspace"))
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


def main():
 records=[]; rounds=[]
 for n in range(1,6):
  path=WORK/"rounds"/f"round-{n}"/"candidate.json"
  if not path.is_file(): continue
  payload=_read_candidate_json(path)
  if payload is None: continue
  candidate=payload.get("candidate") if isinstance(payload,dict) else None
  if isinstance(candidate,dict) and isinstance(candidate.get("formulation"),dict):
   records.append({"round":n,"submitted_candidate":candidate}); rounds.append(candidate)
 (WORK/"trajectory.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records),encoding="utf-8")
 summary={"reconstructed_from_harbor_steps":True,"n_rounds_recorded":len(records),"max_rounds":5}
 (WORK/"run_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
 (WORK/"final_report.json").write_text(json.dumps({"checks":summary},indent=2),encoding="utf-8")
 # rubric.json marks final_answer_declares_task CRITICAL, keyed on "task_id". This wrote
 # {"rounds": ...} only until 2026-09-10, so the public structural contract and the real
 # pipeline disagreed -- unnoticed, because nothing ran the rubric. tests/rubric_check.py
 # runs it now; this line makes it pass honestly.
 (WORK/"final_answer.json").write_text(json.dumps({"task_id":"task_16_tablet_microenvironment_cqa","rounds":rounds},indent=2),encoding="utf-8")
 print(json.dumps(summary)); return 0
if __name__=="__main__": raise SystemExit(main())
