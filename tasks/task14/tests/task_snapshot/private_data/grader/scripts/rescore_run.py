"""Re-score a completed run against the CURRENT config, without calling any LLM.

This is a thin alias for `score_run.py`. Scoring logic lives in exactly one place: the second-round
review (田双燕, priority 2) asked for a check that `rescore_run.py` and the shipped scorer agree, and
the only way to make that check unfalsifiable is for there to be a single implementation. Before the
split this file carried its own copy of the grading loop, which is precisely the arrangement that can
drift. `grader/scripts/check_consistency.py` asserts that this file defines no scoring logic of its
own and that it re-exports `score_run.main`.

Needed whenever the enumeration vocabulary is widened and the reference target moves: the stored
trajectory already contains every submitted molecule, so the score can be recomputed exactly. This is
also the audit path for a cross-reviewer who wants to check a shipped score without spending API
credit.

Run:
    python grader/scripts/rescore_run.py runs/<run_dir>
    python grader/scripts/rescore_run.py --all
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_run import main  # noqa: E402,F401

if __name__ == "__main__":
    main()
