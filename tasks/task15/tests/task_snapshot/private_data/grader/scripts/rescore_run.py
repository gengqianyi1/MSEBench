"""Re-score a completed run against the CURRENT config, without calling any LLM.

This is a thin alias for `score_run.py`, kept because it is the name referenced in the delivery
checklist. Scoring logic lives in exactly one place -- an earlier version of this package had two
scorers that could drift apart, and one of them did.

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

from score_run import main  # noqa: E402

if __name__ == "__main__":
    main()
