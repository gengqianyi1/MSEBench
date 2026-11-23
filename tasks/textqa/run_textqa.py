"""Run the TextQA scorer using the organizer archive's defaults."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from msebench.textqa import LLMError, extract_choices, match_answer, metrics, read_items, run_one, score_answer
from msebench.textqa import main as evaluate


def main(argv=None):
    task = json.loads(Path(__file__).with_name("task.json").read_text(encoding="utf-8"))
    return evaluate(argv, default_input=ROOT / task["dataset"]["path"], default_results=ROOT / "results")


if __name__ == "__main__":
    raise SystemExit(main())
