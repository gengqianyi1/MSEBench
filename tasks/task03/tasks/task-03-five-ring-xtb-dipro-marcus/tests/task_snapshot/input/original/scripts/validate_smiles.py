from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluator import evaluate_candidate
from utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-xtb", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    task_cfg = load_yaml(Path(cfg["task_dir"]) / "task_config.yaml")

    candidate = {
        "candidate_id": "test",
        "parent_candidate_id": None,
        "proposal_mode": "direct_smiles",
        "smiles": args.smiles,
        "structural_modification": "manual validation",
        "modification_reasoning": "manual validation",
    }

    ev = evaluate_candidate(
        candidate,
        task_cfg,
        Path("runs") / f"manual_validation_{os.getpid()}_{time.time_ns()}",
        seen_canonical_smiles=set(),
        run_xtb=args.run_xtb,
    )
    print(json.dumps(ev.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
