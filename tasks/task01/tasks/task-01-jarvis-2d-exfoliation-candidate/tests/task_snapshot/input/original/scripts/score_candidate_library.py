from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candidate_scoring import score_candidate_library


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic candidate-library scoring.")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--candidate-library", required=True)
    parser.add_argument("--generator-entrypoint", required=True)
    parser.add_argument("--generator-config", required=True)
    parser.add_argument("--generation-manifest", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    result = score_candidate_library(
        candidate_path=Path(args.candidate_library),
        train_path=workspace / "data/train.json.gz",
        constraints_path=workspace / "data/candidate_generation_constraints.json",
        workspace=workspace,
        generator_entrypoint=Path(args.generator_entrypoint),
        generator_config=Path(args.generator_config),
        generation_manifest=Path(args.generation_manifest),
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
