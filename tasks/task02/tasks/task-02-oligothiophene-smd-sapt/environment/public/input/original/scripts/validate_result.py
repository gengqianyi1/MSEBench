from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scoring import score_rules


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic V2 scorer on an existing workspace result.")
    parser.add_argument("result_json", help="Path to workspace/result.json")
    parser.add_argument("--run-dir", default=None, help="Run directory containing tool_calls.jsonl and stage_integrity.jsonl")
    args = parser.parse_args()

    result_path = Path(args.result_json).resolve()
    workspace = result_path.parent
    run_dir = Path(args.run_dir).resolve() if args.run_dir else (workspace.parent if workspace.name == "workspace" else workspace)
    result = json.loads(result_path.read_text(encoding="utf-8"))

    schema_path = workspace / "result_schema.json"
    schema_errors = []
    if schema_path.exists():
        try:
            import jsonschema
            jsonschema.validate(result, json.loads(schema_path.read_text(encoding="utf-8")))
        except Exception as exc:
            schema_errors.append(str(exc))

    evaluation = score_rules(run_dir, workspace, result)
    evaluation["schema_validation_errors"] = schema_errors
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
