#!/usr/bin/env python3
"""Fixed, quota-carrying bridge from the Harbor protocol Agent to Task 15 tools."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys
import yaml

PUBLIC = Path(__file__).resolve().parents[1]
SRC = PUBLIC / "input/src"
sys.path.insert(0, str(SRC))
from tools import ToolSession  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--used", required=True, type=int)
    args = parser.parse_args()
    calls = json.loads(base64.b64decode(args.payload).decode("utf-8"))
    config = yaml.safe_load((PUBLIC / "input/tasks/q1_lipophilic/task_config.yaml").read_text(encoding="utf-8"))
    session = ToolSession(
        start_smiles=config["start_smiles"],
        constraints=config["constraints"],
        similarity_cfg=config["constraints"].get("similarity"),
        max_calls=int(config["tools"]["max_calls_per_round"]),
        calls_used=args.used,
    )
    responses = session.call_batch(calls)
    print(json.dumps({"responses": responses, "calls_used": session.calls_used, "log": session.log}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
