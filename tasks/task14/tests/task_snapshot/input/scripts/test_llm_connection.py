\
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_client import call_llm
from utils import load_json, load_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="_llm_connection_tests",
                        help="scratch dir for the probe. NOT under runs/: a partial directory there "
                             "would be picked up as a trajectory by the packaging script.")
    parser.add_argument("--list-models", action="store_true", help="Try to call client.models.list() as an additional check.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    api_config_file = cfg["llm"]["api_config_file"]
    llm_cfg = load_json(api_config_file)

    messages = [
        {
            "role": "system",
            "content": "Return only valid JSON. Do not include markdown.",
        },
        {
            "role": "user",
            "content": 'Return exactly this JSON object with no extra text: {"status":"ok","smiles":"CCO"}',
        },
    ]

    result = call_llm(messages, llm_cfg)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"llm_connection_test_{timestamp}.json"

    report = {
        "api_config_file": api_config_file,
        "model": llm_cfg.get("model"),
        "base_url": llm_cfg.get("base_url"),
        "result": result.to_dict(),
    }

    write_json(out_path, report)

    print(f"Saved debug report to: {out_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not result.ok:
        raise SystemExit(
            "LLM connection test failed or returned empty content. "
            "Open the saved debug report and inspect result.raw_response, result.message, result.finish_reason, and result.error."
        )

    try:
        json.loads(result.content)
        print("JSON parse check: OK")
    except Exception as exc:
        raise SystemExit(f"LLM returned non-JSON content: {exc}")


if __name__ == "__main__":
    main()
