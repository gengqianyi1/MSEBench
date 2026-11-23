from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_path = Path(args.trajectory_jsonl)
    output_path = Path(args.output) if args.output else input_path.with_name("trajectory.conversation.md")

    messages = {}
    message_idx = 1
    seen = set()
    for line in input_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for msg in rec.get("messages_input_to_llm", []):
            if msg.get("role") != "user":
                continue
            key = (msg.get("role"), msg.get("content", ""))
            if key in seen:
                continue
            seen.add(key)
            messages[f"message{message_idx}"] = {
                "role": "user",
                "content": msg.get("content", ""),
            }
            message_idx += 1

    output_path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved Markdown report to: {output_path}")


if __name__ == "__main__":
    main()

