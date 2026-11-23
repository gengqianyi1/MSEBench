#!/usr/bin/env bash
# 复现 textqa：bash run_textqa.sh [profile] [input.jsonl]
set -euo pipefail
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROFILE="${1:-default}"
if [ "$#" -ge 2 ]; then
    exec python3 run.py --task textqa --profile "$PROFILE" --input "$2"
fi
exec python3 run.py --task textqa --profile "$PROFILE"
