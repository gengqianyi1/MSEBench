#!/usr/bin/env bash
# 复现 task1–20：bash run_task1_20.sh [profile] [task列表|all]
set -euo pipefail
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROFILE="${1:-default}"
TASK="${2:-all}"
echo "▶ task1-20  profile=$PROFILE  task=$TASK"
exec python3 run.py --task "$TASK" --profile "$PROFILE"
