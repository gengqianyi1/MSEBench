#!/usr/bin/env bash
# Verify the organizer archive, including every locked Docker tar SHA256.
set -euo pipefail
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"

for file in run.py configs/llm_config.json README_RUN.md load_images.sh run_textqa.sh \
    run_task1_20.sh .env.example frameworks/14-17/docker_build/docker-compose.build.yaml \
    frameworks/19-20/scripts/task_paths.py; do
    [ -f "$file" ] || { echo "Missing file: $file" >&2; exit 1; }
done
for number in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20; do
    [ -d "tasks/task$number" ] || { echo "Missing task: $number" >&2; exit 1; }
done

"$PYTHON" scripts/validate_tasks.py
"$PYTHON" run.py --task textqa --verify
"$PYTHON" - <<'PY'
from pathlib import Path

root = Path.cwd()
for path in root.rglob("*"):
    if ".git" in path.parts or not path.is_symlink():
        continue
    if path.readlink().is_absolute() or not path.exists() or not path.resolve().is_relative_to(root):
        raise SystemExit(f"Non-portable link: {path.relative_to(root)}")
print("PASS: all symlinks are relative and resolve within the archive")
PY
"$PYTHON" scripts/verify_images.py
