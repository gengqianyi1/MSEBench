#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
python -s -m scripts.validate_project --delivery
python -s -m pytest -q
python -s -m scripts.replay_trajectories baseline kimi deepseek glm --output .test_tmp/replay_validation_delivery.json
python -s -m scripts.secret_scan
