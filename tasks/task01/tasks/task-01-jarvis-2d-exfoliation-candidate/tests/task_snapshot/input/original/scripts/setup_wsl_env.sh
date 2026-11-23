#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
if ! command -v conda >/dev/null 2>&1; then
  echo "Conda or Mambaforge is required." >&2
  exit 1
fi
conda env create -f environment-linux.yml
cat <<'MSG'
Environment created.
Activate it with:
  conda activate jarvis_2d_exfoliation_benchmark
Then prepare a frozen dataset with scripts/prepare_dataset.py.
MSG
