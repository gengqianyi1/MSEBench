#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
exec conda run -n xtb-benchmark python "$ROOT/scripts/run_all.py" "$@"
