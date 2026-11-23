#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-chembench_task2_tool}"
ZIP_PATH="${1:-/mnt/e/download/file/pyscf-2.13.1.zip}"
BUILD_ROOT="${BUILD_ROOT:-/tmp/pyscf_smd_build}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Run this inside Linux/WSL with Miniforge/Mambaforge available." >&2
  exit 1
fi

if [ ! -f "$ZIP_PATH" ]; then
  echo "PySCF source zip not found: $ZIP_PATH" >&2
  echo "Usage: bash scripts/install_pyscf_smd_from_local.sh /path/to/pyscf-2.13.1.zip" >&2
  exit 1
fi

mkdir -p "$BUILD_ROOT"
rm -rf "$BUILD_ROOT/pyscf-2.13.1"
python - <<PY
import zipfile
zip_path = r"$ZIP_PATH"
out_dir = r"$BUILD_ROOT"
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(out_dir)
print("extracted", zip_path, "to", out_dir)
PY

SRC_DIR="$BUILD_ROOT/pyscf-2.13.1"
if [ ! -f "$SRC_DIR/pyproject.toml" ]; then
  echo "Extracted PySCF source directory is invalid: $SRC_DIR" >&2
  exit 1
fi

conda run -n "$ENV_NAME" python -m pip uninstall -y pyscf pyscf-forge || true

cd "$SRC_DIR"
echo "Building PySCF with SMD/libsolvent enabled from $SRC_DIR"
conda run -n "$ENV_NAME" bash -lc '
set -euo pipefail
export CMAKE_CONFIGURE_ARGS="-DENABLE_SMD=ON -DBUILD_MARCH_NATIVE=OFF -DENABLE_XCFUN=OFF -DBUILD_XCFUN=OFF"
python -m pip install -v --no-cache-dir --no-build-isolation .
'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda run -n "$ENV_NAME" python "$ROOT/scripts/check_environment.py" --quick
