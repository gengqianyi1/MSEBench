#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Install Miniforge or Mambaforge in WSL first." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda env create -f "$ROOT/environment-linux.yml"

# The binary conda-forge PySCF build may lack libsolvent/SMD support.
# Build PySCF from source with SMD enabled inside the environment.
LOCAL_PYSCF_ZIP="${LOCAL_PYSCF_ZIP:-/mnt/e/download/file/pyscf-2.13.1.zip}"
if [ -f "$LOCAL_PYSCF_ZIP" ]; then
  bash "$ROOT/scripts/install_pyscf_smd_from_local.sh" "$LOCAL_PYSCF_ZIP"
else
  conda run -n chembench_task2_tool python -m pip uninstall -y pyscf pyscf-forge || true
  conda run -n chembench_task2_tool bash -lc '
  set -euo pipefail
  export CMAKE_CONFIGURE_ARGS="-DENABLE_SMD=ON -DBUILD_MARCH_NATIVE=OFF -DENABLE_XCFUN=OFF -DBUILD_XCFUN=OFF"
  python -m pip install -v --no-cache-dir --no-binary pyscf "git+https://github.com/pyscf/pyscf.git@v2.13.1"
  '
fi

conda run -n chembench_task2_tool python "$ROOT/scripts/check_environment.py" --quick
