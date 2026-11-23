#!/usr/bin/env bash
# Verify every locked archive before loading the local evaluation images.
set -euo pipefail
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ "$#" -eq 0 ]; then
    set -- --all
fi
exec "${PYTHON:-python3}" scripts/image_manager.py load "$@"
