#!/usr/bin/env bash
# Build every Docker image tasks 14-17 need and pack them into the offline archive in docker/.
#
# Author side, on a host WITH network (Docker Hub, conda-forge, PyPI, download.docker.com). The organizer
# host never runs this: it loads the archive this produced (docker/load_images.sh). Recipes are the
# package's own Dockerfiles, and image names are read from task.toml / docker-compose.yaml, so the
# archive cannot drift from what Harbor starts.
#
#   docker/build_images.sh [--platform linux/amd64]
#
# Needs docker buildx and a Python >= 3.11 with requirements.txt installed (Harbor computes the name of
# its egress-control image; the archive must contain it under exactly that name).
set -euo pipefail
PKG="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
PYTHONPATH="$PKG${PYTHONPATH:+:$PYTHONPATH}" exec "${PYTHON:-python3}" "$PKG/scripts/offline_images.py" build "$@"
