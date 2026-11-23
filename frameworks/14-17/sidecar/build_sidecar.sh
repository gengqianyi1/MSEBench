#!/usr/bin/env bash
# Build the private response sidecar for task 16 or 17 and print the digest to freeze.
#
#     ./sidecar/build_sidecar.sh 16 [tag]
#
# Run from the PACKAGE ROOT: the build context has to contain both the framework server template
# and the task snapshot, which live in different subtrees.
#
# The digest printed at the end is what goes into
# `tasks/<task>/environment/public/sidecar_deployment.yaml :: release.private_image_digest`.
# Two rules about which digest to use:
#
#   * Use the digest of the image FOR THE ARCHITECTURE THE BENCHMARK WILL RUN ON. A local build on
#     an arm64 laptop and an x86_64 build of the same Dockerfile have different digests. Pass
#     --platform (or set PLATFORM=linux/amd64) to match the evaluation host.
#   * A local `docker build` produces only an image ID; a repo digest exists once the image is
#     pushed to a registry. Until the organizer decides where these live, the ID below is the
#     reproducibility anchor, not a distribution address. Do NOT push to a public registry: this
#     image contains private_data/, i.e. the reference optimum.
set -euo pipefail

NUMBER="${1:?usage: build_sidecar.sh <16|17> [tag]}"
case "$NUMBER" in
  16) TASK_DIR="tasks/task-16-tablet-microenvironment-cqa"; PORT=8716 ;;
  17) TASK_DIR="tasks/task-17-spherical-agglomeration";     PORT=8717 ;;
  *)  echo "task number must be 16 or 17" >&2; exit 2 ;;
esac
TAG="${2:-chembench/task${NUMBER}-sidecar:2.1.0-r10}"
PLATFORM="${PLATFORM:-}"

[ -f templates/sidecar/private_response_server.py ] || {
  echo "run this from the package root (templates/sidecar/private_response_server.py not found)" >&2
  exit 2; }
[ -d "${TASK_DIR}/tests/task_snapshot/private_data" ] || {
  echo "missing ${TASK_DIR}/tests/task_snapshot/private_data" >&2; exit 2; }

echo "building ${TAG} from ${TASK_DIR} (port ${PORT})"
docker build \
  ${PLATFORM:+--platform "${PLATFORM}"} \
  -f sidecar/Dockerfile \
  --build-arg "TASK_DIR=${TASK_DIR}" \
  --build-arg "TASK_NUMBER=${NUMBER}" \
  --build-arg "PORT=${PORT}" \
  -t "${TAG}" \
  .

ID=$(docker image inspect --format '{{.Id}}' "${TAG}")
ARCH=$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "${TAG}")
echo
echo "image      : ${TAG}"
echo "platform   : ${ARCH}"
echo "image id   : ${ID}"
echo
echo "freeze into ${TASK_DIR}/environment/public/sidecar_deployment.yaml as:"
echo "  release:"
echo "    private_image_digest: ${ID}"
echo "    private_image_platform: ${ARCH}"
echo "    release_ready: true"
