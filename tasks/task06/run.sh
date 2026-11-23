#!/usr/bin/env bash
# Thin wrapper: delegates to the unified MatBench engine (src/msebench/orchestrator.py).
# The engine resolves the agent (msebench.agent.unified:MSEBenchAgent) and the protocol
# plugin (msebench.protocols.workspace_sidecar); this package only supplies the task
# definition, environment, verifier and compose payload.
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
INTEG="${MSEBENCH_INTEG_ROOT:-}"
if [[ -z "${INTEG:-}" || ! -f "${INTEG:-/nonexistent}/src/msebench/orchestrator.py" ]]; then
  INTEG=$(CDPATH= cd -- "$ROOT/../.." && pwd)
fi
TASK=$(tr -d '\r\n' < "$ROOT/TASK_ID")
ACTION=${1:-smoke}
shift || true
case "$ACTION" in
  load)
    TARDIR="$INTEG/images/task$TASK"
    [[ -d "$TARDIR" ]] || { echo "missing per-image tar dir: $TARDIR" >&2; exit 2; }
    for tar in "$TARDIR"/*.tar; do docker load -i "$tar"; done
    exit 0 ;;
  smoke|oracle|invalid|model) ;;
  *) echo 'Usage: bash run.sh {load|smoke|oracle|invalid|model}' >&2; exit 2 ;;
esac
if [[ "$ACTION" == model ]]; then
  : "${CHEMBENCH_API_URL:?Set the full chat/completions endpoint on the host}"
  : "${CHEMBENCH_MODEL_ID:?Set the requested provider model ID on the host}"
  : "${CHEMBENCH_API_KEY:?Set the API key on the host; never place it in this package}"
fi
ORGANIZER=$(tr -d '\r\n' < "$ROOT/environment/organizer-image.txt")
SOCKET=${CHEMBENCH_DOCKER_SOCKET:-/var/run/docker.sock}
[[ -S "$SOCKET" ]] || { echo "Docker socket not available: $SOCKET" >&2; exit 2; }
export CHEMBENCH_REQUEST_TIMEOUT_SECONDS=${CHEMBENCH_REQUEST_TIMEOUT_SECONDS:-900}
if [[ "$TASK" == 04 ]]; then
  export CHEMBENCH_MAX_TOKENS=${CHEMBENCH_MAX_TOKENS:-65536}
fi
RUNS="${MSEBENCH_RUNS_ROOT:-$INTEG/runs}"
mkdir -p "$RUNS"
# Trusted organizer only. Its Docker socket/private package are NOT model mounts.
exec docker run --rm --network host \
  --mount "type=bind,source=$SOCKET,target=/var/run/docker.sock" \
  --mount "type=bind,source=$INTEG,target=$INTEG" \
  --mount "type=bind,source=$RUNS,target=$RUNS" \
  --mount "type=bind,source=$ROOT,target=$ROOT" \
  --workdir "$INTEG" \
  --env "PYTHONPATH=$INTEG/src" --env PYTHONDONTWRITEBYTECODE=1 \
  --env MSEBENCH_INTEG_ROOT="$INTEG" \
  --env MSEBENCH_API_KEY --env CHEMBENCH_API_URL --env CHEMBENCH_MODEL_ID \
  --env CHEMBENCH_API_KEY \
  --env CHEMBENCH_MAX_TOKENS --env CHEMBENCH_REQUEST_TIMEOUT_SECONDS \
  --env CHEMBENCH_TEMPERATURE --env CHEMBENCH_TOP_P \
  --env CHEMBENCH_ENABLE_THINKING --env CHEMBENCH_REASONING_EFFORT \
  --env CHEMBENCH_LLM_CALL_LOG \
  "$ORGANIZER" \
  python "$INTEG/src/msebench/orchestrator.py" --task "$TASK" --mode "$ACTION" \
  --output-root "$RUNS"
