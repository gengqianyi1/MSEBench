#!/usr/bin/env bash
# Run a task from this package with Harbor. Written for an EMPTY, OFFLINE Docker environment: it assumes
# a Docker daemon, the Docker CLI with the compose plugin, harbor from requirements.txt (all three are in
# the shipped runner image), and the images from docker/ already loaded (./chembench load-images).
# It reads nothing outside this package directory and never pulls or builds an image: the organizer
# host has no route to Docker Hub (2026-09-14), so every run starts shipped images by name.
#
#   scripts/run_task.sh preflight
#   scripts/run_task.sh oracle   <task-dir-name>  [--local-public-network]
#   scripts/run_task.sh llm      <task-dir-name>  [--local-public-network]
#   scripts/run_task.sh llm-mock <task-dir-name>  [--scenario normal|large|invalid-round] [--local-public-network]
#   scripts/run_task.sh summary  <job-name>
#
# llm-mock runs the REAL llm path -- agent, task container, 16/17 sidecar, round setup scripts,
# verifiers -- against scripts/mock_llm_server.py instead of a model account, and judges the run with
# scripts/llm_mock_check.py. Run it before spending model credit. The organizer's first DeepSeek run
# (2026-09-16) found the llm path broken on all four tasks, after every Oracle run and every offline
# agent check had passed: none of them exercised the agent inside a real task container.
#
# PORTS: none are published on the host. Tasks 16/17 reach their private sidecar at 127.0.0.1:8716/8717
# INSIDE the trial's own agent network namespace (compose network_mode: service:main), so concurrent
# trials and the host never contend for a port.
#
# NETWORK POLICY: every task declares network_mode = "no-network". Harbor enforces that with an egress
# control sidecar which needs CONFIG_NFT_FIB_INET in the Docker HOST kernel; without it Harbor's docker
# provider refuses the task before building (ValueError: network_mode='no-network' is not supported).
# Native Linux Docker hosts normally have it; Docker Desktop (macOS/Windows) does not. `preflight`
# reports which case you are in. `--local-public-network` exists ONLY so runnability can be checked on a
# host without that support: it copies the task into jobs/_public_network_copies/, sets public network
# there, and runs the copy. The shipped task is never modified. A run made this way is NOT an
# isolation-equivalent evaluation and must not be reported as one; the result summary says so.
set -euo pipefail

PKG="$(cd "$(dirname "$0")/.." && pwd)"
JOBS="$PKG/jobs"
cd "$PKG"

die() { echo "ERROR: $*" >&2; exit 2; }
need_harbor() { command -v harbor >/dev/null || die "harbor not found; activate the venv built from requirements.txt"; }

py() { PYTHONPATH="$PKG${PYTHONPATH:+:$PYTHONPATH}" python "$@"; }

probe_image() {
  # any image that is already loaded; the probe used to `docker run alpine`, which is a Docker Hub pull
  local img
  for img in "$(cat "$PKG/docker/runner/IMAGE")" $(py "$PKG/scripts/offline_images.py" required --role environment --format refs); do
    docker image inspect "$img" >/dev/null 2>&1 && { echo "$img"; return 0; }
  done
  return 1
}

egress_supported() {
  # the same kernel probe Harbor runs (harbor/environments/docker/docker.py::_egress_control_kernel_support);
  # returns 2 when no shipped image is loaded to run it in
  local img
  img="$(probe_image)" || return 2
  docker run --rm --pull never --entrypoint sh "$img" -c \
    'if [ ! -f /proc/config.gz ]; then exit 0; fi; zcat /proc/config.gz 2>/dev/null | grep -qE "^CONFIG_NFT_FIB_INET=[ym]"' \
    >/dev/null 2>&1
}

task_path() {
  local task="$1" public="$2"
  [ -d "$PKG/tasks/$task" ] || die "no such task: $task (have: $(ls "$PKG/tasks" | tr '\n' ' '))"
  if [ "$public" = "1" ]; then
    local copy="$JOBS/_public_network_copies/$task"
    rm -rf "$copy"; mkdir -p "$(dirname "$copy")"
    cp -R "$PKG/tasks/$task" "$copy"
    sed -i.bak 's/network_mode = "no-network"/network_mode = "public"/' "$copy/task.toml" && rm -f "$copy/task.toml.bak"
    # the 16/17 sidecar build context is the package root (../../.. from environment/); keep that true
    for f in "$copy/environment/docker-compose.yaml"; do
      [ -f "$f" ] && sed -i.bak "s#context: ../../..#context: $PKG#" "$f" && rm -f "$f.bak"
    done
    echo "$copy"
  else
    echo "$(cd "$PKG/tasks/$task" && pwd -P)"
  fi
}

run_trial() {
  # run_trial <oracle|llm> <task-dir-name> <public 0|1> <job-name>
  local kind="$1" task="$2" public="$3" job="$4" path rc egress_flag
  need_harbor
  if [ "$public" = "0" ]; then
    rc=0; egress_supported || rc=$?
    [ "$rc" = 2 ] && die "shipped images are not loaded: ./chembench load-images"
    [ "$rc" = 0 ] || die "this Docker host cannot enforce no-network (see 'preflight'); rerun with --local-public-network for a runnability check"
  fi
  # Every image this trial starts must already be loaded: Harbor would otherwise try to pull or build
  # it, which on an offline host hangs until the registry timeout and then fails.
  egress_flag=""; [ "$public" = "1" ] && egress_flag="--no-egress"
  py "$PKG/scripts/offline_images.py" check --task "$task" --quiet $egress_flag \
    || die "images for $task are missing or stale (listed above): ./chembench load-images"
  path="$(task_path "$task" "$public")"
  mkdir -p "$JOBS"
  if [ "$kind" = "oracle" ]; then
    harbor run -p "$path" -a oracle -e docker -n 1 -o "$JOBS" --job-name "$job" --yes || echo "harbor exited $?"
  else
    : "${CHEMBENCH_MODEL_ID:?export CHEMBENCH_MODEL_ID}"
    : "${CHEMBENCH_API_KEY:?export CHEMBENCH_API_KEY (never write it into a file)}"
    if [ -z "${CHEMBENCH_API_BASE:-}" ] && [ -z "${CHEMBENCH_API_URL:-}" ]; then
      die "export CHEMBENCH_API_BASE (OpenAI-compatible base URL) or CHEMBENCH_API_URL (full chat/completions URL)"
    fi
    # ChemBenchAgent is a harbor BaseAgent: it runs in THIS process and reaches into the task
    # container with environment.exec, so the credentials stay on the runner and the model
    # traffic never leaves from the no-network task container.
    PYTHONPATH="$PKG${PYTHONPATH:+:$PYTHONPATH}" \
      harbor run -p "$path" -a agents.chembench_agent:ChemBenchAgent -e docker -n 1 -o "$JOBS" --job-name "$job" --yes \
      || echo "harbor exited $?"
  fi
  if [ "$public" = "1" ]; then
    echo "NOTE: run used --local-public-network (task copy with public network). Not an isolation-equivalent evaluation."
  fi
  return 0
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  preflight)
    echo "package      : $PKG"
    docker version --format 'docker       : CLI {{.Client.Version}} -> daemon {{.Server.Version}}' || die "cannot reach a Docker daemon"
    docker compose version >/dev/null 2>&1 && echo "compose      : $(docker compose version --short)" || die "docker compose plugin missing"
    command -v harbor >/dev/null && echo "harbor       : $(harbor --version)" || echo "harbor       : NOT FOUND (activate the venv)"
    echo "### shipped images (docker/image_manifest.json)"
    images_ok=1
    py "$PKG/scripts/offline_images.py" check --all-roles || images_ok=0
    rc=0; egress_supported || rc=$?
    if [ "$rc" = 0 ]; then
      echo "no-network   : SUPPORTED (host kernel has CONFIG_NFT_FIB_INET) -- run tasks as shipped"
    elif [ "$rc" = 2 ]; then
      echo "no-network   : UNKNOWN -- no shipped image loaded to probe with; run ./chembench load-images"
    else
      echo "no-network   : NOT SUPPORTED on this Docker host -- Harbor will refuse the shipped tasks here;"
      echo "               use a native Linux Docker host, or --local-public-network for a runnability check only"
    fi
    [ "$images_ok" = 1 ] || die "shipped images missing or stale (see above): ./chembench load-images"
    echo "tasks        : $(ls "$PKG/tasks" | tr '\n' ' ')"
    ;;

  oracle|llm)
    task="${1:?usage: run_task.sh $cmd <task-dir-name> [--local-public-network]}"; shift
    public=0; [ "${1:-}" = "--local-public-network" ] && public=1
    job="${cmd}-${task}-$(date -u +%Y%m%dT%H%M%SZ)"
    run_trial "$cmd" "$task" "$public" "$job"
    "$0" summary "$job"
    ;;

  llm-mock)
    task="${1:?usage: run_task.sh llm-mock <task-dir-name> [--scenario normal|large|invalid-round] [--local-public-network]}"; shift
    scenario=normal; public=0
    while [ $# -gt 0 ]; do
      case "$1" in
        --scenario) scenario="${2:?--scenario needs a value}"; shift 2 ;;
        --local-public-network) public=1; shift ;;
        *) die "unknown option: $1" ;;
      esac
    done
    [ -f "$PKG/tasks/$task/task.toml" ] || die "no such task: $task"
    job="llm-mock-${scenario}-${task}-$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$JOBS/_mock_llm"
    mock_log="$JOBS/_mock_llm/$job.log"; port_file="$JOBS/_mock_llm/$job.port"
    python "$PKG/scripts/mock_llm_server.py" --package "$PKG" --task "$task" --scenario "$scenario" \
      --port-file "$port_file" > "$mock_log" 2>&1 &
    mock_pid=$!
    trap 'kill "$mock_pid" 2>/dev/null || true' EXIT
    for _ in $(seq 1 100); do [ -s "$port_file" ] && break; sleep 0.1; done
    [ -s "$port_file" ] || die "mock model server did not start (see $mock_log)"
    # A loopback endpoint and a placeholder credential (the xxxx form the delivery secret scan recognises
    # as a placeholder): nothing leaves this machine, no account is used.
    (
      export CHEMBENCH_API_BASE="http://127.0.0.1:$(cat "$port_file")/v1" CHEMBENCH_API_URL="" \
             CHEMBENCH_MODEL_ID="mock-$scenario" CHEMBENCH_API_KEY="mock-xxxx" CHEMBENCH_REQUEST_RETRIES=1
      run_trial llm "$task" "$public" "$job"
    ) || echo "trial did not complete"
    kill "$mock_pid" 2>/dev/null || true
    "$0" summary "$job" || true
    echo "### llm-mock verdict (scenario: $scenario)"
    py "$PKG/scripts/llm_mock_check.py" --job "$JOBS/$job" --task "$task" --scenario "$scenario" --mock-log "$mock_log"
    ;;

  summary)
    job="${1:?usage: run_task.sh summary <job-name>}"
    python - "$JOBS/$job" <<'PY'
import json, sys
from pathlib import Path
job = Path(sys.argv[1])
if not job.is_dir():
    sys.exit(f"no job directory {job}")
trials = sorted(job.glob("*/result.json"))
if not trials:
    sys.exit(f"{job}: no trial result.json -- the run did not reach a trial (see {job}/job.log)")
invalid = 0
for t in trials:
    r = json.loads(t.read_text())
    exc = (r.get("exception_info") or {}).get("exception_type")
    rounds = [(s.get("verifier_result") or {}).get("rewards", {}).get("reward") for s in r.get("step_results") or []]
    final = (r.get("verifier_result") or {}).get("rewards", {}).get("reward")
    print(f"trial     : {t.parent.name}")
    print(f"task      : {r.get('task_name')}")
    print(f"exception : {exc}")
    print(f"rewards   : per round {rounds}  final {final}")
    details = t.parent / "steps" / "round-5" / "verifier" / "details.json"
    if not details.is_file():
        details = t.parent / "verifier" / "details.json"
    if details.is_file():
        d = json.loads(details.read_text())
        lr = d.get("legacy_result") or {}
        gate = d.get("reward_gate") or {}
        print(f"score     : {lr.get('score')}  codes {lr.get('failure_codes')}")
        print(f"gate      : completeness {gate.get('protocol_completeness_verdict')}  reward_written {gate.get('reward_written')}")
    print(f"artifacts : {t.parent}")
    # Harbor still runs the verifier after an agent exception, so a round that never reached the model
    # is scored like a bad answer. The organizer's 2026-09-16 run reported tasks 16/17 as 0.0 while every
    # round had failed before the first model call. A run with any exception is not a model score.
    broken = [(s.get("step_name") or s.get("name"), s["exception_info"]) for s in r.get("step_results") or []
              if s.get("exception_info")]
    if r.get("exception_info"):
        broken.insert(0, ("trial", r["exception_info"]))
    if broken or final is None:
        invalid += 1
        print("verdict   : NOT A MODEL SCORE -- the run did not measure the model:")
        for name, info in broken:
            print(f"              {name}: {info.get('exception_type')}: {str(info.get('exception_message'))[:240]}")
        if final is None:
            print("              no final reward")
        print("            Causes are infrastructure (sidecar, API endpoint or account, container I/O). Fix and rerun;")
        print("            do not report the reward above.")
    else:
        print("verdict   : scored run (no exception in any round)")
sys.exit(3 if invalid else 0)
PY
    ;;

  *)
    sed -n '2,32p' "$0"
    ;;
esac
