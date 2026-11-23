#!/usr/bin/env bash
# 从头构建题 14–17（revision10）的全部镜像，并让题包认可这些本地镜像。
#
#   ./build.sh                         # 构建全部 12 个镜像（平台默认取本机 Docker daemon）
#   ./build.sh task16-sidecar ...      # 只构建点名的服务（服务名见 docker-compose.build.yaml）
#   NO_ADOPT=1 ./build.sh              # 只构建，不改题包的 docker/image_manifest.json
#
# 可选环境变量（也可写进本目录的 .env，见 .env.example）：
#   CHEMBENCH_PKG       题包目录，默认 ..
#   CHEMBENCH_PLATFORM  目标平台，默认本机 daemon 平台（组织方主机为 linux/amd64）
#   MICROMAMBA_IMAGE / PYTHON_IMAGE / GOST_IMAGE / CONDA_CHANNEL / PIP_INDEX_URL /
#   DEBIAN_MIRROR / DOCKER_APT_REPO   镜像源，默认都是官方源
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
die() { echo "ERROR: $*" >&2; exit 1; }

# .env 里的变量只在 shell 未设置时生效（与 docker compose 的优先级一致）
if [ -f .env ]; then
  while IFS='=' read -r key value; do
    [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] && [ -z "${!key+x}" ] && export "$key=$value" || true
  done < .env
fi
PKG_REL="${CHEMBENCH_PKG:-..}"
[ -d "$PKG_REL" ] || die "题包目录 $PKG_REL 不存在：先解压 14_17_曹阳_revision10_20260916.zip，或设置 CHEMBENCH_PKG"
PKG="$(cd "$PKG_REL" && pwd)"
export CHEMBENCH_PKG="$PKG"
RELEASE_ROOT="$(cd "$PKG/../.." && pwd)"
[ -f "$PKG/docker/runner/IMAGE" ] && [ -d "$PKG/tasks" ] || die "$PKG 不像是题 14–17 的题包"

docker version >/dev/null 2>&1 || die "连不上 Docker daemon"
docker compose version >/dev/null 2>&1 || die "需要 Docker Compose v2（docker compose）"
docker buildx version >/dev/null 2>&1 || die "需要 buildx 插件（BuildKit）：additional_contexts 依赖它"
DAEMON_PLATFORM="$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}')"
export CHEMBENCH_PLATFORM="${CHEMBENCH_PLATFORM:-$DAEMON_PLATFORM}"

# 运行器镜像的 requirements.txt 以题包为准
cmp -s "$PKG/requirements.txt" runner-context/requirements.txt \
  || { cp "$PKG/requirements.txt" runner-context/requirements.txt; echo "refreshed runner-context/requirements.txt from $PKG"; }

echo "package  : $PKG"
echo "platform : $CHEMBENCH_PLATFORM (daemon $DAEMON_PLATFORM)"
echo "### docker compose build"
docker compose -f docker-compose.build.yaml build "$@"

[ "${NO_ADOPT:-0}" = 1 ] && { echo "NO_ADOPT=1: built only; docker/image_manifest.json untouched"; exit 0; }
[ "$#" -eq 0 ] || { echo "built a subset; run ./build.sh with no arguments to adopt the full set"; exit 0; }
[ "$CHEMBENCH_PLATFORM" = "$DAEMON_PLATFORM" ] || {
  echo "built for $CHEMBENCH_PLATFORM on a $DAEMON_PLATFORM daemon: not adopting (Harbor here would run $DAEMON_PLATFORM)"
  exit 0; }

echo "### adopt: record the local images in $PKG/docker/image_manifest.json and check them"
RUNNER="$(cat "$PKG/docker/runner/IMAGE")"
docker run --rm --pull never \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$RELEASE_ROOT":"$RELEASE_ROOT" -w "$PKG" \
  "$RUNNER" python "$HERE/adopt_local_build.py" "$PKG"
