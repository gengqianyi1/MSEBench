#!/usr/bin/env bash
# Load the offline images into this host's Docker daemon, then verify them.
# Needs only the Docker CLI and sha256sum -- no Python, no network.      docker/load_images.sh
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PKG="$(cd "$DIR/.." && pwd)"
cd "$DIR"
shopt -s nullglob
sums=(chembench_14_17_images_*.tar.gz.sha256)
if [ "${#sums[@]}" -ne 1 ]; then
  echo "ERROR: expected one image archive checksum in $DIR, found ${#sums[@]}" >&2; exit 2
fi
archive="${sums[0]%.sha256}"
bundle_platform="$(sed -n 's/^  "platform": "\([^"]*\)".*/\1/p' image_manifest.json)"
daemon_platform="$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}')"
echo "archive  : $DIR/$archive ($(du -h "$archive" | cut -f1))"
echo "platform : images $bundle_platform, Docker daemon $daemon_platform"
if [ "$bundle_platform" != "$daemon_platform" ]; then
  echo "WARNING: platform mismatch -- containers would run under emulation or not at all." >&2
fi
echo "### sha256"
if command -v sha256sum >/dev/null; then sha256sum -c "${sums[0]}"; else shasum -a 256 -c "${sums[0]}"; fi
echo "### docker load (no registry is contacted)"
docker load -i "$archive"
echo "### verify every image by layer content (inside the runner image just loaded)"
RUNNER="$(cat "$DIR/runner/IMAGE")"
if ! docker run --rm --pull never \
    -v /var/run/docker.sock:/var/run/docker.sock -v "$PKG":"$PKG" -w "$PKG" \
    "$RUNNER" python scripts/offline_images.py check --all-roles; then
  echo "" >&2
  echo "If the error above is 'No such file or directory' for scripts/offline_images.py, this Docker daemon" >&2
  echo "cannot see $PKG at the same path. Harbor has the same requirement (it bind-mounts trial" >&2
  echo "directories by absolute path), so put the package on a path the daemon resolves identically." >&2
  exit 1
fi
