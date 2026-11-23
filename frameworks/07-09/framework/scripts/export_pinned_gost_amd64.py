#!/usr/bin/env python3
"""Fetch and export the pinned gogost/gost linux/amd64 image for docker load."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
from urllib.parse import urlencode
from urllib.request import Request, urlopen


REPOSITORY = "gogost/gost"
INDEX_DIGEST = "sha256:afc0137758ab4ce399d47a299f9abbacbf522b52a17e59cbb4b4e7a1a66e9196"
AMD64_MANIFEST_DIGEST = "sha256:c41758e5be07bd07b062cafe301d5dfe34ebc1529888be8e9c7224500f26b7ac"
LOCAL_TAG = "gogost/gost:offline-pinned-amd64"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "egress-control-offline-fix"
OUTPUT_TAR = OUTPUT_DIR / "gogost-gost-pinned-linux-amd64.tar"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get(url: str, *, token: str | None = None, accept: str | None = None) -> bytes:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if accept:
        headers["Accept"] = accept
    with urlopen(Request(url, headers=headers), timeout=120) as response:
        return response.read()


def add(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    query = urlencode({
        "service": "registry.docker.io",
        "scope": f"repository:{REPOSITORY}:pull",
    })
    token = json.loads(get(f"https://auth.docker.io/token?{query}"))["token"]
    registry = f"https://registry-1.docker.io/v2/{REPOSITORY}"
    manifest_bytes = get(
        f"{registry}/manifests/{AMD64_MANIFEST_DIGEST}",
        token=token,
        accept="application/vnd.oci.image.manifest.v1+json",
    )
    assert "sha256:" + sha256(manifest_bytes) == AMD64_MANIFEST_DIGEST
    manifest = json.loads(manifest_bytes)
    descriptors = [manifest["config"], *manifest["layers"]]
    blobs: dict[str, bytes] = {}
    for descriptor in descriptors:
        value = get(f"{registry}/blobs/{descriptor['digest']}", token=token)
        assert "sha256:" + sha256(value) == descriptor["digest"]
        assert len(value) == descriptor["size"]
        blobs[descriptor["digest"].split(":", 1)[1]] = value
    manifest_key = AMD64_MANIFEST_DIGEST.split(":", 1)[1]
    blobs[manifest_key] = manifest_bytes

    nested = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.index.v1+json",
        "manifests": [{
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "digest": AMD64_MANIFEST_DIGEST,
            "size": len(manifest_bytes),
            "platform": {"architecture": "amd64", "os": "linux"},
        }],
        "annotations": {
            "org.opencontainers.image.source": f"docker.io/{REPOSITORY}",
            "org.opencontainers.image.base.digest": INDEX_DIGEST,
        },
    }
    nested_bytes = json.dumps(nested, sort_keys=True, separators=(",", ":")).encode()
    nested_digest = "sha256:" + sha256(nested_bytes)
    blobs[nested_digest.split(":", 1)[1]] = nested_bytes
    index = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.index.v1+json",
        "manifests": [{
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "digest": nested_digest,
            "size": len(nested_bytes),
            "annotations": {
                "io.containerd.image.name": f"docker.io/{LOCAL_TAG}",
                "org.opencontainers.image.ref.name": "offline-pinned-amd64",
            },
        }],
    }
    docker_manifest = [{
        "Config": "blobs/sha256/" + manifest["config"]["digest"].split(":", 1)[1],
        "RepoTags": [LOCAL_TAG],
        "Layers": [
            "blobs/sha256/" + layer["digest"].split(":", 1)[1]
            for layer in manifest["layers"]
        ],
    }]
    with tarfile.open(OUTPUT_TAR, "w", format=tarfile.PAX_FORMAT) as archive:
        add(archive, "oci-layout", b'{"imageLayoutVersion":"1.0.0"}\n')
        add(archive, "index.json", json.dumps(index, sort_keys=True, separators=(",", ":")).encode())
        add(archive, "manifest.json", json.dumps(docker_manifest, sort_keys=True, separators=(",", ":")).encode())
        for key, value in sorted(blobs.items()):
            add(archive, f"blobs/sha256/{key}", value)
    digest = hashlib.sha256()
    with OUTPUT_TAR.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    checksum = digest.hexdigest()
    (OUTPUT_TAR.with_suffix(".tar.sha256")).write_text(
        f"{checksum}  {OUTPUT_TAR.name}\n", encoding="ascii", newline="\n"
    )
    receipt = {
        "source": f"docker.io/{REPOSITORY}@{INDEX_DIGEST}",
        "platform": "linux/amd64",
        "platform_manifest_digest": AMD64_MANIFEST_DIGEST,
        "local_tag": LOCAL_TAG,
        "archive": OUTPUT_TAR.name,
        "archive_sha256": checksum,
        "archive_bytes": OUTPUT_TAR.stat().st_size,
        "config_digest": manifest["config"]["digest"],
        "layer_digests": [layer["digest"] for layer in manifest["layers"]],
        "complete_blob_count": len(descriptors),
    }
    (OUTPUT_DIR / "image_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
