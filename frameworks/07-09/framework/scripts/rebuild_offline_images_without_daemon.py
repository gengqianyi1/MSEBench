#!/usr/bin/env python3
"""Re-layer Tasks 07-09 OCI images from the frozen source tree, without Docker.

This is a deterministic fallback for a host whose Docker daemon is unavailable.
It preserves the previously frozen base layers and appends one source overlay,
then rewrites the OCI descriptors and Docker-load compatibility manifest.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent / "docker" / "organizer_images_7_9.tar"
MANIFEST_PATH = ROOT.parent / "docker" / "image_manifest.json"
CREATED = "2026-09-10T14:00:00Z"
TAG_SUFFIX = "20260910-sixth-review"


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def allowed(path: Path) -> bool:
    blocked = {"__pycache__", ".pytest_cache", ".test_tmp"}
    return not any(part in blocked for part in path.parts) and path.suffix != ".pyc"


def overlay_sources(task: int, role: str) -> list[tuple[Path, str, int]]:
    task_root = ROOT / "tasks" / {
        7: "task-07-qed-molecular-design",
        8: "task-08-xtb-failure-diagnosis-restart",
        9: "task-09-adaptive-xtb-torsion-profile",
    }[task]
    if role == "environment":
        return [(task_root / "environment" / "public", "workspace", 57439)]
    tests = task_root / "tests"
    return [
        (tests / "task_snapshot", "task", 57439),
        (tests / "run_legacy_grader.py", "tests/run_legacy_grader.py", 57439),
        (tests / "protocol_grader.py", "tests/protocol_grader.py", 57439),
        (tests / "private_suite_adapter.py", "tests/private_suite_adapter.py", 57439),
        (tests / "test.sh", "tests/test.sh", 57439),
        (tests / "hidden_suite_server.py", "opt/matbench/hidden_suite_server.py", 0),
    ]


def make_layer(task: int, role: str) -> tuple[bytes, bytes]:
    raw_buffer = io.BytesIO()
    with tarfile.open(fileobj=raw_buffer, mode="w", format=tarfile.USTAR_FORMAT) as layer:
        for source, target, owner in overlay_sources(task, role):
            candidates = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
            for path in candidates:
                if not allowed(path):
                    continue
                relative = "" if source.is_file() else path.relative_to(source).as_posix()
                name = target if source.is_file() else f"{target}/{relative}"
                data = path.read_bytes()
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                info.mtime = 0
                info.uid = owner
                info.gid = owner
                info.uname = "root" if owner == 0 else "mambauser"
                info.gname = info.uname
                info.mode = 0o755 if path.suffix == ".sh" else 0o644
                layer.addfile(info, io.BytesIO(data))
    raw = raw_buffer.getvalue()
    zipped = io.BytesIO()
    with gzip.GzipFile(fileobj=zipped, mode="wb", compresslevel=9, mtime=0) as handle:
        handle.write(raw)
    return raw, zipped.getvalue()


def read_json(archive: tarfile.TarFile, name: str) -> object:
    handle = archive.extractfile(name)
    if handle is None:
        raise RuntimeError(f"missing archive member: {name}")
    return json.load(handle)


def blob_json(archive: tarfile.TarFile, descriptor: dict) -> object:
    return read_json(archive, "blobs/sha256/" + descriptor["digest"].split(":", 1)[1])


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


def main() -> int:
    output = ARCHIVE.with_suffix(".sixth-review.tmp.tar")
    metadata = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    new_blobs: dict[str, bytes] = {}
    updates: dict[str, dict] = {}

    with tarfile.open(ARCHIVE, "r") as source:
        index = read_json(source, "index.json")
        docker_manifest = read_json(source, "manifest.json")
        for descriptor in index["manifests"]:
            full_name = descriptor.get("annotations", {}).get("io.containerd.image.name", "")
            short_tag = full_name.removeprefix("docker.io/")
            match = next((x for x in metadata["images"] if x.get("tag") == short_tag), None)
            if not match or match.get("task") not in {7, 8, 9} or match.get("role") not in {"sidecar", "environment"}:
                continue
            task, role = int(match["task"]), match["role"]
            nested = blob_json(source, descriptor)
            platform_descriptor = next(x for x in nested["manifests"] if x.get("platform", {}).get("architecture") == "amd64")
            image_manifest = blob_json(source, platform_descriptor)
            config = blob_json(source, image_manifest["config"])
            raw_layer, compressed_layer = make_layer(task, role)
            raw_digest, layer_digest = digest(raw_layer), digest(compressed_layer)
            new_blobs[layer_digest] = compressed_layer
            config = copy.deepcopy(config)
            config["created"] = CREATED
            config["rootfs"]["diff_ids"].append("sha256:" + raw_digest)
            config.setdefault("history", []).append({
                "created": CREATED,
                "created_by": "MatBench sixth-review deterministic source overlay",
                "comment": "daemonless OCI relayer",
            })
            config_bytes = canonical(config)
            config_digest = digest(config_bytes)
            new_blobs[config_digest] = config_bytes
            image_manifest = copy.deepcopy(image_manifest)
            image_manifest["config"] = {
                **image_manifest["config"], "digest": "sha256:" + config_digest, "size": len(config_bytes)
            }
            media_type = image_manifest["layers"][-1]["mediaType"]
            image_manifest["layers"].append({
                "mediaType": media_type,
                "digest": "sha256:" + layer_digest,
                "size": len(compressed_layer),
            })
            image_manifest_bytes = canonical(image_manifest)
            image_manifest_digest = digest(image_manifest_bytes)
            new_blobs[image_manifest_digest] = image_manifest_bytes
            nested = copy.deepcopy(nested)
            nested["manifests"] = [{
                **platform_descriptor,
                "digest": "sha256:" + image_manifest_digest,
                "size": len(image_manifest_bytes),
            }]
            nested_bytes = canonical(nested)
            image_digest = digest(nested_bytes)
            new_blobs[image_digest] = nested_bytes
            descriptor.update(digest="sha256:" + image_digest, size=len(nested_bytes))
            new_tag = short_tag.split(":", 1)[0] + ":" + TAG_SUFFIX
            descriptor["annotations"]["io.containerd.image.name"] = "docker.io/" + new_tag
            descriptor["annotations"]["org.opencontainers.image.ref.name"] = TAG_SUFFIX
            docker_entry = next(x for x in docker_manifest if x.get("RepoTags") == [short_tag])
            docker_entry["Config"] = "blobs/sha256/" + config_digest
            docker_entry["RepoTags"] = [new_tag]
            docker_entry["Layers"].append("blobs/sha256/" + layer_digest)
            updates[short_tag] = {
                "tag": new_tag,
                "image_digest": "sha256:" + image_digest,
                "config_digest": "sha256:" + config_digest,
                "diff_id": "sha256:" + raw_digest,
                "raw_size": len(raw_layer),
            }

        index_bytes = canonical(index)
        docker_manifest_bytes = canonical(docker_manifest)
        skip = {"index.json", "manifest.json"} | {"blobs/sha256/" + key for key in new_blobs}
        with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as target:
            for member in source:
                if member.name in skip:
                    continue
                handle = source.extractfile(member) if member.isfile() else None
                target.addfile(member, handle)
            add_bytes(target, "index.json", index_bytes)
            add_bytes(target, "manifest.json", docker_manifest_bytes)
            for key, data in sorted(new_blobs.items()):
                add_bytes(target, "blobs/sha256/" + key, data)

    os.replace(output, ARCHIVE)
    archive_digest = hashlib.sha256()
    with ARCHIVE.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            archive_digest.update(block)
    archive_hash = archive_digest.hexdigest()
    metadata["archive_sha256"] = archive_hash
    metadata["archive_bytes"] = ARCHIVE.stat().st_size
    metadata["rebuild_method"] = "daemonless_deterministic_oci_overlay"
    metadata["rebuild_timestamp_utc"] = CREATED
    for image in metadata["images"]:
        old_tag = image.get("tag")
        if old_tag not in updates:
            continue
        update = updates[old_tag]
        image["tag"] = update["tag"]
        image["image_digest"] = update["image_digest"]
        image["immutable_reference"] = update["tag"].split(":", 1)[0] + "@" + update["image_digest"]
        inspect = image["image_inspect"]
        inspect["Id"] = update["config_digest"]
        inspect["Created"] = CREATED
        inspect["RepoTags"] = [update["tag"]]
        inspect["RepoDigests"] = [image["immutable_reference"]]
        inspect["Size"] = int(inspect["Size"]) + update["raw_size"]
        inspect["RootFS"]["Layers"].append(update["diff_id"])
    MANIFEST_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    checksum_path = ARCHIVE.with_suffix(ARCHIVE.suffix + ".sha256")
    checksum_path.write_text(f"{archive_hash}  {ARCHIVE.name}\n", encoding="ascii", newline="\n")
    print(json.dumps(updates, indent=2))
    print(f"archive_sha256={archive_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
