"""Package-safe v2 contract for proving historical releases stayed immutable.

The live recomputation reads the historical sibling projects, but the emitted
receipt contains only deterministic counts and SHA-256 commitments.  Delivery
validation can therefore verify the closed receipt without requiring access
to organizer-local sibling directories.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping

from package_tools.provenance import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


HISTORICAL_INTEGRITY_SCHEMA_VERSION = "historical-integrity/2.0"
TREE_ALGORITHM = (
    "tree-sha256-v1: ordinal(relative_path + NUL + size + "
    "NUL + file_sha256 + LF)"
)
V1_PROJECT_BASELINE: dict[str, int | str] = {
    "file_count": 24523,
    "total_bytes": 163815800,
    "tree_sha256": (
        "1062B61D2E4580C534591A3AE0A9CD7AEA598A60F021C661DDED47B1FE06A185"
    ),
}
V1_ARCHIVE_BASELINE: dict[str, int | str] = {
    "size_bytes": 3827793,
    "sha256": (
        "F96C2938473B1791F9F027CA80E140C550C345F6C978C6C7769A82EC59F053DE"
    ),
}
V1_1_PROJECT_BASELINE: dict[str, int | str] = {
    "file_count": 6618,
    "total_bytes": 42086685,
    "tree_sha256": (
        "4EE17DBBC189E2446FD38CA8FE0CD469027F04F27B49F540BE0ED183A0DAD0A4"
    ),
}
V1_1_CAMPAIGN_IDENTITY: dict[str, str] = {
    "benchmark_version": "1.1.0",
    "case_id": "2_fluorobiphenyl_central_torsion_v1",
    "formal_campaign_id": "formal_campaign_v1_1_rc1",
    "protocol": "adaptive_torsion_sampling_tool_loop/1.1",
    "task_id": "adaptive_xtb_torsion_profile_agent_v1_1",
}
V1_1_CAMPAIGN_BASELINE: dict[str, int | str] = {
    "file_count": 180,
    "total_bytes": 1300479,
    "content_commitment_sha256": (
        "ADBE3C9CCD50BDEA3ED3F8C13541C6729AC3CA4BE651200C74E831DFB8C0C3D9"
    ),
}


class HistoricalIntegrityError(ValueError):
    """Historical integrity evidence is missing, malformed, or mismatched."""


def _is_link_or_reparse(path: Path) -> bool:
    attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    is_junction = getattr(path, "is_junction", None)
    return bool(
        path.is_symlink()
        or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        or (callable(is_junction) and is_junction())
    )


def tree_commitment(root: Path) -> dict[str, int | str]:
    """Recompute the frozen ordinal tree commitment without writing files."""

    if _is_link_or_reparse(root):
        raise HistoricalIntegrityError("HISTORICAL_TREE_LINK_OR_REPARSE")
    resolved = root.resolve(strict=True)
    files: list[Path] = []
    for path in resolved.rglob("*"):
        if _is_link_or_reparse(path):
            raise HistoricalIntegrityError("HISTORICAL_TREE_LINK_OR_REPARSE")
        if path.is_file():
            files.append(path)
        elif path.exists() and not path.is_dir():
            raise HistoricalIntegrityError("HISTORICAL_TREE_SPECIAL_FILE")
    files.sort(key=lambda path: path.relative_to(resolved).as_posix())
    records: list[str] = []
    total_bytes = 0
    for path in files:
        relative = path.relative_to(resolved).as_posix()
        size = path.stat().st_size
        records.append(f"{relative}\0{size}\0{sha256_file(path)}\n")
        total_bytes += size
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "tree_sha256": hashlib.sha256(
            "".join(records).encode("utf-8")
        ).hexdigest().upper(),
    }


def _closed_mapping(
    value: Any,
    fields: set[str],
    error: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise HistoricalIntegrityError(error)
    return value


def _campaign_snapshot(project_root: Path) -> dict[str, int | str]:
    """Validate every frozen v1.1 campaign member against its snapshot."""

    snapshot_path = project_root / "configs" / "campaign_snapshot.json"
    try:
        raw = snapshot_path.read_bytes()
        snapshot = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HistoricalIntegrityError(
            "HISTORICAL_V1_1_CAMPAIGN_SNAPSHOT_INVALID"
        ) from exc
    if not isinstance(snapshot, dict):
        raise HistoricalIntegrityError(
            "HISTORICAL_V1_1_CAMPAIGN_SNAPSHOT_INVALID"
        )
    for field, expected in V1_1_CAMPAIGN_IDENTITY.items():
        if snapshot.get(field) != expected:
            raise HistoricalIntegrityError(
                "HISTORICAL_V1_1_CAMPAIGN_IDENTITY_MISMATCH"
            )
    raw_entries = snapshot.get("files")
    if not isinstance(raw_entries, list):
        raise HistoricalIntegrityError(
            "HISTORICAL_V1_1_CAMPAIGN_FILES_INVALID"
        )
    entries: list[dict[str, int | str]] = []
    names: set[str] = set()
    folded_names: set[str] = set()
    resolved = project_root.resolve(strict=True)
    for raw_entry in raw_entries:
        entry = _closed_mapping(
            raw_entry,
            {"path", "sha256", "size_bytes"},
            "HISTORICAL_V1_1_CAMPAIGN_ENTRY_INVALID",
        )
        relative = entry["path"]
        size = entry["size_bytes"]
        digest = entry["sha256"]
        if not isinstance(relative, str):
            raise HistoricalIntegrityError(
                "HISTORICAL_V1_1_CAMPAIGN_PATH_INVALID"
            )
        pure = PurePosixPath(relative)
        if (
            not relative
            or "\\" in relative
            or pure.is_absolute()
            or pure.as_posix() != relative
            or any(part in {"", ".", ".."} or ":" in part for part in pure.parts)
            or relative in names
            or relative.casefold() in folded_names
        ):
            raise HistoricalIntegrityError(
                "HISTORICAL_V1_1_CAMPAIGN_PATH_INVALID"
            )
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789ABCDEF" for character in digest)
        ):
            raise HistoricalIntegrityError(
                "HISTORICAL_V1_1_CAMPAIGN_ENTRY_INVALID"
            )
        path = resolved.joinpath(*pure.parts)
        try:
            path.resolve(strict=True).relative_to(resolved)
        except (OSError, ValueError) as exc:
            raise HistoricalIntegrityError(
                "HISTORICAL_V1_1_CAMPAIGN_PATH_ESCAPE"
            ) from exc
        if _is_link_or_reparse(path):
            raise HistoricalIntegrityError(
                "HISTORICAL_V1_1_CAMPAIGN_LINK_OR_REPARSE"
            )
        if (
            not path.is_file()
            or path.stat().st_size != size
            or sha256_file(path) != digest
        ):
            raise HistoricalIntegrityError(
                "HISTORICAL_V1_1_CAMPAIGN_FILE_MISMATCH"
            )
        names.add(relative)
        folded_names.add(relative.casefold())
        entries.append(
            {"path": relative, "sha256": digest, "size_bytes": size}
        )
    if entries != sorted(entries, key=lambda item: str(item["path"])):
        raise HistoricalIntegrityError(
            "HISTORICAL_V1_1_CAMPAIGN_ORDER_INVALID"
        )
    commitment = sha256_bytes(
        canonical_json_bytes(
            {"identity": V1_1_CAMPAIGN_IDENTITY, "files": entries}
        )
    )
    total_bytes = sum(int(item["size_bytes"]) for item in entries)
    if (
        snapshot.get("file_count") != len(entries)
        or snapshot.get("total_bytes") != total_bytes
        or snapshot.get("content_commitment_sha256") != commitment
    ):
        raise HistoricalIntegrityError(
            "HISTORICAL_V1_1_CAMPAIGN_COMMITMENT_MISMATCH"
        )
    return {
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "content_commitment_sha256": commitment,
    }


def build_historical_integrity_receipt(parent_root: Path) -> dict[str, Any]:
    """Recompute v1, v1 archive, v1.1 tree, and v1.1 campaign evidence."""

    parent = parent_root.resolve(strict=True)
    v1 = tree_commitment(parent / "adaptive_xtb_torsion_profile_agent_v1")
    v1_archive_path = (
        parent / "LiuChengchun_Adaptive_xTB_Torsion_Profile_Agent_Benchmark.zip"
    )
    if _is_link_or_reparse(v1_archive_path):
        raise HistoricalIntegrityError("HISTORICAL_ARCHIVE_LINK_OR_REPARSE")
    v1_archive = {
        "size_bytes": v1_archive_path.stat().st_size,
        "sha256": sha256_file(v1_archive_path),
    }
    v1_1_root = parent / "adaptive_xtb_torsion_profile_agent_v1_1"
    v1_1 = tree_commitment(v1_1_root)
    campaign = _campaign_snapshot(v1_1_root)
    matches = {
        "v1": v1 == V1_PROJECT_BASELINE,
        "v1_archive": v1_archive == V1_ARCHIVE_BASELINE,
        "v1_1": v1_1 == V1_1_PROJECT_BASELINE,
        "v1_1_campaign": {
            key: campaign[key] for key in V1_1_CAMPAIGN_BASELINE
        }
        == V1_1_CAMPAIGN_BASELINE,
    }
    payload = {
        "schema_version": HISTORICAL_INTEGRITY_SCHEMA_VERSION,
        "status": "PASS" if all(matches.values()) else "FAIL",
        "algorithm": TREE_ALGORITHM,
        "historical_write_operations": False,
        "v1": {**v1, "matches_pre_work": matches["v1"]},
        "v1_archive": {
            **v1_archive,
            "matches_pre_work": matches["v1_archive"],
        },
        "v1_1": {
            **v1_1,
            "matches_pre_work": matches["v1_1"],
            "campaign": {
                **campaign,
                "matches_pre_work": matches["v1_1_campaign"],
            },
        },
    }
    return payload


def validate_historical_integrity_receipt(payload: Any) -> dict[str, Any]:
    """Validate a closed, path-free historical integrity receipt."""

    value = _closed_mapping(
        payload,
        {
            "algorithm",
            "historical_write_operations",
            "schema_version",
            "status",
            "v1",
            "v1_1",
            "v1_archive",
        },
        "HISTORICAL_INTEGRITY_SCHEMA",
    )
    if (
        value["schema_version"] != HISTORICAL_INTEGRITY_SCHEMA_VERSION
        or value["algorithm"] != TREE_ALGORITHM
        or value["status"] != "PASS"
        or value["historical_write_operations"] is not False
    ):
        raise HistoricalIntegrityError("HISTORICAL_INTEGRITY_STATUS")
    v1 = _closed_mapping(
        value["v1"],
        {"file_count", "matches_pre_work", "total_bytes", "tree_sha256"},
        "HISTORICAL_V1_SCHEMA",
    )
    v1_archive = _closed_mapping(
        value["v1_archive"],
        {"matches_pre_work", "sha256", "size_bytes"},
        "HISTORICAL_V1_ARCHIVE_SCHEMA",
    )
    v1_1 = _closed_mapping(
        value["v1_1"],
        {
            "campaign",
            "file_count",
            "matches_pre_work",
            "total_bytes",
            "tree_sha256",
        },
        "HISTORICAL_V1_1_SCHEMA",
    )
    campaign = _closed_mapping(
        v1_1["campaign"],
        {
            "content_commitment_sha256",
            "file_count",
            "matches_pre_work",
            "total_bytes",
        },
        "HISTORICAL_V1_1_CAMPAIGN_SCHEMA",
    )
    if (
        {key: v1[key] for key in V1_PROJECT_BASELINE}
        != V1_PROJECT_BASELINE
        or v1["matches_pre_work"] is not True
        or {key: v1_archive[key] for key in V1_ARCHIVE_BASELINE}
        != V1_ARCHIVE_BASELINE
        or v1_archive["matches_pre_work"] is not True
        or {key: v1_1[key] for key in V1_1_PROJECT_BASELINE}
        != V1_1_PROJECT_BASELINE
        or v1_1["matches_pre_work"] is not True
        or {
            key: campaign[key] for key in V1_1_CAMPAIGN_BASELINE
        }
        != V1_1_CAMPAIGN_BASELINE
        or campaign["matches_pre_work"] is not True
    ):
        raise HistoricalIntegrityError("HISTORICAL_INTEGRITY_MISMATCH")
    return {
        "passed": True,
        "v1_tree_sha256": v1["tree_sha256"],
        "v1_archive_sha256": v1_archive["sha256"],
        "v1_1_tree_sha256": v1_1["tree_sha256"],
        "v1_1_campaign_commitment": campaign[
            "content_commitment_sha256"
        ],
    }


__all__ = [
    "HISTORICAL_INTEGRITY_SCHEMA_VERSION",
    "HistoricalIntegrityError",
    "TREE_ALGORITHM",
    "V1_1_CAMPAIGN_BASELINE",
    "V1_1_PROJECT_BASELINE",
    "V1_ARCHIVE_BASELINE",
    "V1_PROJECT_BASELINE",
    "build_historical_integrity_receipt",
    "tree_commitment",
    "validate_historical_integrity_receipt",
]
