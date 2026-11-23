"""Freeze the exact v1.1 code/config/prompt/evaluator/oracle campaign snapshot."""

from __future__ import annotations

import json
from pathlib import Path

from tools.campaign import (
    CAMPAIGN_IDENTITY,
    campaign_file_entries,
    content_commitment,
)
from tools.science import canonical_json_bytes, sha256_file


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "configs" / "campaign_snapshot.json"
def main() -> int:
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError("CAMPAIGN_SNAPSHOT_ALREADY_EXISTS")
    entries = campaign_file_entries(ROOT)
    if not entries:
        raise RuntimeError("CAMPAIGN_SNAPSHOT_EMPTY")
    forbidden = [
        item["path"]
        for item in entries
        if str(item["path"]).casefold().endswith(".env")
        or "trajectory" in Path(str(item["path"])).parts
    ]
    if forbidden:
        raise RuntimeError("CAMPAIGN_SNAPSHOT_FORBIDDEN_INPUT")
    payload = {
        "schema_version": "1.0",
        **CAMPAIGN_IDENTITY,
        "file_count": len(entries),
        "total_bytes": sum(int(item["size_bytes"]) for item in entries),
        "content_commitment_sha256": content_commitment(entries),
        "files": entries,
    }
    with OUTPUT.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))
    print(
        json.dumps(
            {
                "status": "PASS",
                "path": OUTPUT.relative_to(ROOT).as_posix(),
                "file_count": len(entries),
                "content_commitment_sha256": payload[
                    "content_commitment_sha256"
                ],
                "snapshot_file_sha256": sha256_file(OUTPUT),
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
