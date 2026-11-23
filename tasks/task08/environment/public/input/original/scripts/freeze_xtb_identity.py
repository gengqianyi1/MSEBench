"""Create a reviewed, platform-aware xTB identity receipt.

``freeze`` refuses to replace an existing receipt. ``refreeze`` requires an
existing receipt and writes a separate candidate for explicit review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.preflight_xtb import ROOT, collect_preflight
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from preflight_xtb import ROOT, collect_preflight


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _below_root(relative: str) -> Path:
    target = (ROOT / relative).resolve()
    try:
        target.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("receipt path must remain below the project root") from exc
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "refreeze"))
    parser.add_argument("--config", default="configs/task.yaml")
    parser.add_argument("--current", default="configs/xtb_identity_receipt.json")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    current = _below_root(args.current)
    if args.mode == "freeze":
        if current.exists():
            raise FileExistsError(f"receipt already exists: {current.relative_to(ROOT)}")
        output = _below_root(args.output or args.current)
        previous_present = False
    else:
        if not current.is_file():
            raise FileNotFoundError(f"current receipt is required: {current.relative_to(ROOT)}")
        json.loads(current.read_text(encoding="utf-8"))
        output = _below_root(args.output or "validation/xtb_identity_refreeze_candidate.json")
        previous_present = True

    if output.exists():
        raise FileExistsError(f"refusing to replace: {output.relative_to(ROOT)}")
    preflight = collect_preflight(_below_root(args.config))
    if not preflight.get("passed"):
        print(json.dumps(preflight, ensure_ascii=True, indent=2, sort_keys=True))
        return 1

    xtb = preflight["xtb"]
    receipt = {
        "schema_version": "xtb_identity_receipt/2.0",
        "identity_policy": "xtb_portable_identity/2.0",
        "mode": args.mode,
        "portable_contract": {
            "version": xtb["version"],
            "required_help_tokens": xtb["required_help_tokens"],
            "version_output_sha256": xtb["version_output_sha256"],
            "help_output_sha256": xtb["help_output_sha256"],
        },
        "platform_observation": {
            "operating_system": preflight["runtime"]["operating_system"],
            "architecture": preflight["runtime"]["architecture"],
            "binary_sha256": xtb["binary_sha256"],
            "executable_alias": xtb["executable_alias"],
        },
        "previous_receipt_present": previous_present,
        "review_required_before_promotion": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(receipt))
    print(json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
