"""Fetch the reviewed dataset files from a pinned Hugging Face commit."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
from urllib.parse import quote
from urllib.request import urlopen


def download(repo: str, revision: str, manifest: Path, destination: Path) -> None:
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo, flags=re.ASCII):
        raise ValueError("Use an owner/dataset Hugging Face repository ID")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use the full Hugging Face dataset commit SHA")
    document = json.loads(manifest.read_text(encoding="utf-8"))
    if document.get("release_mode") == "local-evaluation":
        from prepare_local import prepare

        prepare(manifest.parent, destination, repo=repo, revision=revision)
        return
    tasks = document["tasks"]
    if not tasks:
        raise ValueError("The manifest has no approved datasets")
    root = destination.resolve()
    for task in tasks:
        for record in task["files"]:
            relative = Path(record["path"])
            target = root / relative
            if relative.is_absolute() or ".." in relative.parts or target.resolve() != target:
                raise ValueError(f"Unsafe dataset path: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(relative.as_posix())}"
            with tempfile.TemporaryDirectory(dir=target.parent) as temp:
                partial = Path(temp) / "download"
                digest = hashlib.sha256()
                size = 0
                with urlopen(url, timeout=60) as response, partial.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > record["bytes"]:
                            raise ValueError(f"Downloaded file exceeds manifest size: {relative}")
                        digest.update(chunk)
                        handle.write(chunk)
                if size != record["bytes"] or digest.hexdigest() != record["sha256"]:
                    raise ValueError(f"Downloaded file does not match manifest: {relative}")
                partial.replace(target)
            print(f"Verified {relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Hugging Face owner/dataset")
    parser.add_argument("--revision", required=True, help="Full dataset commit SHA")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "data_manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        download(args.repo, args.revision, args.manifest, args.output)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Dataset download failed: {exc}\n")


if __name__ == "__main__":
    main()
