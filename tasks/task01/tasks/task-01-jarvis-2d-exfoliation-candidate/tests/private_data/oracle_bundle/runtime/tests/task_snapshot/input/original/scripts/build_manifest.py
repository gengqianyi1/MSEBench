from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".git"}
LOCAL_CONFIGS = {
    "api_config.json",
    "YOUR_API_KEY.json",
    "api_config_kimi.json",
    "api_config_qwen.json",
    "evaluator_api_config.json",
}
EXCLUDED_FILES = {
    "docker_build.stdout.log",
    "docker_build.stderr.log",
    "psi4_sapt_smoke.log",
    "timer.dat",
    "output.dat",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def publishable(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_DIRS for part in relative.parts):
        return False
    if relative.parts[0] == "runs":
        return str(relative).replace("\\", "/") == "runs/.gitkeep"
    if relative.parts[0] == "private" and path.suffixes[-2:] == [".json", ".gz"]:
        return False
    if path.suffix == ".pyc" or path.name in EXCLUDED_FILES:
        return False
    if path.parent.name == "configs" and (
        path.name in LOCAL_CONFIGS
        or ("api_config" in path.name and not path.name.endswith(".example.json"))
    ):
        return False
    return True


def main() -> None:
    manifest_path = ROOT / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    files = sorted(
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in ROOT.rglob("*")
        if path.is_file() and publishable(path)
    )
    manifest["files"] = files
    manifest["sha256"] = {
        relative: digest(ROOT / relative)
        for relative in files
        if relative != "MANIFEST.json"
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"updated {manifest_path} with {len(files)} files")


if __name__ == "__main__":
    main()
