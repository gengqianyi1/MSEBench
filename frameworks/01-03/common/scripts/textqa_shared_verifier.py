"""Host-side lifecycle for one persistent isolated TextQA verifier."""
from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFIER_DIR = ROOT / "textqa" / "shared_verifier"


def _image() -> str:
    digest = hashlib.sha256(b"".join((VERIFIER_DIR / name).read_bytes() for name in ("Dockerfile", "batch_grader.py"))).hexdigest()[:16]
    return f"chembench/textqa-shared-verifier:{digest}"


def make_private_index(dataset: Path, destination: Path) -> Path:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    index = {}
    for entry in manifest["tasks"]:
        ref = json.loads((dataset / entry["task_directory"] / "tests" / "private_reference.json").read_text(encoding="utf-8"))
        index[entry["task_name"]] = {key: ref[key] for key in ("question_type", "answer")}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    return destination


class SharedTextQAVerifier:
    def __init__(self, private_index: Path):
        self.image = _image()
        if subprocess.run(["docker", "image", "inspect", self.image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            subprocess.run(["docker", "build", "-t", self.image, str(VERIFIER_DIR)], check=True)
        self.process = subprocess.Popen([
            "docker", "run", "--rm", "-i", "-v", f"{private_index.resolve()}:/private/reference_index.json:ro", self.image,
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def grade(self, task_id: str, answer: str) -> dict:
        assert self.process.stdin and self.process.stdout
        self.process.stdin.write(json.dumps({"task_id": task_id, "answer": answer}, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError("shared verifier exited without a response")
        return json.loads(response)

    def close(self) -> None:
        if self.process.stdin:
            self.process.stdin.close()
        self.process.wait(timeout=15)
