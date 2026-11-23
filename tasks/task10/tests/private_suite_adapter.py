"""Organizer-only evidence adapter for a single-step Harbor task.

The evaluated model never imports this module.  The verifier first snapshots the
allowlisted workspace artifacts, then independently replays the unchanged private
scientific grader against that immutable evidence directory.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import yaml


MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
ARTIFACT_ALLOWLIST = {
    "task_10_matbench_bandgap_pipeline": (
        "final_answer.json", "predictions.csv", "pipeline.json", "training_script.py"
    ),
    "task_11_lithium_electrolyte_design": ("final_answer.json", "candidate.json"),
    "task_12_pybamm_fast_charge": ("final_answer.json", "candidate.json"),
    "task_13_retrosynthesis_route": ("final_answer.json", "route.json"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Adapter:
    """Capture and replay only declared single-step submission artifacts."""

    def __init__(self, snapshot: Path | str):
        self.snapshot = Path(snapshot).resolve()
        metadata = yaml.safe_load((self.snapshot / "task.yaml").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise TypeError("task.yaml must contain an object")
        self.metadata = metadata
        self.task_id = str(metadata["task_id"])
        try:
            self.allowlist = ARTIFACT_ALLOWLIST[self.task_id]
        except KeyError as exc:
            raise ValueError(f"no evidence allowlist for {self.task_id}") from exc

    def capture_evidence(self, workspace: Path | str, evidence_dir: Path | str) -> dict:
        workspace = Path(workspace).resolve()
        evidence_dir = Path(evidence_dir).resolve()
        evidence_dir.mkdir(parents=True, exist_ok=True)
        submission = evidence_dir / "submission"
        submission.mkdir()
        artifacts = []
        for relative in self.allowlist:
            source = (workspace / relative).resolve()
            if source != workspace / relative or source.is_symlink() or not source.is_file():
                continue
            size = source.stat().st_size
            if size > MAX_ARTIFACT_BYTES:
                raise ValueError(f"artifact_too_large:{relative}")
            destination = submission / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            artifacts.append({"path": relative, "size": size, "sha256": _sha256(destination)})
        manifest = {
            "schema_version": "chembench-organizer-evidence/1.0",
            "task_id": self.task_id,
            "source": "harbor_workspace_allowlist_snapshot",
            "artifacts": artifacts,
        }
        canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        manifest["evidence_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        (evidence_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest

    def grade_evidence(self, evidence_dir: Path | str, run_dir: Path | str) -> dict:
        evidence_dir = Path(evidence_dir).resolve()
        manifest = json.loads((evidence_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "chembench-organizer-evidence/1.0":
            raise ValueError("invalid_evidence_schema")
        if manifest.get("task_id") != self.task_id:
            raise ValueError("evidence_task_mismatch")
        submission = (evidence_dir / "submission").resolve()
        declared = set()
        for row in manifest.get("artifacts", []):
            relative = row.get("path")
            if relative not in self.allowlist or relative in declared:
                raise ValueError("invalid_evidence_artifact")
            declared.add(relative)
            path = (submission / relative).resolve()
            if path != submission / relative or path.is_symlink() or not path.is_file():
                raise ValueError("missing_evidence_artifact")
            if path.stat().st_size != row.get("size") or _sha256(path) != row.get("sha256"):
                raise ValueError("evidence_digest_mismatch")
        actual = {
            path.relative_to(submission).as_posix()
            for path in submission.rglob("*")
            if path.is_file()
        }
        if actual != declared:
            raise ValueError("undeclared_evidence_artifact")
        grader_path = (self.snapshot / str(self.metadata.get("grader_path", "private_grader/grader.py"))).resolve()
        if self.snapshot not in grader_path.parents or not grader_path.is_file():
            raise ValueError("invalid_private_grader_path")
        module_name = "chembench_private_replay_" + hashlib.sha256(str(grader_path).encode()).hexdigest()[:12]
        spec = importlib.util.spec_from_file_location(module_name, grader_path)
        if spec is None or spec.loader is None:
            raise ImportError("cannot_load_private_grader")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        context = SimpleNamespace(
            task_dir=self.snapshot,
            submission_dir=submission,
            private_data_dir=(self.snapshot / str(self.metadata.get("private_data_path", "private_data"))).resolve(),
            run_dir=Path(run_dir).resolve(),
        )
        result = getattr(module, str(self.metadata.get("grader_entrypoint", "grade")))(context)
        if not isinstance(result, dict):
            raise TypeError("private grader must return an object")
        score = result.get("score")
        minimum = float(self.metadata.get("min_score", 0.0))
        maximum = float(self.metadata.get("max_score", 100.0))
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError("private grader returned a non-finite score")
        if float(score) < minimum or float(score) > maximum:
            raise ValueError("private grader returned an out-of-range score")
        return result


def create_adapter(snapshot: Path | str) -> Adapter:
    return Adapter(snapshot)
