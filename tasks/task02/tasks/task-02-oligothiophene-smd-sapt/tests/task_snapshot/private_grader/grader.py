from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import shutil
from pathlib import Path

import yaml


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grade(context):
    """Replay deterministic scoring after public Agent execution has ended."""
    if context.private_data_dir is None:
        raise RuntimeError("Private grader requires private_data_dir")

    private_reference = context.private_data_dir / "scoring_reference.json"
    if not private_reference.is_file():
        raise FileNotFoundError(private_reference)

    source_root = context.task_dir / "input" / "original" / "src"
    sys.path.insert(0, str(source_root))
    scoring = _load_module("task02_private_scoring", source_root / "scoring.py")
    instance_generator = _load_module("task02_private_instance_generator", source_root / "instance_generator.py")

    result_path = context.submission_dir / "result.json"
    if not result_path.is_file():
        return {
            "score": 0.0,
            "subscores": {"deterministic": 0.0, "independent_scientific": 0.0},
            "failure_codes": ["MISSING_RESULT_JSON"],
            "evidence": {"path": "result.json"},
        }
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            "score": 0.0,
            "subscores": {"deterministic": 0.0, "independent_scientific": 0.0},
            "failure_codes": ["INVALID_RESULT_JSON"],
            "evidence": {"error": str(exc)},
        }
    if not isinstance(result, dict):
        return {
            "score": 0.0,
            "subscores": {"deterministic": 0.0, "independent_scientific": 0.0},
            "failure_codes": ["INVALID_RESULT_SCHEMA"],
            "evidence": {"expected": "JSON object"},
        }
    task_cfg_path = context.task_dir / "input" / "original" / "tasks" / "oligothiophene_smd_sapt_v3" / "task_config.yaml"
    task_cfg = yaml.safe_load(task_cfg_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="task02-private-replay-") as tmp:
        native_root=Path(tmp)/"native"
        native_root.mkdir()
        native_workspace=native_root/"workspace"
        shutil.copytree(context.submission_dir,native_workspace)
        public_instance, truth = instance_generator.prepare_runtime_instance(task_cfg,native_workspace)
        instance = public_instance
        truth["scoring_reference"] = json.loads(private_reference.read_text(encoding="utf-8"))
        rule = scoring.score_rules(
            native_workspace,
            native_workspace,
            result,
            instance_truth=truth,
            task_cfg=task_cfg,
        )
    score = float(rule["score"])
    return {
        "score": score,
        "subscores": {"deterministic": score, "independent_scientific": 0.0},
        "failure_codes": [] if score >= 60 else ["DETERMINISTIC_EVALUATION_BELOW_PASS"],
        "evidence": {
            "evaluation_mode": "postrun_private_deterministic_replay",
            "agent_private_mount": False,
            "rule_evaluation": rule,
        },
    }
