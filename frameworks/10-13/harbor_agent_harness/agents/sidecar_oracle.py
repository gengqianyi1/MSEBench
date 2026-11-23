"""Organizer-only Oracle Agent for deployed sidecar lifecycle validation.

This Agent is never part of the participant bundle.  It exercises Harbor's Agent,
private sidecar, artifact transfer and separate-verifier phases without calling a
model API or recomputing the scientific Golden.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import urllib.request
from pathlib import Path, PurePosixPath

from harbor.agents.base import BaseAgent


class SidecarOracleAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "sidecar-oracle"

    def version(self) -> str:
        return "1.0.0"

    def __init__(self, *args, extra_env: dict[str, str] | None = None, **kwargs):
        super().__init__(*args, extra_env=extra_env, **kwargs)
        self._oracle_env = dict(extra_env or {})

    def _setting(self, name: str) -> str:
        value = self._oracle_env.get(name) or os.environ.get(name)
        if not value:
            raise RuntimeError(f"Missing required organizer setting: {name}")
        return value

    @staticmethod
    def _post_json(url: str, payload: dict) -> dict:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, dict) or not value.get("ok"):
            raise RuntimeError(f"Sidecar rejected Oracle lifecycle request: {value!r}")
        return value

    @staticmethod
    def _materialize_golden(
        golden_root: Path,
        evidence_root: Path,
        evidence_id: str,
        case_id: str,
    ) -> Path:
        parts = PurePosixPath(evidence_id).parts
        if len(parts) != 2 or any(part in {"", ".", ".."} for part in parts):
            raise RuntimeError("Invalid sidecar evidence id")
        root = evidence_root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        target = (root / Path(*parts)).resolve()
        if root not in target.parents:
            raise RuntimeError("Sidecar evidence path escapes organizer root")
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)

        if case_id == "task05-caffeine-water-binding-decomposition":
            source = golden_root / "submission"
            if not source.is_dir():
                raise RuntimeError(f"Task 05 Oracle submission not found: {source}")
            shutil.copytree(source, target / "submission")
        else:
            if not golden_root.is_dir():
                raise RuntimeError(f"Oracle run not found: {golden_root}")
            shutil.copytree(golden_root, target / "run")

        (target / "harbor_lifecycle_oracle.json").write_text(
            json.dumps(
                {
                    "schema_version": "chembench-harbor-lifecycle-oracle/1.0",
                    "case_id": case_id,
                    "evidence_id": evidence_id,
                    "source": "validated_private_golden_repetition_01",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return target

    @staticmethod
    async def _write_case_result(environment, value: dict) -> None:
        encoded = base64.b64encode(
            (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
        ).decode("ascii")
        command = (
            "python -c \"import base64,pathlib; "
            "pathlib.Path('/workspace/cases/case-1').mkdir(parents=True,exist_ok=True); "
            "open('/workspace/cases/case-1/case_result.json','wb').write("
            f"base64.b64decode('{encoded}'))\""
        )
        result = await environment.exec(command=command, cwd="/workspace", timeout_sec=30)
        if getattr(result, "return_code", 1) != 0:
            raise RuntimeError("Failed to write Harbor Oracle case result")

    async def setup(self, environment) -> None:
        del environment

    async def run(self, instruction: str, environment, context) -> None:
        del instruction
        sidecar_url = self._setting("CHEMBENCH_ORACLE_SIDECAR_URL").rstrip("/")
        evidence_root = Path(self._setting("CHEMBENCH_ORACLE_EVIDENCE_ROOT"))
        golden_root = Path(self._setting("CHEMBENCH_ORACLE_GOLDEN_ROOT"))

        started = self._post_json(sidecar_url + "/session", {})
        case = started.get("case")
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
            raise RuntimeError("Sidecar did not return a valid public case")
        submitted = self._post_json(
            sidecar_url + "/submit",
            {
                "session_id": started.get("session_id"),
                "candidate": {"status": "partial", "oracle_lifecycle": True},
            },
        )
        evidence_id = submitted.get("evidence_id")
        if not isinstance(evidence_id, str):
            raise RuntimeError("Sidecar submission did not return an evidence id")
        self._materialize_golden(
            golden_root,
            evidence_root,
            evidence_id,
            case["case_id"],
        )

        artifact = {
            "case_id": case["case_id"],
            "public_case": case,
            "candidate": {"status": "oracle_lifecycle"},
            "tool_events": [],
            "submission": submitted,
            "agent_metrics": {
                "agent_turns_used": 1,
                "scientific_turns_used": 1,
                "finalization_turns_used": 0,
                "tool_calls_used": 1,
                "termination_reason": "oracle_lifecycle_submitted",
            },
        }
        await self._write_case_result(environment, artifact)
        if hasattr(context, "metadata"):
            context.metadata = {
                "oracle_lifecycle": True,
                "case_id": case["case_id"],
                "evidence_id": evidence_id,
            }

