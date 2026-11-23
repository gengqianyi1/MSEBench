"""Upload organizer evidence only after Harbor enters its separate verifier container."""

import json
from pathlib import Path

from harbor.verifier.verifier import Verifier


class WorkspaceSidecarVerifier(Verifier):
    def __init__(self, *args, evidence_root, attempt_id, task_id, **kwargs):
        super().__init__(*args, **kwargs)
        self.evidence_root = Path(evidence_root).resolve(strict=True)
        self.attempt_id = attempt_id
        self.task_id = task_id

    async def verify(self):
        artifact = json.loads((self.evidence_root / "current.json").read_text())
        if artifact["public_case"]["task"]["id"] != self.task_id:
            raise RuntimeError("ORGANIZER_TASK_MISMATCH")
        if artifact["organizer_metadata"]["attempt_id"] != self.attempt_id:
            raise RuntimeError("ORGANIZER_ATTEMPT_MISMATCH")
        for path in self.evidence_root.rglob("*"):
            if path.is_symlink():
                raise RuntimeError("ORGANIZER_EVIDENCE_SYMLINK")
        await self.environment.upload_dir(
            source_dir=self.evidence_root, target_dir="/organizer_evidence"
        )
        return await super().verify()
