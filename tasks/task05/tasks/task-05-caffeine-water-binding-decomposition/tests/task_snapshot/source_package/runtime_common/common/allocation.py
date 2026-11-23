"""Evidence gates for genuinely sequential post-screen calculation allocation."""

from __future__ import annotations

import re
from typing import Any


_JOB_NUMBER = re.compile(r"(\d+)$")


def _sequence(job: Any) -> int:
    match = _JOB_NUMBER.search(str(getattr(job, "job_id", "")))
    return int(match.group(1)) if match else -1


def standalone_hessian_parent(hessian_job: Any, jobs: dict[str, Any]) -> str | None:
    """Return the earlier optimization whose output exactly feeds a Hessian.

    A direct ``--ohess`` on an unscreened seed is not enough.  A post-screen
    ``--ohess`` is accepted when its input hash exactly matches an earlier
    converged optimization; that evidence still demonstrates the required
    screen-then-allocate decision and yields a real complete Hessian.
    """
    if hessian_job is None:
        return None
    request = hessian_job.record.get("request") or {}
    arguments = [str(value).lower() for value in request.get("arguments", [])]
    if not ({"--hess", "--ohess"} & set(arguments)) or any(
        value == "--opt" or value.startswith("--opt=") for value in arguments
    ):
        return None
    inputs = hessian_job.record.get("inputs") or {}
    geometry_input = inputs.get("geometry") or {}
    input_hash = geometry_input.get("source_sha256") or hessian_job.record.get("input_geometry_sha256")
    if not isinstance(input_hash, str) or not input_hash:
        return None
    hessian_sequence = _sequence(hessian_job)
    parents = []
    for candidate in jobs.values():
        output_hash = (candidate.record.get("files_sha256") or {}).get("xtbopt.xyz")
        converged = bool((candidate.parsed.get("status") or {}).get("optimization_converged"))
        if output_hash == input_hash and converged and 0 <= _sequence(candidate) < hessian_sequence:
            parents.append(candidate)
    if not parents:
        return None
    return max(parents, key=_sequence).job_id
