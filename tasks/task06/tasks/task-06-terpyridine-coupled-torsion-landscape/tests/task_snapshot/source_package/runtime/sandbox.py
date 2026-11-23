"""Render the bubblewrap command used for one model process."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


_INNER_ENVIRONMENT = (
    ("HOME", "/model/workspace"),
    ("TMPDIR", "/model/workspace"),
    ("LANG", "C.UTF-8"),
    ("LC_ALL", "C.UTF-8"),
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
)


def build_bubblewrap_command(
    public_root: Path,
    workspace_root: Path,
    submission_root: Path,
    command: Sequence[str],
    *,
    runtime_paths: Sequence[Path] = (),
    masked_paths: Sequence[Path] = (),
    bubblewrap_executable: Path = Path("/usr/bin/bwrap"),
) -> list[str]:
    rendered = [
        str(bubblewrap_executable),
        "--unshare-net",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
    ]
    for name, value in _INNER_ENVIRONMENT:
        rendered.extend(("--setenv", name, value))
    rendered.extend(
        [
        "--ro-bind",
        str(Path(public_root).resolve()),
        "/model/public",
        "--bind",
        str(Path(workspace_root).resolve()),
        "/model/workspace",
        "--bind",
        str(Path(submission_root).resolve()),
        "/model/submission",
        ]
    )
    runtime_mounts: list[tuple[Path, Path]] = []
    for runtime_path in runtime_paths:
        requested_target = Path(runtime_path)
        if not requested_target.is_absolute():
            raise ValueError("runtime mount targets must be absolute")
        resolved_source = requested_target.resolve()
        runtime_mounts.append((resolved_source, requested_target))
        rendered.extend(
            ("--ro-bind", str(resolved_source), str(requested_target))
        )
    for masked_path in masked_paths:
        resolved_mask = Path(masked_path).resolve()
        for resolved_source, requested_target in runtime_mounts:
            if resolved_mask == resolved_source:
                mask_target = requested_target
            elif resolved_source in resolved_mask.parents:
                mask_target = requested_target / resolved_mask.relative_to(resolved_source)
            else:
                continue
            rendered.extend(("--ro-bind", "/dev/null", str(mask_target)))
            break
    rendered.extend(("--chdir", "/model/workspace"))
    rendered.append("--")
    rendered.extend(command)
    return rendered
