from __future__ import annotations

import hashlib
from pathlib import Path

from tools.tool_registry import _project_native_log


def test_native_log_projection_redacts_host_path_and_preserves_science(
    tmp_path: Path,
) -> None:
    executable = (
        r"C:\Users\admin\.conda\envs\xtb-benchmark"
        r"\Library\bin\xtb.exe"
    )
    source = tmp_path / "raw.log"
    destination = tmp_path / "safe.log"
    raw = (
        f"program call : {executable} input.xyz --gfn 2\n"
        "GEOMETRY OPTIMIZATION CONVERGED\n"
        "| TOTAL ENERGY -34.989457373293 Eh |\n"
    ).encode("utf-8")
    source.write_bytes(raw)
    raw_sha = hashlib.sha256(raw).hexdigest().upper()

    _project_native_log(source, destination, executable)

    projected = destination.read_text(encoding="utf-8")
    assert executable not in projected
    assert "<XTB_EXECUTABLE>" in projected
    assert "GEOMETRY OPTIMIZATION CONVERGED" in projected
    assert "| TOTAL ENERGY -34.989457373293 Eh |" in projected
    assert source.read_bytes() == raw
    assert hashlib.sha256(source.read_bytes()).hexdigest().upper() == raw_sha
    assert hashlib.sha256(destination.read_bytes()).hexdigest().upper() != raw_sha
