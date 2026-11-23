from __future__ import annotations

import inspect
from pathlib import Path
import subprocess

import pytest

import framework.sandbox as sandbox_module
from framework.sandbox import (
    AllowlistedCommandRunner,
    PathSandbox,
    SHELL_ENABLED,
    SandboxViolation,
)


@pytest.fixture
def sandbox_root(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "artifacts").mkdir()
    (root / "artifacts" / "result.json").write_text("{}\n", encoding="utf-8")
    return root


@pytest.mark.parametrize(
    "unsafe",
    [
        r"C:\Windows\System32\config",
        r"C:drive-relative.txt",
        r"\\server\share\secret",
        r"\rooted\secret",
        "/etc/passwd",
        "../outside.txt",
        "artifacts/../../outside.txt",
        "artifact.json:secret-stream",
        r"\\?\C:\Windows\System32",
        r"\\.\PhysicalDrive0",
        "NUL",
        "safe/CON.txt",
        "trailing./file",
        "trailing /file",
        "artifacts//result.json",
        "artifacts/./result.json",
    ],
)
def test_rejects_absolute_drive_unc_parent_ads_and_ambiguous_paths(
    sandbox_root: Path, unsafe: str
) -> None:
    sandbox = PathSandbox(sandbox_root)
    with pytest.raises(SandboxViolation):
        sandbox.resolve(unsafe)


def test_resolves_only_regular_relative_paths(sandbox_root: Path) -> None:
    sandbox = PathSandbox(sandbox_root)
    expected = (sandbox_root / "artifacts" / "result.json").resolve()
    assert sandbox.resolve_for_read("artifacts/result.json") == expected
    assert sandbox.resolve_for_write("artifacts/new.json") == (
        sandbox_root / "artifacts" / "new.json"
    ).resolve()
    with pytest.raises(SandboxViolation, match="does not exist"):
        sandbox.resolve_for_read("artifacts/missing.json")
    with pytest.raises(SandboxViolation, match="parent must already exist"):
        sandbox.resolve_for_write("missing/new.json")


def test_forbidden_roots_are_unreachable_and_not_listable(sandbox_root: Path) -> None:
    oracle = sandbox_root / "private" / "oracle"
    oracle.mkdir(parents=True)
    (oracle / "profile.json").write_text("secret\n", encoding="utf-8")
    env_file = sandbox_root / ".env"
    env_file.write_text("DO_NOT_READ\n", encoding="utf-8")
    sandbox = PathSandbox(sandbox_root, forbidden_roots=(oracle, env_file))
    with pytest.raises(SandboxViolation, match="forbidden"):
        sandbox.resolve_for_read("private/oracle/profile.json")
    with pytest.raises(SandboxViolation, match="forbidden"):
        sandbox.resolve("private", must_exist=True)
    with pytest.raises(SandboxViolation, match="forbidden"):
        sandbox.resolve_for_read(".env")


def test_symlink_escape_is_rejected_when_creation_is_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret\n", encoding="utf-8")
    link = root / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        link.mkdir()
        real_check = sandbox_module._is_reparse_or_symlink
        monkeypatch.setattr(
            sandbox_module,
            "_is_reparse_or_symlink",
            lambda path: path.name == "escape" or real_check(path),
        )
    sandbox = PathSandbox(root)
    with pytest.raises(SandboxViolation, match="reparse|symlink|junction"):
        sandbox.resolve_for_read("escape/secret.txt")


def test_reparse_escape_detection_is_fail_closed_when_simulated(
    sandbox_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trap = sandbox_root / "artifacts" / "trap"
    trap.mkdir()
    sandbox = PathSandbox(sandbox_root)
    real_check = sandbox_module._is_reparse_or_symlink

    def simulated(path: Path) -> bool:
        return path.name == "trap" or real_check(path)

    monkeypatch.setattr(sandbox_module, "_is_reparse_or_symlink", simulated)
    with pytest.raises(SandboxViolation, match="reparse"):
        sandbox.resolve("artifacts/trap/file.json")


def test_sandbox_root_itself_cannot_be_a_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "root-link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        link.mkdir()
        real_check = sandbox_module._is_reparse_or_symlink
        monkeypatch.setattr(
            sandbox_module,
            "_is_reparse_or_symlink",
            lambda path: path.name == "root-link" or real_check(path),
        )
    with pytest.raises(SandboxViolation, match="root.*symlink|root.*reparse"):
        PathSandbox(link)


def test_allowlisted_runner_always_uses_argv_and_shell_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "xtb.exe"
    executable.write_bytes(b"placeholder")
    workdir = tmp_path / "attempt"
    workdir.mkdir()
    runner = AllowlistedCommandRunner({"xtb": executable})
    captured: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["argv"] = argv
        captured.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "xtb version 6.7.1\n", "")

    monkeypatch.setattr(sandbox_module.subprocess, "run", fake_run)
    result = runner.run(
        "xtb",
        ["input.xyz", "--version", "; whoami"],
        cwd=workdir,
        env={"OMP_NUM_THREADS": "1"},
        timeout_seconds=10,
    )
    assert result.returncode == 0
    assert captured["argv"] == [
        str(executable.resolve()),
        "input.xyz",
        "--version",
        "; whoami",
    ]
    assert captured["shell"] is False
    assert captured["check"] is False
    assert SHELL_ENABLED is False
    assert "shell" not in inspect.signature(runner.run).parameters


def test_allowlisted_runner_rejects_override_and_command_strings(tmp_path: Path) -> None:
    executable = tmp_path / "xtb.exe"
    executable.write_bytes(b"placeholder")
    runner = AllowlistedCommandRunner({"xtb": executable})
    with pytest.raises(SandboxViolation, match="not allowlisted"):
        runner.argv("powershell", ["-Command", "whoami"])
    with pytest.raises(SandboxViolation, match="argv sequence"):
        runner.argv("xtb", "--version")


def test_relative_or_reparse_allowlist_entries_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(SandboxViolation, match="absolute"):
        AllowlistedCommandRunner({"xtb": Path("xtb.exe")})
    with pytest.raises(SandboxViolation, match="does not exist"):
        AllowlistedCommandRunner({"xtb": tmp_path / "missing.exe"})
