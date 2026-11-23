"""The two-action public runtime for protocol 2.0.

Raw host provenance is written only below ``audit_tmp``.  Releasable run
artifacts contain a closed, path-free scientific projection.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time
from typing import Any, Callable, Mapping

from tools.auto_profile import compact_profile_analysis, select_profile
from tools.sampling_state import (
    MAX_ATTEMPTS_PER_ANGLE,
    MAX_BATCHES,
    MAX_UNIQUE_ANGLES,
    MIN_VALID_UNIQUE_ANGLES,
    SamplingStateError,
    expected_batch_size,
    reconstruct_batch_state,
    validate_batch_angles,
)
from tools.science import (
    FINAL_ANGLE_TOLERANCE_DEG,
    HARTREE_TO_KCAL_MOL,
    XTB_BINARY_SHA256,
    XTB_VERSION,
    canonical_json_bytes,
    circular_distance_deg,
    run_native_scan,
    sha256_file,
    verify_case_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "benchmark_case"
RAW_PROVENANCE_ROOT = ROOT / "audit_tmp" / "local_host_provenance" / "runs"
CONFIDENCE_VALUES = {"low", "medium", "high"}
PUBLIC_ACTIONS = ("run_scan_batch", "inspect_failed_attempt")
STRUCTURE_VALIDATION_KEYS = (
    "atom_count_ok",
    "atom_order_ok",
    "composition_ok",
    "formal_charge_ok",
    "connectivity_ok",
)
NATIVE_LOG_PROJECTION = (
    "native-log-path-redaction/1.0"
)
_WINDOWS_ABSOLUTE_TEXT = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s'\"<>|?*]+"
)
_UNC_ABSOLUTE_TEXT = re.compile(r"(?<![\\/])(?:\\\\|//)[^\s'\"<>|?*]+")
_POSIX_HOST_ABSOLUTE_TEXT = re.compile(
    r"(?<![:A-Za-z0-9_])/(?:home|Users|tmp|workspace|mnt)/"
    r"[^\s'\"<>|?*]+"
)


class ToolRuntimeError(RuntimeError):
    """A closed public action violates runtime policy."""


def _exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value))


def _project_native_log(source: Path, destination: Path, executable: str) -> None:
    """Release a deterministic path-redacted projection of a native log."""

    text = source.read_bytes().decode("utf-8", errors="replace")
    variants = {
        executable,
        executable.replace("\\", "/"),
        executable.replace("/", "\\"),
    }
    for value in sorted(
        (item for item in variants if item),
        key=len,
        reverse=True,
    ):
        text = text.replace(value, "<XTB_EXECUTABLE>")
    text = _WINDOWS_ABSOLUTE_TEXT.sub(
        "<REDACTED_ABSOLUTE_PATH>", text
    )
    text = _UNC_ABSOLUTE_TEXT.sub("<REDACTED_ABSOLUTE_PATH>", text)
    text = _POSIX_HOST_ABSOLUTE_TEXT.sub(
        "<REDACTED_ABSOLUTE_PATH>", text
    )
    destination.write_bytes(text.encode("utf-8"))


def _closed_object(
    value: Any, *, allowed: set[str], required: set[str]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ToolRuntimeError("action must be a JSON object")
    if set(value) != required or set(value) - allowed:
        raise ToolRuntimeError("action does not match the closed schema")
    return dict(value)


def load_tool_contracts() -> list[dict[str, Any]]:
    payload = json.loads(
        (ROOT / "configs" / "tool_contracts.json").read_text(encoding="utf-8")
    )
    actions = payload.get("actions")
    if not isinstance(actions, list):
        raise ToolRuntimeError("public action registry is invalid")
    names = [item.get("name") for item in actions]
    if names != list(PUBLIC_ACTIONS):
        raise ToolRuntimeError("public action allowlist or order changed")
    return actions


def _safe_run_id(value: str) -> str:
    if (
        not value
        or len(value) > 96
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in value)
    ):
        raise ToolRuntimeError("run id is not a safe relative identifier")
    return value


def _hash_action(action: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(action)).hexdigest().upper()


def _finite_number(value: Any) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


class ToolSession:
    """Stateful native batch runtime with an independently reconstructable ledger."""

    def __init__(
        self,
        run_root: Path,
        *,
        run_id: str | None = None,
        raw_provenance_root: Path | None = None,
        native_runner: Callable[..., dict[str, Any]] = run_native_scan,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not verify_case_manifest(CASE_ROOT):
            raise ToolRuntimeError("frozen public case manifest is invalid")
        if not run_root.is_dir():
            raise ToolRuntimeError("run root must exist before ToolSession construction")
        resolved_root = run_root.resolve(strict=True)
        project_root = ROOT.resolve(strict=True)
        if resolved_root != project_root and project_root not in resolved_root.parents:
            raise ToolRuntimeError("run root must remain inside the v2 project")
        self.run_root = resolved_root
        self.run_id = _safe_run_id(run_id or run_root.name)
        self.artifacts_root = self.run_root / "artifacts"
        self.batch_root = self.artifacts_root / "batches"
        self.summary_root = self.artifacts_root / "scan_summaries"
        self.profile_root = self.artifacts_root / "profiles"
        for directory in (
            self.artifacts_root,
            self.batch_root,
            self.summary_root,
            self.profile_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        base = (
            raw_provenance_root.resolve()
            if raw_provenance_root is not None
            else RAW_PROVENANCE_ROOT.resolve()
        )
        project = ROOT.resolve()
        if base != project and project not in base.parents:
            raise ToolRuntimeError("raw provenance root must remain inside the project")
        self.raw_attempt_root = base / self.run_id / "attempts"
        if self.raw_attempt_root.exists() or self.raw_attempt_root.is_symlink():
            raise ToolRuntimeError("raw attempt root already exists")
        self.raw_attempt_root.mkdir(parents=True, exist_ok=False)
        self._native_runner = native_runner
        self._monotonic = monotonic
        self._started = monotonic()
        self._ledger: list[dict[str, Any]] = []
        self._attempts: dict[int, list[dict[str, Any]]] = {}
        self._valid_energy_hartree: dict[int, float] = {}
        self._profiles: dict[str, dict[str, Any]] = {}
        self._public_action_count = 0
        self.terminal_action: dict[str, Any] | None = None
        self.terminal_action_hash: str | None = None
        self.terminal_failure: str | None = None
        self.last_tool_result_chars = 0

    @property
    def ledger(self) -> list[dict[str, Any]]:
        return deepcopy(self._ledger)

    @property
    def attempt_summaries(self) -> dict[int, list[dict[str, Any]]]:
        return deepcopy(self._attempts)

    @property
    def profiles_root(self) -> Path:
        """Compatibility alias used by independent evaluators."""

        return self.profile_root

    @property
    def final_submission(self) -> dict[str, Any] | None:
        if self.terminal_action is None:
            return None
        return {
            "case_id": "2_fluorobiphenyl_central_torsion_v1",
            "final_status": "completed",
            "confidence": self.terminal_action["confidence"],
            "rationale": self.terminal_action["rationale"],
            "finalize_action_hash": self.terminal_action_hash,
        }

    def sampling_state(self) -> dict[str, Any]:
        return reconstruct_batch_state(self._ledger)

    def budget_snapshot(self) -> dict[str, Any]:
        state = self.sampling_state()
        return {
            "scan_batches_used": state["batch_count"],
            "scan_batches_remaining": max(0, MAX_BATCHES - state["batch_count"]),
            "unique_angles_used": state["unique_angle_count"],
            "unique_angles_remaining": state["unique_angles_remaining"],
            "valid_unique_angles": state["valid_unique_angle_count"],
            "failed_unique_angles": len(state["failed_angles"]),
            "feedback_rounds": state["feedback_rounds"],
            "profile_fits": state["profile_fit_rounds"],
            "public_actions_used": self._public_action_count,
            "native_attempts": sum(len(items) for items in self._attempts.values()),
            "terminal_failure": self.terminal_failure,
            "elapsed_seconds": round(self._monotonic() - self._started, 6),
        }

    def dispatch(
        self,
        action_or_name: Mapping[str, Any] | str,
        arguments: Mapping[str, Any] | None = None,
        *,
        assistant_turn: int,
    ) -> dict[str, Any]:
        """Dispatch one closed action.

        The two-argument form is retained only for runner/test ergonomics; the
        closed ``action`` discriminator must still be present in ``arguments``.
        """

        self._public_action_count += 1
        if not isinstance(assistant_turn, int) or isinstance(assistant_turn, bool):
            result = {"ok": False, "status": "INVALID_ASSISTANT_TURN"}
        else:
            payload: Any
            if isinstance(action_or_name, Mapping):
                payload = dict(action_or_name)
                name = payload.get("action")
            else:
                name = action_or_name
                payload = dict(arguments or {})
            try:
                if self.terminal_action is not None:
                    raise ToolRuntimeError("TERMINAL_FINALIZATION_ALREADY_ACCEPTED")
                if self.terminal_failure is not None:
                    raise ToolRuntimeError("TERMINAL_FAILURE_ALREADY_RECORDED")
                if name == "run_scan_batch":
                    result = self.run_scan_batch(payload, assistant_turn=assistant_turn)
                elif name == "inspect_failed_attempt":
                    result = self.inspect_failed_attempt(
                        payload, assistant_turn=assistant_turn
                    )
                else:
                    result = {"ok": False, "status": "ACTION_NOT_ALLOWLISTED"}
            except (ToolRuntimeError, SamplingStateError, ValueError) as exc:
                result = {
                    "ok": False,
                    "status": (
                        str(exc)
                        if str(exc).isupper() and " " not in str(exc)
                        else type(exc).__name__.upper()
                    ),
                    "error": str(exc)[:240],
                }
        result = {**result, "budget": self.budget_snapshot()}
        serialized = canonical_json_bytes(result)
        self.last_tool_result_chars = len(serialized.decode("utf-8"))
        if self.last_tool_result_chars > 6000:
            raise ToolRuntimeError("compact tool result exceeds 6000 characters")
        return result

    def _validate_batch_action(self, payload: Any) -> dict[str, Any]:
        action = _closed_object(
            payload,
            allowed={"action", "angles_deg", "finalize", "confidence", "rationale"},
            required={"action", "angles_deg", "finalize", "confidence", "rationale"},
        )
        if action["action"] != "run_scan_batch":
            raise ToolRuntimeError("ACTION_DISCRIMINATOR_MISMATCH")
        action["angles_deg"] = validate_batch_angles(action["angles_deg"])
        if not isinstance(action["finalize"], bool):
            raise ToolRuntimeError("finalize must be boolean")
        confidence = action["confidence"]
        if action["finalize"]:
            if confidence not in CONFIDENCE_VALUES:
                raise ToolRuntimeError("terminal confidence must be low, medium, or high")
        elif confidence is not None:
            raise ToolRuntimeError("confidence must be null before terminal phase")
        rationale = action["rationale"]
        if not isinstance(rationale, str) or not 1 <= len(rationale) <= 400:
            raise ToolRuntimeError("rationale must contain 1 through 400 characters")
        return action

    def _validate_phase(self, action: Mapping[str, Any]) -> str:
        state = self.sampling_state()
        index = state["batch_count"]
        if index >= MAX_BATCHES:
            raise ToolRuntimeError("SCAN_BATCH_BUDGET_EXCEEDED")
        phase = ("phase0", "phase1", "phase2", "recovery")[index]
        low, high = expected_batch_size(index)
        count = len(action["angles_deg"])
        if not low <= count <= high:
            raise ToolRuntimeError(f"{phase.upper()}_BATCH_SIZE_INVALID")
        duplicates = set(action["angles_deg"]) & set(state["unique_angles"])
        if duplicates:
            raise ToolRuntimeError("PRIOR_ANGLE_DUPLICATE")
        if state["unique_angle_count"] + count > MAX_UNIQUE_ANGLES:
            raise ToolRuntimeError("UNIQUE_ANGLE_BUDGET_EXCEEDED")
        if index < 2 and action["finalize"]:
            raise ToolRuntimeError("EARLY_FINALIZE_FORBIDDEN")
        if index == 2 and not action["finalize"]:
            raise ToolRuntimeError("PHASE2_REQUIRES_TERMINAL_FINALIZE")
        if index == 3:
            if not state["recovery_allowed"]:
                raise ToolRuntimeError("RECOVERY_NOT_ALLOWED")
            if not action["finalize"]:
                raise ToolRuntimeError("RECOVERY_REQUIRES_TERMINAL_FINALIZE")
        return phase

    def _project_attempt(
        self,
        *,
        batch_number: int,
        angle: int,
        attempt_number: int,
        raw_root: Path,
        record: Mapping[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        attempt_id = (
            f"batch_{batch_number:02d}_angle_{angle:03d}"
            f"_attempt_{attempt_number:02d}"
        )
        safe_root = self.summary_root / attempt_id
        safe_root.mkdir(parents=True, exist_ok=False)
        copied: list[dict[str, Any]] = []
        xtb = record.get("xtb")
        if not isinstance(xtb, Mapping):
            xtb = {}
        executable = (
            str(xtb.get("executable"))
            if isinstance(xtb.get("executable"), str)
            else ""
        )
        # Persist the minimum native evidence needed for an evaluator that does
        # not trust this adapter's parsed booleans.  The logs are xTB process
        # artifacts (never provider responses) and are required to re-parse
        # convergence and the final TOTAL ENERGY independently.
        for name in (
            "input.xyz",
            "xcontrol.inp",
            "stdout.log",
            "stderr.log",
            "xtbopt.xyz",
        ):
            source = raw_root / name
            if not source.is_file():
                continue
            destination = safe_root / name
            if name in {"stdout.log", "stderr.log"}:
                _project_native_log(source, destination, executable)
            else:
                shutil.copyfile(source, destination)
            copied.append(
                {
                    "path": f"artifacts/scan_summaries/{attempt_id}/{name}",
                    "size_bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                }
            )
        energy = _finite_number(record.get("total_energy_hartree"))
        final_angle = _finite_number(record.get("final_angle_deg"))
        angle_error = (
            circular_distance_deg(final_angle, angle)
            if final_angle is not None
            else None
        )
        validation = record.get("structure_validation")
        if not isinstance(validation, Mapping):
            validation = {}
        stdout = raw_root / "stdout.log"
        stderr = raw_root / "stderr.log"
        raw_manifest = raw_root / "attempt_manifest.json"
        expected_environment = {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        expected_argv_tail = [
            "input.xyz",
            "--gfn",
            "2",
            "--chrg",
            "0",
            "--uhf",
            "0",
            "--opt",
            "tight",
            "--input",
            "xcontrol.inp",
        ]
        raw_argv = record.get("argv")
        executable = xtb.get("executable")
        invocation_valid = bool(
            isinstance(raw_argv, list)
            and len(raw_argv) == len(expected_argv_tail) + 1
            and isinstance(raw_argv[0], str)
            and raw_argv[0]
            and raw_argv[0] == executable
            and raw_argv[1:] == expected_argv_tail
            and record.get("shell") is False
            and record.get("environment") == expected_environment
            and xtb.get("version") == XTB_VERSION
            and xtb.get("binary_sha256") == XTB_BINARY_SHA256
            and xtb.get("version_exit_code") == 0
        )
        protocol_valid = bool(
            record.get("method") == "GFN2-xTB"
            and record.get("phase") == "gas"
            and record.get("charge") == 0
            and record.get("uhf") == 0
            and record.get("threads") == 1
            and record.get("requested_final_angle_tolerance_deg")
            == FINAL_ANGLE_TOLERANCE_DEG
        )
        expected_reference_sha = sha256_file(CASE_ROOT / "reference.xyz")
        prepared = record.get("prepared_input")
        if not isinstance(prepared, Mapping):
            prepared = {}
        safe_hashes = {
            item["path"].rsplit("/", 1)[-1]: item["sha256"]
            for item in copied
        }
        expected_signed = float(angle if angle <= 180 else angle - 360)
        expected_constraint = (
            "$constrain\n"
            "  force constant=1.0\n"
            f"  dihedral: 6, 7, 8, 9, {expected_signed:.1f}\n"
            "$end\n"
        ).encode("ascii")
        provenance_valid = bool(
            record.get("reference_sha256") == expected_reference_sha
            and prepared.get("reference_sha256") == expected_reference_sha
            and prepared.get("input_sha256") == safe_hashes.get("input.xyz")
            and (raw_root / "xcontrol.inp").is_file()
            and (raw_root / "xcontrol.inp").read_bytes()
            == expected_constraint
            and stdout.is_file()
            and stderr.is_file()
            and raw_manifest.is_file()
        )
        summary = {
            "schema_version": "2.0",
            "attempt_id": attempt_id,
            "requested_angle_deg": angle,
            "attempt_number": attempt_number,
            "return_code": record.get("return_code"),
            "timed_out": record.get("timed_out") is True,
            "converged": record.get("converged") is True,
            "energy_parsed": energy is not None,
            "total_energy_hartree": energy,
            "final_angle_deg": final_angle,
            "angle_error_deg": angle_error,
            "structure_validation": {
                key: validation.get(key) is True
                for key in STRUCTURE_VALIDATION_KEYS
            },
            # Never trust the native adapter's aggregate boolean by itself.
            # Recompute the complete public eligibility predicate from the
            # projected primitive facts, including the circular angle error.
            "profile_eligible": False,
            "scientific_protocol": {
                "method": "GFN2-xTB" if protocol_valid else None,
                "phase": "gas" if protocol_valid else None,
                "charge": 0 if protocol_valid else None,
                "uhf": 0 if protocol_valid else None,
                "threads": 1 if protocol_valid else None,
                "requested_final_angle_tolerance_deg": (
                    FINAL_ANGLE_TOLERANCE_DEG if protocol_valid else None
                ),
            },
            "native_invocation": {
                "argv": (
                    ["<XTB_EXECUTABLE>", *expected_argv_tail]
                    if invocation_valid
                    else []
                ),
                "shell": False if invocation_valid else None,
                "environment": (
                    expected_environment if invocation_valid else {}
                ),
                "xtb_version": XTB_VERSION if invocation_valid else None,
                "xtb_binary_sha256": (
                    XTB_BINARY_SHA256 if invocation_valid else None
                ),
            },
            "provenance": {
                "reference_sha256": record.get("reference_sha256"),
                "input_sha256": (
                    record.get("prepared_input", {}).get("input_sha256")
                    if isinstance(record.get("prepared_input"), Mapping)
                    else None
                ),
                "constraint_sha256": (
                    sha256_file(raw_root / "xcontrol.inp")
                    if (raw_root / "xcontrol.inp").is_file()
                    else None
                ),
                "stdout_sha256": sha256_file(stdout) if stdout.is_file() else None,
                "stderr_sha256": sha256_file(stderr) if stderr.is_file() else None,
                "stdout_projection_sha256": safe_hashes.get("stdout.log"),
                "stderr_projection_sha256": safe_hashes.get("stderr.log"),
                "native_log_projection": NATIVE_LOG_PROJECTION,
                "raw_attempt_manifest_sha256": (
                    sha256_file(raw_manifest) if raw_manifest.is_file() else None
                ),
            },
            "safe_files": copied,
            "raw_host_paths_released": False,
        }
        summary["profile_eligible"] = bool(
            isinstance(summary["return_code"], int)
            and not isinstance(summary["return_code"], bool)
            and summary["return_code"] == 0
            and summary["timed_out"] is False
            and summary["converged"] is True
            and summary["energy_parsed"] is True
            and protocol_valid
            and invocation_valid
            and provenance_valid
            and {
                "input.xyz",
                "xcontrol.inp",
                "stdout.log",
                "stderr.log",
                "xtbopt.xyz",
            }
            .issubset(safe_hashes)
            and summary["final_angle_deg"] is not None
            and summary["angle_error_deg"] is not None
            and 0.0
            <= float(summary["angle_error_deg"])
            <= FINAL_ANGLE_TOLERANCE_DEG
            and all(
                summary["structure_validation"].get(key) is True
                for key in STRUCTURE_VALIDATION_KEYS
            )
        )
        _exclusive_json(safe_root / "summary.json", summary)
        return summary, bool(
            protocol_valid and invocation_valid and provenance_valid
        )

    def _run_angle(self, *, batch_number: int, angle: int) -> dict[str, Any]:
        attempts: list[dict[str, Any]] = []
        remaining_total = 900.0 - (self._monotonic() - self._started)
        for attempt_number in range(1, MAX_ATTEMPTS_PER_ANGLE + 1):
            if remaining_total <= 0:
                raise ToolRuntimeError("TOTAL_NATIVE_TIMEOUT_EXCEEDED")
            raw_root = (
                self.raw_attempt_root
                / f"batch_{batch_number:02d}_angle_{angle:03d}_attempt_{attempt_number:02d}"
            )
            record = self._native_runner(
                reference_path=CASE_ROOT / "reference.xyz",
                angle_deg=angle,
                attempt_root=raw_root,
                timeout_seconds=max(1, min(180, int(remaining_total))),
            )
            summary, invariant_valid = self._project_attempt(
                batch_number=batch_number,
                angle=angle,
                attempt_number=attempt_number,
                raw_root=raw_root,
                record=record,
            )
            attempts.append(summary)
            # Commit each immutable attempt to the independently reconstructable
            # budget immediately.  A total-timeout before a later retry must not
            # make already executed native work disappear.
            self._attempts[angle] = list(attempts)
            if not invariant_valid:
                self.terminal_failure = "NATIVE_INVARIANT_BREACH"
                raise ToolRuntimeError("NATIVE_INVARIANT_BREACH")
            remaining_total = 900.0 - (self._monotonic() - self._started)
            if summary["profile_eligible"]:
                self._valid_energy_hartree[angle] = float(
                    summary["total_energy_hartree"]
                )
                break
        self._attempts[angle] = attempts
        return {
            "angle_deg": angle,
            "status": "valid" if angle in self._valid_energy_hartree else "failed",
            "attempt_count": len(attempts),
            "final_angle_deg": attempts[-1]["final_angle_deg"],
            "angle_error_deg": attempts[-1]["angle_error_deg"],
        }

    def _fit_current_profile(self, batch_number: int) -> dict[str, Any]:
        minimum = min(self._valid_energy_hartree.values())
        samples = {
            angle: (energy - minimum) * HARTREE_TO_KCAL_MOL
            for angle, energy in self._valid_energy_hartree.items()
        }
        profile = select_profile(samples)
        profile_id = f"profile_{batch_number:02d}_{profile['profile_hash'][:16].lower()}"
        profile = {
            **profile,
            "schema_version": "2.0",
            "profile_id": profile_id,
            "source_attempt_ids": [
                self._attempts[angle][-1]["attempt_id"] for angle in sorted(samples)
            ],
            "oracle_used": False,
        }
        _exclusive_json(self.profile_root / f"{profile_id}.json", profile)
        self._profiles[profile_id] = profile
        return profile

    def run_scan_batch(
        self, payload: Any, *, assistant_turn: int
    ) -> dict[str, Any]:
        del assistant_turn
        action = self._validate_batch_action(payload)
        phase = self._validate_phase(action)
        batch_number = len(self._ledger) + 1
        point_results = [
            self._run_angle(batch_number=batch_number, angle=angle)
            for angle in action["angles_deg"]
        ]
        valid = [
            item["angle_deg"] for item in point_results if item["status"] == "valid"
        ]
        failed = [
            item["angle_deg"] for item in point_results if item["status"] == "failed"
        ]
        if len(self._valid_energy_hartree) < 4:
            # With six phase-0 angles, fewer than four valid points also means
            # fewer than ten can ever be valid within the 12-angle budget.
            # Persist the consumed angles and both native attempts before
            # failing closed; never make failed native work disappear from the
            # independently reconstructable budget.
            provisional = {
                "schema_version": "2.0",
                "batch_number": batch_number,
                "phase": phase,
                "angles_deg": list(action["angles_deg"]),
                "valid_angles_deg": valid,
                "failed_angles_deg": failed,
                "finalize": bool(action["finalize"]),
                "confidence": action["confidence"],
                "rationale": action["rationale"],
                "action_hash": _hash_action(action),
                "profile_id": None,
                "profile_hash": None,
                "terminal_accepted": False,
            }
            self._ledger.append(provisional)
            self.terminal_failure = "NATIVE_ELIGIBILITY_IMPOSSIBLE"
            state = self.sampling_state()
            _exclusive_json(
                self.batch_root / f"batch_{batch_number:02d}.json",
                {
                    **provisional,
                    "point_results": point_results,
                    "reconstructed_state": {
                        key: value for key, value in state.items() if key != "ledger"
                    },
                    "fatal_reason": "NATIVE_ELIGIBILITY_IMPOSSIBLE",
                },
            )
            return {
                "ok": False,
                "status": "NATIVE_ELIGIBILITY_IMPOSSIBLE",
                "batch_number": batch_number,
                "phase": phase,
                "point_results": point_results,
                "profile": None,
                "state": {
                    "sampled_angles_deg": state["unique_angles"],
                    "valid_angles_deg": state["valid_angles"],
                    "failed_angles_deg": state["failed_angles"],
                    "unique_angles_used": state["unique_angle_count"],
                    "unique_angles_remaining": state["unique_angles_remaining"],
                    "valid_unique_angles": state["valid_unique_angle_count"],
                    "eligibility_gaps": state["eligibility_gaps"],
                    "eligibility_mathematically_possible": False,
                },
                "terminal": {
                    "requested": action["finalize"],
                    "accepted": False,
                    "provider_must_stop": True,
                },
                "batch_record": deepcopy(provisional),
            }
        profile = self._fit_current_profile(batch_number)
        provisional = {
            "schema_version": "2.0",
            "batch_number": batch_number,
            "phase": phase,
            "angles_deg": list(action["angles_deg"]),
            "valid_angles_deg": valid,
            "failed_angles_deg": failed,
            "finalize": bool(action["finalize"]),
            "confidence": action["confidence"],
            "rationale": action["rationale"],
            "action_hash": _hash_action(action),
            "profile_id": profile["profile_id"],
            "profile_hash": profile["profile_hash"],
            "terminal_accepted": False,
        }
        state_if_recorded = reconstruct_batch_state(self._ledger + [provisional])
        if phase == "phase2" and action["finalize"]:
            provisional["terminal_accepted"] = state_if_recorded["eligible"]
        elif phase == "recovery" and action["finalize"]:
            provisional["terminal_accepted"] = state_if_recorded["eligible"]
        self._ledger.append(provisional)
        state = self.sampling_state()
        _exclusive_json(self.batch_root / f"batch_{batch_number:02d}.json", {
            **provisional,
            "point_results": point_results,
            "reconstructed_state": {
                key: value for key, value in state.items() if key != "ledger"
            },
        })
        compact = compact_profile_analysis(profile)
        terminal_accepted = provisional["terminal_accepted"]
        if terminal_accepted:
            self.terminal_action = deepcopy(action)
            self.terminal_action_hash = provisional["action_hash"]
            status = "FINAL_ACCEPTED"
        elif phase == "phase2" and action["finalize"] and state["recovery_allowed"]:
            status = "RECOVERY_REQUIRED"
        elif phase in {"phase2", "recovery"} and action["finalize"]:
            status = "FINAL_ELIGIBILITY_FAILED"
            self.terminal_failure = status
        else:
            status = "BATCH_COMPLETE"
        model_state = {
            "phase_completed": phase,
            "next_phase": state["current_phase"],
            "sampled_angles_deg": state["unique_angles"],
            "valid_angles_deg": state["valid_angles"],
            "failed_angles_deg": state["failed_angles"],
            "unique_angles_used": state["unique_angle_count"],
            "unique_angles_remaining": state["unique_angles_remaining"],
            "valid_unique_angles": state["valid_unique_angle_count"],
            "feedback_rounds": state["feedback_rounds"],
            "profile_fit_rounds": state["profile_fit_rounds"],
            "recovery_allowed": state["recovery_allowed"],
            "finalize_allowed": state["current_phase"] in {"phase2", "recovery"},
            "eligible": state["eligible"],
            "eligibility_gaps": state["eligibility_gaps"],
        }
        return {
            "ok": status in {"BATCH_COMPLETE", "FINAL_ACCEPTED", "RECOVERY_REQUIRED"},
            "status": status,
            "batch_number": batch_number,
            "phase": phase,
            "point_results": point_results,
            "profile": {
                "profile_id": profile["profile_id"],
                "profile_hash": profile["profile_hash"],
                "grid_size": 72,
                "all_finite": all(
                    math.isfinite(float(value))
                    for value in profile["predicted_energies"]
                ),
                "analysis": compact,
            },
            "state": model_state,
            "terminal": {
                "requested": action["finalize"],
                "accepted": terminal_accepted,
                "provider_must_stop": terminal_accepted
                or status == "FINAL_ELIGIBILITY_FAILED",
            },
            # Included for independent event reconstruction; it is bounded and
            # contains no raw host path or full profile.
            "batch_record": deepcopy(provisional),
        }

    def inspect_failed_attempt(
        self, payload: Any, *, assistant_turn: int
    ) -> dict[str, Any]:
        del assistant_turn
        action = _closed_object(
            payload,
            allowed={"action", "angle_deg"},
            required={"action", "angle_deg"},
        )
        if action["action"] != "inspect_failed_attempt":
            raise ToolRuntimeError("ACTION_DISCRIMINATOR_MISMATCH")
        angles = validate_batch_angles([action["angle_deg"]])
        angle = angles[0]
        attempts = self._attempts.get(angle)
        if (
            attempts is None
            or len(attempts) != MAX_ATTEMPTS_PER_ANGLE
            or any(item["profile_eligible"] for item in attempts)
        ):
            raise ToolRuntimeError("FAILED_ATTEMPT_NOT_INSPECTABLE")
        return {
            "ok": True,
            "status": "FAILED_ATTEMPT_INSPECTION",
            "requested_angle_deg": angle,
            "attempt_count": len(attempts),
            "attempts": [
                {
                    "attempt_id": item["attempt_id"],
                    "return_code": item["return_code"],
                    "timed_out": item["timed_out"],
                    "converged": item["converged"],
                    "energy_parsed": item["energy_parsed"],
                    "angle_error_deg": item["angle_error_deg"],
                    "structure_validation": item["structure_validation"],
                }
                for item in attempts
            ],
            "replacement_sampling_allowed": self.sampling_state()[
                "recovery_allowed"
            ],
        }

    def profile(self, profile_id: str) -> dict[str, Any]:
        if profile_id not in self._profiles:
            raise ToolRuntimeError("unknown current-run profile id")
        return deepcopy(self._profiles[profile_id])


BatchToolSession = ToolSession


__all__ = [
    "BatchToolSession",
    "PUBLIC_ACTIONS",
    "ToolRuntimeError",
    "ToolSession",
    "load_tool_contracts",
]
