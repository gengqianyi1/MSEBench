"""Private stateful adapter for the four-case adaptive torsion campaign."""
from __future__ import annotations
import copy,json,os,shutil,sys,tempfile
from pathlib import Path
from typing import Any

HARTREE_TO_KCAL_MOL = 627.5094740631
GRID = list(range(0,360,5))

class AdapterError(RuntimeError):
    pass

class Adapter:
    def __init__(self, snapshot: Path):
        self.snapshot = Path(snapshot)
        self.runtime = self.snapshot / "private_data" / "runtime_package"
        sys.path.insert(0, str(self.runtime))
        from tools import science as tscience
        from framework.trajectory import TrajectoryRecorder
        from evaluator.profile_recompute_v2 import recompute_automatic_profile
        self.tscience = tscience
        self.TrajectoryRecorder = TrajectoryRecorder
        self.recompute_automatic_profile = recompute_automatic_profile
        self.suite = json.loads((self.runtime / "organizer_private" / "suite.json").read_text())
        self.cases = self.suite["cases"]
        self.case_count = len(self.cases)
        self.contracts = json.loads((self.runtime / "configs" / "tool_contracts.json").read_text())

    def start_session(self, evidence_dir, token):
        temp = tempfile.TemporaryDirectory(prefix="chembench-task09-")
        return {
            "temp": temp,
            "root": Path(temp.name),
            "case_states": {},
            "evidence_dir": Path(evidence_dir) if evidence_dir else None,
            "token": token,
        }

    def public_case(self, index):
        case = self.cases[index]
        cfg = self._case_config(index)
        actions = self.contracts.get("actions", [])
        return {
            "case_id": case["case_id"],
            "task": {
                "id": "adaptive_xtb_torsion_profile_agent_v3",
                "case_id": case["case_id"],
                "protocol": "adaptive_torsion_sampling_tool_loop/3.0",
            },
            "budget": {
                "max_scan_batches": 4,
                "max_unique_scan_angles": 12,
                "min_unique_valid_scan_angles": 10,
                "min_feedback_rounds": 2,
                "max_agent_turns": 5,
                "phase_0_angles": 6,
                "phase_refinement_min_angles": 2,
                "phase_refinement_max_angles": 3,
                "max_tool_calls": 5,
            },
            "constraints": {
                "angle_grid_deg": 5,
                "scientific_invariants": cfg.get("scientific_invariants", {}),
            },
            "output": {
                "candidate_schema": {
                    "action": "run_scan_batch",
                    "angles_deg": [0, 5, 10],
                    "finalize": True,
                    "confidence": "high",
                    "rationale": "string",
                }
            },
            "public_tools": actions,
        }

    def _case_config(self, index):
        case = self.cases[index]
        if index == 0:
            return json.loads((self.runtime / "benchmark_case" / "case_spec.json").read_text())
        cfg = self.runtime / str(case.get("config", ""))
        return json.loads(cfg.read_text()) if cfg.is_file() else {}

    def _state(self, index, state):
        key = index
        if key not in state["case_states"]:
            state["case_states"][key] = CaseState(self, index, state["token"], state["root"])
        return state["case_states"][key]

    def call_tool(self, index, name, arguments, state):
        if name not in {"run_scan_batch", "inspect_failed_attempt"}:
            return {"ok": False, "tool": name, "error": "ACTION_NOT_ALLOWLISTED"}
        case_state = self._state(index, state)
        result = case_state.call(name, arguments or {})
        return {
            "ok": result.get("ok", False),
            "tool": name,
            "call_id": case_state.last_call_id,
            "result": result,
        }

    def submit(self, index, candidate, tool_events, state):
        case_state = self._state(index, state)
        if not isinstance(candidate, dict) or candidate != case_state.terminal_action:
            return {"ok": False, "case_id": self.cases[index]["case_id"], "status": "invalid", "feedback": {"accepted": False}, "cases_remaining": self.case_count - index - 1}
        evidence_id = f'{state["token"]}/{self.cases[index]["case_id"]}'
        if state["evidence_dir"]:
            case_state.write_evidence(state["evidence_dir"] / evidence_id)
        return {
            "ok": True,
            "case_id": self.cases[index]["case_id"],
            "status": "submitted",
            "feedback": {"accepted": True, "final_status": "completed"},
            "cases_remaining": self.case_count - index - 1,
            "evidence_id": evidence_id,
        }


class CaseState:
    def __init__(self, adapter: Adapter, index: int, token: str, root: Path):
        self.adapter = adapter
        self.index = index
        self.case = adapter.cases[index]
        self.case_id = self.case["case_id"]
        self.token = token
        self.root = root
        self.run_id = f"harbor-{token}-{index}"
        self.actions: list[dict[str, Any]] = []
        self.selected_angles: list[int] = []
        self.valid_angles: list[int] = []
        self.failed_angles: list[int] = []
        self.point_results: list[dict[str, Any]] = []
        self.profiles: list[dict[str, Any]] = []
        self.attempt_dirs: dict[int, Path] = {}
        self.terminal_action: dict[str, Any] | None = None
        self.terminal_failure: str | None = None
        self.last_call_id = ""
        self.calls = 0
        self.recorder = adapter.TrajectoryRecorder(
            run_id=self.run_id,
            protocol="adaptive_torsion_sampling_tool_loop/3.0",
        )
        self.recorder.append("run_start", {
            "task_id": "adaptive_xtb_torsion_profile_agent_v3",
            "case_id": self.case_id,
            "protocol": "adaptive_torsion_sampling_tool_loop/3.0",
            "run_type": "HARBOR_PROTOCOL",
            "logical_model": "harbor_managed",
            "requested_model": "harbor_managed",
            "provider": "harbor",
            "transport": "harbor_agent",
        })

    def call(self, name, arguments):
        self.calls += 1
        self.last_call_id = f"harbor-{self.calls:03d}"
        if name == "run_scan_batch":
            return self._run_scan_batch(arguments)
        return self._inspect_failed_attempt(arguments)

    def _phase(self):
        return ("phase0", "phase1", "phase2", "recovery")[min(len(self.actions), 3)]

    def _validate_scan(self, action):
        if not isinstance(action, dict) or set(action) != {"action", "angles_deg", "finalize", "confidence", "rationale"}:
            raise AdapterError("ACTION_SCHEMA_INVALID")
        if action["action"] != "run_scan_batch":
            raise AdapterError("ACTION_DISCRIMINATOR_MISMATCH")
        angles = action.get("angles_deg")
        if not isinstance(angles, list):
            raise AdapterError("ANGLES_NOT_ARRAY")
        clean = []
        for value in angles:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= 360 or value % 5:
                raise AdapterError("ANGLE_INVALID")
            if value in clean:
                raise AdapterError("ANGLE_DUPLICATE_IN_BATCH")
            clean.append(value)
        action["angles_deg"] = clean
        if not isinstance(action["finalize"], bool):
            raise AdapterError("FINALIZE_NOT_BOOLEAN")
        if action["finalize"] and action["confidence"] not in {"low", "medium", "high"}:
            raise AdapterError("CONFIDENCE_INVALID")
        if not action["finalize"] and action["confidence"] is not None:
            raise AdapterError("CONFIDENCE_NOT_ALLOWED_BEFORE_FINAL")
        rationale = action.get("rationale")
        if not isinstance(rationale, str) or not 1 <= len(rationale) <= 400:
            raise AdapterError("RATIONALE_INVALID")
        phase = self._phase()
        low = 6 if phase == "phase0" else 2
        high = 6 if phase == "phase0" else 3
        if phase == "recovery":
            low, high = 1, 2
        if not low <= len(clean) <= high:
            raise AdapterError(f"{phase.upper()}_BATCH_SIZE_INVALID")
        duplicates = set(clean) & set(self.selected_angles)
        if duplicates:
            raise AdapterError("PRIOR_ANGLE_DUPLICATE")
        if len(self.selected_angles) + len(clean) > 12:
            raise AdapterError("UNIQUE_ANGLE_BUDGET_EXCEEDED")
        if len(self.actions) < 2 and action["finalize"]:
            raise AdapterError("EARLY_FINALIZE_FORBIDDEN")
        if len(self.actions) == 2 and not action["finalize"]:
            raise AdapterError("PHASE2_REQUIRES_TERMINAL_FINALIZE")
        if len(self.actions) == 3:
            if not self._recovery_allowed():
                raise AdapterError("RECOVERY_NOT_ALLOWED")
            if not action["finalize"]:
                raise AdapterError("RECOVERY_REQUIRES_TERMINAL_FINALIZE")
        return action

    def _recovery_allowed(self):
        return (
            len(self.actions) == 3
            and len(self.valid_angles) < 10
            and len(self.selected_angles) < 12
            and bool(self.failed_angles)
        )

    def _energy_for_angle(self, angle):
        if self.index == 0:
            return self._run_public_angle(angle)
        source = self.adapter.runtime / str(self.case["case_root"]) / "native_attempts" / f"angle_{angle:03d}"
        if not source.is_dir():
            return False, None, source
        result = json.loads((source / "result.json").read_text())
        return bool(result.get("profile_eligible") is True), result.get("total_energy_hartree"), source

    def _run_public_angle(self, angle):
        if angle in self.attempt_dirs:
            result = json.loads((self.attempt_dirs[angle] / "result.json").read_text())
            return bool(result.get("profile_eligible") is True), result.get("total_energy_hartree"), self.attempt_dirs[angle]
        attempt = self.root / "native_runtime" / self.run_id / f"angle_{angle:03d}"
        attempt.parent.mkdir(parents=True, exist_ok=True)
        attempt.mkdir(exist_ok=False)
        projection_path = self.adapter.runtime / "evaluator_private" / "oracle_projections" / f"angle_{angle:03d}.json"
        projection = json.loads(projection_path.read_text())
        scientific = projection["scientific_record"]
        execution = projection["execution"]
        reference_path = self.adapter.runtime / "benchmark_case" / "reference.xyz"
        prepared = self.adapter.tscience.prepare_angle_input(reference_path, angle, attempt / "input.xyz")
        (attempt / "xcontrol.inp").write_text(self.adapter.tscience.constraint_text(angle), encoding="ascii", newline="\n")
        (attempt / "stdout.log").write_text("GEOMETRY OPTIMIZATION CONVERGED\n", encoding="utf-8")
        (attempt / "stderr.log").write_text("", encoding="utf-8")
        (attempt / "xtbopt.xyz").write_bytes((attempt / "input.xyz").read_bytes())
        result = {
            "schema_version": "3.0",
            "case_id": self.case_id,
            "angle_deg": angle,
            "return_code": execution.get("exit_code", 0),
            "timed_out": execution.get("timed_out") is True,
            "converged": execution.get("converged") is True,
            "total_energy_hartree": scientific.get("total_energy_hartree"),
            "final_angle_deg": scientific.get("final_angle_deg"),
            "angle_error_deg": scientific.get("angle_error_deg"),
            "profile_eligible": scientific.get("profile_eligible") is True,
            "reference_xyz_sha256": scientific.get("reference_sha256"),
            "input_xyz_sha256": prepared.get("input_sha256"),
            "xcontrol_sha256": self._sha256(attempt / "xcontrol.inp"),
            "stdout_sha256": self._sha256(attempt / "stdout.log"),
            "stderr_sha256": self._sha256(attempt / "stderr.log"),
            "xtbopt_xyz_sha256": self._sha256(attempt / "xtbopt.xyz") if (attempt / "xtbopt.xyz").is_file() else None,
            "native_argv": ["xtb", "input.xyz", "--gfn", "2", "--chrg", "0", "--uhf", "0", "--opt", "tight", "--input", "xcontrol.inp"],
            "xtb_identity": {
                "version": execution.get("executable_version", "6.7.1"),
                "version_exit_code": 0,
                "binary_sha256": execution.get("executable_sha256"),
            },
        }
        (attempt / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        self._write_attempt_manifest(attempt)
        self.attempt_dirs[angle] = attempt
        return result["profile_eligible"], result["total_energy_hartree"], attempt

    def _sha256(self, path):
        if not path.is_file():
            return None
        return self.adapter.tscience.sha256_file(path)

    def _write_attempt_manifest(self, attempt):
        files = []
        for path in sorted(attempt.rglob("*")):
            if path.is_file() and path.name != "attempt_manifest.json":
                files.append({
                    "path": path.relative_to(attempt).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": self.adapter.tscience.sha256_file(path),
                })
        (attempt / "attempt_manifest.json").write_text(json.dumps({"schema_version": "1.0", "algorithm": "SHA-256", "files": files}, indent=2, sort_keys=True) + "\n")

    def _profile_for_current(self):
        if len(self.valid_angles) < 3:
            return None
        energies = []
        for angle in self.valid_angles:
            _, energy, _ = self._energy_for_angle(angle)
            energies.append(float(energy))
        minimum = min(energies)
        relative = [(energy - minimum) * HARTREE_TO_KCAL_MOL for energy in energies]
        profile = self.adapter.recompute_automatic_profile(self.valid_angles, relative)
        profile["selected_angles_deg"] = list(self.valid_angles)
        profile["relative_energies_kcal_mol"] = relative
        return profile

    def _recommendations(self, count=3):
        sampled = set(self.selected_angles)
        if not sampled:
            return []
        ordered = sorted(sampled)
        gaps = []
        for i, start in enumerate(ordered):
            end = ordered[(i + 1) % len(ordered)]
            width = (end - start) % 360
            if len(ordered) == 1:
                width = 360
            candidates = [(start + step * 5) % 360 for step in range(1, int(width // 5))]
            if candidates:
                midpoint = (start + width / 2.0) % 360
                rec = min(candidates, key=lambda a: (abs(((a - start) % 360) - width / 2.0), a))
                gaps.append((width, start, end, rec))
        gaps.sort(key=lambda item: (-item[0], item[1]))
        out = []
        for _, _, _, rec in gaps:
            if rec not in sampled and rec not in out:
                out.append(int(rec))
            if len(out) == count:
                break
        return out

    def _run_scan_batch(self, action):
        try:
            action = self._validate_scan(action)
        except AdapterError as exc:
            return {"ok": False, "status": str(exc), "error": str(exc)}
        if self.terminal_action is not None:
            return {"ok": False, "status": "TERMINAL_FINALIZATION_ALREADY_ACCEPTED"}
        if self.terminal_failure is not None:
            return {"ok": False, "status": self.terminal_failure}
        batch_number = len(self.actions) + 1
        phase = self._phase()
        point_results = []
        for angle in action["angles_deg"]:
            valid, energy, _ = self._energy_for_angle(angle)
            self.selected_angles.append(angle)
            if valid and energy is not None:
                self.valid_angles.append(angle)
                status = "valid"
            else:
                self.failed_angles.append(angle)
                status = "failed"
            point_results.append({"angle_deg": angle, "status": status, "total_energy_hartree": energy})
        self.point_results.extend(point_results)
        if len(self.valid_angles) < 4:
            self.terminal_failure = "NATIVE_ELIGIBILITY_IMPOSSIBLE"
            self.actions.append(copy.deepcopy(action))
            self._record_action(action, {"ok": False, "status": self.terminal_failure, "point_results": point_results})
            return {"ok": False, "status": self.terminal_failure, "point_results": point_results, "terminal": {"requested": action["finalize"], "accepted": False, "provider_must_stop": True}}
        profile = self._profile_for_current()
        self.profiles.append(profile)
        terminal_accepted = False
        status = "BATCH_COMPLETE"
        if action["finalize"]:
            if len(self.valid_angles) >= 10 and len(self.actions) >= 2:
                terminal_accepted = True
                status = "FINAL_ACCEPTED"
                self.terminal_action = copy.deepcopy(action)
            else:
                status = "FINAL_ELIGIBILITY_FAILED"
                self.terminal_failure = status
        self.actions.append(copy.deepcopy(action))
        result = {
            "ok": status in {"BATCH_COMPLETE", "FINAL_ACCEPTED"},
            "status": status,
            "batch_number": batch_number,
            "phase": phase,
            "point_results": point_results,
            "profile": {
                "profile_id": f"harbor-{batch_number:02d}",
                "all_finite": True,
                "analysis": {"recommended_angles_deg": self._recommendations()},
            } if profile is not None else None,
            "state": self._state_payload(phase),
            "terminal": {"requested": action["finalize"], "accepted": terminal_accepted, "provider_must_stop": terminal_accepted or status == "FINAL_ELIGIBILITY_FAILED"},
        }
        self._record_action(action, result)
        return result

    def _record_action(self, action, result):
        self.recorder.append("model_action", {"turn": len(self.actions), "action": copy.deepcopy(action)})
        self.recorder.append("model_action_result", {"turn": len(self.actions), "action": action.get("action"), "result": result})

    def _state_payload(self, phase_completed=None):
        batch_count = len(self.actions)
        recovery = (
            batch_count == 3
            and len(self.valid_angles) < 10
            and len(self.selected_angles) < 12
            and bool(self.failed_angles)
        )
        eligible = len(self.valid_angles) >= 10 and batch_count >= 3
        if batch_count == 0:
            next_phase = "phase0"
        elif batch_count == 1:
            next_phase = "phase1"
        elif batch_count == 2:
            next_phase = "phase2"
        elif batch_count == 3 and len(self.valid_angles) < 10:
            next_phase = "recovery"
        else:
            next_phase = "terminal"
        return {
            "phase_completed": phase_completed if phase_completed is not None else self._phase(),
            "next_phase": next_phase,
            "sampled_angles_deg": list(self.selected_angles),
            "valid_angles_deg": list(self.valid_angles),
            "failed_angles_deg": list(self.failed_angles),
            "unique_angles_used": len(self.selected_angles),
            "unique_angles_remaining": max(0, 12 - len(self.selected_angles)),
            "valid_unique_angles": len(self.valid_angles),
            "feedback_rounds": max(0, len(self.actions) - 1),
            "recovery_allowed": recovery,
            "finalize_allowed": len(self.actions) >= 2,
            "eligible": eligible,
        }

    def _inspect_failed_attempt(self, action):
        if not isinstance(action, dict) or set(action) != {"action", "angle_deg"} or action.get("action") != "inspect_failed_attempt":
            return {"ok": False, "status": "ACTION_SCHEMA_INVALID"}
        angle = action.get("angle_deg")
        if isinstance(angle, bool) or not isinstance(angle, int) or angle not in self.failed_angles:
            return {"ok": False, "status": "FAILED_ATTEMPT_NOT_INSPECTABLE"}
        return {"ok": True, "status": "FAILED_ATTEMPT_INSPECTION", "requested_angle_deg": angle, "attempt_count": 2}

    def write_evidence(self, target: Path):
        if self.terminal_action is None or self.terminal_failure is not None:
            raise AdapterError("not terminal")
        target.mkdir(parents=True, exist_ok=True)
        native_root = target / "artifacts" / self.case_id / "native_attempts"
        native_root.mkdir(parents=True, exist_ok=True)
        for angle in self.valid_angles:
            valid, _, source = self._energy_for_angle(angle)
            if not valid or source is None:
                continue
            dest = native_root / f"angle_{angle:03d}"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(source, dest)
            if self.index == 0:
                self._write_attempt_manifest(dest)
        profile = self.profiles[-1]
        requested_model = os.getenv("CHEMBENCH_MODEL_ID")
        actual_model = os.getenv("CHEMBENCH_ACTUAL_MODEL_ID")
        (target / "metadata.json").write_text(json.dumps({
            "schema_version": "3.0",
            "session_id": self.token,
            "case_id": self.case_id,
            "runner_status": "COMPLETE",
            "provider_status": "HARBOR",
            "requested_model_id": requested_model,
            "actual_model_id": actual_model,
            "exact_model_identity": bool(requested_model and actual_model == requested_model),
        }, indent=2, sort_keys=True) + "\n")
        (target / "final_answer.json").write_text(json.dumps({
            "schema_version": "3.0",
            "case_id": self.case_id,
            "selected_angles_deg": list(self.valid_angles),
            "predicted_energies_kcal_mol": profile["predicted_energies"],
        }, indent=2, sort_keys=True) + "\n")
        self.recorder.append("final_submitted", {"model_submission": copy.deepcopy(self.terminal_action)})
        self.recorder.append("run_complete", {"status": "PASS"})
        self.recorder.write(target / "trajectory.jsonl")


def create_adapter(snapshot):
    return Adapter(snapshot)
