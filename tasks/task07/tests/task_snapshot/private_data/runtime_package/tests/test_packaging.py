from __future__ import annotations

import json
import shutil
import stat
import sys
import uuid
import zipfile
from pathlib import Path, PurePosixPath

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import package_submission as packager  # noqa: E402
import validate_package as validator  # noqa: E402
from framework.schemas import ModelConfig  # noqa: E402


@pytest.fixture
def case_workspace() -> Path:
    parent = PROJECT_ROOT / "tests" / ".packaging_cases"
    parent.mkdir(parents=True, exist_ok=True)
    case = parent / uuid.uuid4().hex
    case.mkdir()
    try:
        yield case
    finally:
        shutil.rmtree(case, ignore_errors=True)
        try:
            parent.rmdir()
        except OSError:
            pass


def test_ready_status_is_derived_and_cannot_be_forced(case_workspace: Path) -> None:
    project = case_workspace / "incomplete_project"
    (project / "tests").mkdir(parents=True)
    (project / "tests" / "test_smoke.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    assessment = packager.derive_readiness(project, test_result={"passed": True})
    assert assessment["delivery_status"] == "BLOCKED"
    assert assessment["checks"]["project_license"]["passed"] is False
    assert assessment["checks"]["project_notice"]["passed"] is False
    assert assessment["checks"]["real_trajectories"]["passed"] is False

    staging = case_workspace / "stage"
    staging.mkdir()
    (staging / "README.md").write_text("partial review package\n", encoding="utf-8")
    manifest = packager.build_manifest(staging, assessment)
    assert manifest["delivery_status"] == "BLOCKED"
    assert manifest["blocking_reasons"]


def test_zip_slip_duplicate_and_symlink_entries_are_rejected(case_workspace: Path) -> None:
    slip = case_workspace / "slip.zip"
    with zipfile.ZipFile(slip, "w") as archive:
        archive.writestr("root/../escape.txt", "no")
    slip_result = validator.preflight_archive(slip)
    assert slip_result["passed"] is False
    assert any("parent" in error or "unsafe" in error for error in slip_result["errors"])

    duplicate = case_workspace / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("root/README.md", "one")
        archive.writestr("root/readme.md", "two")
    duplicate_result = validator.preflight_archive(duplicate)
    assert duplicate_result["passed"] is False
    assert any("colliding" in error for error in duplicate_result["errors"])

    linked = case_workspace / "symlink.zip"
    info = zipfile.ZipInfo("root/README.md")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(linked, "w") as archive:
        archive.writestr(info, "target.txt")
    link_result = validator.preflight_archive(linked)
    assert link_result["passed"] is False
    assert any("symlink" in error for error in link_result["errors"])


def test_secret_scanner_finds_nonliteral_test_credential(case_workspace: Path) -> None:
    source = case_workspace / "README.md"
    fake_value = "not-a-real-" + "credential-value-12345"
    source.write_text("Authorization: Bearer " + fake_value + "\n", encoding="utf-8")

    findings = packager.scan_for_secrets([source], case_workspace)
    assert findings == [{"path": "README.md", "line": 1, "rule": "bearer"}]


def test_secret_scanners_detect_configured_api_key_assignment(
    case_workspace: Path,
) -> None:
    source = case_workspace / "README.md"
    configured_name = "GJLD_" + "API_KEY"
    synthetic_value = "synthetic-" + "credential-material-123456789"
    source.write_text(
        configured_name + "=" + synthetic_value + "\n", encoding="utf-8"
    )

    expected = {
        "path": "README.md",
        "line": 1,
        "rule": "configured_api_key_assignment",
    }
    assert packager.scan_for_secrets([source], case_workspace) == [expected]
    validation = validator.check_security(case_workspace)
    assert validation["passed"] is False
    assert validation["secret_findings"] == [expected]


def test_manifest_hashes_and_inventory_are_verified(case_workspace: Path) -> None:
    root = case_workspace / "package"
    root.mkdir()
    readme = root / "README.md"
    readme.write_text("manifest fixture\n", encoding="utf-8")
    assessment = {
        "delivery_status": "BLOCKED",
        "blocking_reasons": ["fixture intentionally incomplete"],
        "checks": {},
    }
    manifest = packager.build_manifest(root, assessment)
    (root / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assert validator.check_manifest(root)["passed"] is True
    readme.write_text("tampered\n", encoding="utf-8")
    result = validator.check_manifest(root)
    assert result["passed"] is False
    assert any("mismatch" in error for error in result["errors"])


def test_permanent_extraction_refuses_nonempty_destination(case_workspace: Path) -> None:
    destination = case_workspace / "already_used"
    destination.mkdir()
    (destination / "evidence.txt").write_text("keep\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        validator.extract_archive_safely(
            case_workspace / "unused.zip", destination, {"passed": True}
        )
    assert (destination / "evidence.txt").read_text(encoding="utf-8") == "keep\n"


def test_workspace_paths_keep_staging_validation_and_temp_below_caller_root(
    case_workspace: Path,
) -> None:
    workspace = case_workspace / "caller_workspace"
    workspace.mkdir()
    assert packager.resolve_write_path("outputs", workspace) == workspace / "outputs"
    assert packager.resolve_write_path("submission_packages", workspace) == workspace / "submission_packages"
    assert validator.resolve_write_path("temporary_validation/extracted", workspace) == (
        workspace / "temporary_validation" / "extracted"
    )
    with pytest.raises(ValueError):
        packager.resolve_write_path(workspace.parent / "outside", workspace)
    with pytest.raises(ValueError):
        validator.resolve_write_path(workspace, workspace)

    assert packager.derived_workspace_root(PROJECT_ROOT) == PROJECT_ROOT.parents[1]
    assert validator.derived_workspace_root(PROJECT_ROOT) == PROJECT_ROOT.parents[1]


def test_runner_metadata_identity_contract_uses_exact_provider_mapping() -> None:
    config = ModelConfig.from_mapping(
        {
            "provider": "siliconflow",
            "transport": "openai_compatible",
            "model": "Pro/moonshotai/Kimi-K2.6",
            "base_url": "https://api.siliconflow.cn/v1",
            "api_key_env": "GJLD_API_KEY",
        }
    )
    actual_model = config.model
    metadata = {
        "status": "COMPLETE",
        "execution_mode": "REAL_API",
        "is_real_model_run": True,
        "failure": None,
        "provider": config.provider,
        "transport": config.transport,
        "base_url": config.base_url,
        "endpoint": config.base_url,
        "request_endpoint": config.base_url + "/chat/completions",
        "requested_model": config.model,
        "requested_model_id": config.model,
        "model": actual_model,
        "actual_model_id": actual_model,
        "observed_model_ids": [actual_model],
        "model_config": config.sanitized_dict(),
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timestamp": "2026-07-18T12:00:00+00:00",
        "provider_response_metadata": [{"model": actual_model}],
    }

    package_consistency = packager.check_model_identity_consistency(
        metadata, expected_directory="kimi"
    )
    validation_consistency = validator.check_model_identity_consistency(
        metadata, expected_directory="kimi"
    )
    assert package_consistency["passed"] is True
    assert validation_consistency["passed"] is True
    assert validation_consistency["trusted_endpoint"] is True
    assert validation_consistency["response_model_verified"] is True
    assert validation_consistency["run_status_complete"] is True

    untrusted = dict(metadata)
    untrusted["base_url"] = "https://api.siliconflow.example/v1"
    untrusted["endpoint"] = "https://api.siliconflow.example/v1"
    assert validator.check_model_identity_consistency(untrusted)["passed"] is False
    assert validator.check_model_identity_consistency(
        metadata, expected_directory="glm"
    )["passed"] is False

    for rejected_actual in (
        config.model + "-202607",
        "moonshotai/Kimi-K2.6",
    ):
        substituted = json.loads(json.dumps(metadata))
        substituted["model"] = rejected_actual
        substituted["actual_model_id"] = rejected_actual
        substituted["observed_model_ids"] = [rejected_actual]
        substituted["provider_response_metadata"] = [{"model": rejected_actual}]
        assert packager.check_model_identity_consistency(
            substituted, expected_directory="kimi"
        )["passed"] is False
        assert validator.check_model_identity_consistency(
            substituted, expected_directory="kimi"
        )["passed"] is False

    wrong_key_env = json.loads(json.dumps(metadata))
    wrong_key_env["model_config"]["api_key_env"] = "OTHER_API_KEY"
    assert packager.check_model_identity_consistency(
        wrong_key_env, expected_directory="kimi"
    )["passed"] is False
    assert validator.check_model_identity_consistency(
        wrong_key_env, expected_directory="kimi"
    )["passed"] is False


def test_packaging_allowlist_requires_legal_files_and_exact_trajectory_layout() -> None:
    assert "LICENSE" in packager.ROOT_FILES
    assert "NOTICE" in packager.ROOT_FILES
    assert "LICENSE" in validator.ROOT_FILES
    assert "NOTICE" in validator.ROOT_FILES
    assert "framework/siliconflow_adapter.py" in packager.REQUIRED_CORE_FILES
    assert "framework/siliconflow_adapter.py" in validator.REQUIRED_STRUCTURE_FILES
    default_filename = packager.build_parser().parse_args([]).filename
    assert default_filename.startswith("qed_mol_design_agent_benchmark_v3_")
    assert default_filename.endswith(".zip")
    assert "configs/model_transition.json" in packager.EXACT_FILES
    assert "configs/model_transition.json" in validator.EXACT_FILES
    assert set(packager.REQUIRED_REPORTS) == set(validator.REQUIRED_REPORTS)
    for report in (
        "reports/TRAJECTORY_HANDOFF_20260718.md",
        "reports/model_transition_report.md",
        "reports/legacy_model_failures.md",
        "reports/final_release_test_report.md",
        "reports/final_security_report.md",
    ):
        assert report in packager.REQUIRED_REPORTS
    # This post-package report binds the final ZIP SHA and validator result, so
    # requiring it inside the ZIP would create an impossible self-reference.
    assert "reports/final_release_report.md" not in packager.REQUIRED_REPORTS
    for checker in (
        packager.is_package_relative_allowed,
        validator.is_package_relative_allowed,
    ):
        assert checker(PurePosixPath("trajectories/README.md")) is True
        assert checker(PurePosixPath("framework/siliconflow_adapter.py")) is True
        assert checker(PurePosixPath("trajectories/glm/metadata.json")) is True
        assert checker(PurePosixPath("trajectories/deepseek/trajectory.jsonl")) is True
        assert checker(PurePosixPath("trajectories/kimi/evaluation.json")) is True
        assert checker(PurePosixPath("trajectories/other/metadata.json")) is False
        assert checker(PurePosixPath("trajectories/glm/debug.json")) is False
        assert checker(PurePosixPath("trajectories/glm/raw_response.json")) is False
        assert checker(PurePosixPath("reports/runtime.log")) is False
        assert checker(PurePosixPath("reports/credentials_backup.json")) is False
        assert checker(PurePosixPath("reports/raw_response_001.json")) is False
        assert checker(PurePosixPath("reports/.env.yaml")) is False
        assert checker(PurePosixPath(".env")) is False


def test_positive_objective_gate_rejects_ties_and_zero_scores() -> None:
    task_config = {
        "task": {"objective": {"require_strict_improvement": True}}
    }
    evaluation = {
        "status": "ok",
        "valid_submission": True,
        "eligible": True,
        "hard_constraints_passed": True,
        "objective_passed": True,
        "process_eligible": True,
        "score": 82.5,
        "total_score": 82.5,
        "score_breakdown": {"qed_improvement": 12.5},
        "metrics": {"start_qed": 0.70, "final_qed": 0.80, "qed_delta": 0.10},
        "errors": [],
    }
    assert packager.check_positive_objective_evidence(
        evaluation, task_config
    )["passed"] is True
    assert validator.check_positive_objective_evidence(
        evaluation, task_config
    )["passed"] is True

    tied = json.loads(json.dumps(evaluation))
    tied["metrics"] = {"start_qed": 0.70, "final_qed": 0.70, "qed_delta": 0.0}
    tied["score_breakdown"]["qed_improvement"] = 0.0
    assert validator.check_positive_objective_evidence(tied, task_config)["passed"] is False

    zero_score = json.loads(json.dumps(evaluation))
    zero_score["score"] = zero_score["total_score"] = 0.0
    assert packager.check_positive_objective_evidence(
        zero_score, task_config
    )["passed"] is False


def test_trajectory_readiness_requires_exact_named_directory_set(
    case_workspace: Path,
) -> None:
    project = case_workspace / "trajectory_layout"
    for name in ("glm", "deepseek", "renamed_kimi"):
        (project / "trajectories" / name).mkdir(parents=True)

    packaged = packager.assess_trajectories(project)
    validated = validator.check_trajectories(project)
    for result in (packaged, validated):
        assert result["passed"] is False
        assert result["missing_directories"] == ["kimi"]
        assert result["unexpected_directories"] == ["renamed_kimi"]
        assert result["expected_directory_models"] == {
            "glm": "zai-org/GLM-5.2",
            "deepseek": "deepseek-ai/DeepSeek-V4-Pro",
            "kimi": "Pro/moonshotai/Kimi-K2.6",
        }


def test_process_gate_rejects_budget_or_too_few_unique_candidates() -> None:
    tool_events = [
        {
            "event": "tool_call",
            "tool": "evaluate_candidate",
            "status": "SUCCESS",
            "result": {"ok": True, "valid": True, "canonical_smiles": "CCN"},
        }
    ]
    metadata = {"budget_exceeded": False}
    final_answer = {
        "run_metadata": {
            "budget_exceeded": False,
            "tool_call_count": 1,
            "unique_evaluated_candidates": ["CCN"],
        }
    }
    evaluation = {
        "status": "ok",
        "valid_submission": True,
        "process_eligible": True,
        # Chemical quality is intentionally irrelevant to this process gate.
        "eligible": False,
        "hard_constraints_passed": False,
        "process_evidence": {
            "valid": True,
            "budget_exceeded": False,
            "tool_call_count": 1,
            "unique_evaluated_candidates": ["CCN"],
        },
        "process_checks": {"fixture_check": True},
    }
    arguments = {
        "metadata": metadata,
        "final_answer": final_answer,
        "stored_evaluation": evaluation,
        "accepted_tool_events": tool_events,
        "successful_tool_events": tool_events,
    }

    assert validator.assess_process_eligibility(
        **arguments, minimum_unique_candidates=1
    )["passed"] is True
    assert packager.assess_process_eligibility(
        **arguments, minimum_unique_candidates=2
    )["passed"] is False
    assert validator.assess_process_eligibility(
        **arguments, minimum_unique_candidates=2
    )["passed"] is False

    over_budget_metadata = {"budget_exceeded": True}
    assert validator.assess_process_eligibility(
        **{**arguments, "metadata": over_budget_metadata},
        minimum_unique_candidates=1,
    )["passed"] is False


def test_complete_lifecycle_rejects_post_terminal_events() -> None:
    events = [
        {"event": "run_start"},
        {"event": "provider_request"},
        {"event": "provider_response"},
        {"event": "message", "source": "provider", "message": {"role": "assistant"}},
        {"event": "tool_call"},
        {"event": "provider_request"},
        {"event": "provider_response"},
        {"event": "message", "source": "provider", "message": {"role": "assistant"}},
        {"event": "run_complete", "status": "COMPLETE"},
    ]
    assert packager.check_complete_run_lifecycle(events)["passed"] is True
    assert validator.check_complete_run_lifecycle(events)["passed"] is True

    appended = [*events, {"event": "provider_response"}]
    result = validator.check_complete_run_lifecycle(appended)
    assert result["passed"] is False
    assert any("final event" in error for error in result["errors"])


def test_real_readiness_requires_iterative_candidate_feedback_rounds() -> None:
    candidate_a = {
        "event": "tool_call",
        "call_id": "call-a",
        "tool": "evaluate_candidate",
        "status": "SUCCESS",
        "result": {"ok": True, "valid": True, "canonical_smiles": "CCN"},
    }
    candidate_b = {
        "event": "tool_call",
        "call_id": "call-b",
        "tool": "evaluate_candidate",
        "status": "SUCCESS",
        "result": {"ok": True, "valid": True, "canonical_smiles": "CCO"},
    }
    one_batch = [
        {"event": "provider_request", "turn": 1, "messages": []},
        {"event": "provider_response", "turn": 1},
        {
            "event": "message",
            "source": "provider",
            "message": {
                "role": "assistant",
                "tool_calls": [{"id": "call-a"}, {"id": "call-b"}],
            },
        },
        candidate_a,
        candidate_b,
    ]
    assert validator.assess_closed_loop_rounds(one_batch)["passed"] is False

    feedback_a = {
        "role": "tool",
        "tool_call_id": "call-a",
        "name": "evaluate_candidate",
        "content": "measured feedback A",
    }
    iterative = [
        {"event": "provider_request", "turn": 1, "messages": []},
        {"event": "provider_response", "turn": 1},
        {
            "event": "message",
            "source": "provider",
            "message": {
                "role": "assistant",
                "tool_calls": [{"id": "call-a"}],
            },
        },
        candidate_a,
        {"event": "message", "source": "rdkit_tool", "message": feedback_a},
        {
            "event": "provider_request",
            "turn": 2,
            "messages": [feedback_a],
        },
        {"event": "provider_response", "turn": 2},
        {
            "event": "message",
            "source": "provider",
            "message": {
                "role": "assistant",
                "tool_calls": [{"id": "call-b"}],
            },
        },
        candidate_b,
    ]
    result = packager.assess_closed_loop_rounds(iterative)
    assert result["passed"] is True
    assert validator.assess_closed_loop_rounds(iterative)["passed"] is True
    assert result["closed_loop_tool_round_count"] == 2
    assert result["candidate_feedback_round_count"] == 2

    ok_false_rounds = json.loads(json.dumps(iterative))
    ok_false_events = [
        event
        for event in ok_false_rounds
        if event.get("event") == "tool_call"
    ]
    for event in ok_false_events:
        event["result"]["ok"] = False
        assert event["status"] == "SUCCESS"  # Still an accepted/countable attempt.
        assert packager._is_successful_tool_event(event) is False
        assert validator._is_successful_tool_event(event) is False
    for checker in (
        packager.assess_closed_loop_rounds,
        validator.assess_closed_loop_rounds,
    ):
        rejected = checker(ok_false_rounds)
        assert rejected["passed"] is False
        assert rejected["closed_loop_tool_round_count"] == 0
        assert rejected["candidate_feedback_round_count"] == 0


def test_closed_loop_gate_rejects_unpaired_and_turn_fabrication() -> None:
    candidate_a = {
        "event": "tool_call",
        "call_id": "call-a",
        "tool": "evaluate_candidate",
        "status": "SUCCESS",
        "result": {"ok": True, "valid": True, "canonical_smiles": "CCN"},
    }
    candidate_b = {
        "event": "tool_call",
        "call_id": "call-b",
        "tool": "evaluate_candidate",
        "status": "SUCCESS",
        "result": {"ok": True, "valid": True, "canonical_smiles": "CCO"},
    }
    feedback_a = {
        "role": "tool",
        "tool_call_id": "call-a",
        "name": "evaluate_candidate",
        "content": "measured feedback A",
    }
    unpaired = [
        {"event": "provider_request", "turn": 1, "messages": []},
        {"event": "provider_response", "turn": 1},
        candidate_a,
        {"event": "message", "source": "rdkit_tool", "message": feedback_a},
        {"event": "provider_request", "turn": 2, "messages": [feedback_a]},
        {"event": "provider_response", "turn": 2},
        candidate_b,
    ]
    for checker in (
        packager.assess_closed_loop_rounds,
        validator.assess_closed_loop_rounds,
    ):
        result = checker(unpaired)
        assert result["passed"] is False
        assert any("without an ordered provider assistant" in error for error in result["errors"])

    paired_first_round = [
        {"event": "provider_request", "turn": 1, "messages": []},
        {"event": "provider_response", "turn": 1},
        {
            "event": "message",
            "source": "provider",
            "message": {"role": "assistant", "tool_calls": [{"id": "call-a"}]},
        },
        candidate_a,
        {"event": "message", "source": "rdkit_tool", "message": feedback_a},
    ]
    fabricated_turn = [
        *paired_first_round,
        {"event": "provider_request", "turn": 2, "messages": [feedback_a]},
        {"event": "provider_response", "turn": 99},
        {
            "event": "message",
            "source": "provider",
            "message": {"role": "assistant", "tool_calls": [{"id": "call-b"}]},
        },
        candidate_b,
        {"event": "provider_request", "turn": 99, "messages": [feedback_a]},
    ]
    for checker in (
        packager.assess_closed_loop_rounds,
        validator.assess_closed_loop_rounds,
    ):
        result = checker(fabricated_turn)
        assert result["passed"] is False
        assert any("mismatched provider_response" in error for error in result["errors"])


def test_dependency_consistency_rejects_injected_modes() -> None:
    provenance = {
        name: {
            "mode": "default",
            "canonical": True,
            "identifier": identifier,
            "attestation": validator.CANONICAL_DEPENDENCY_ATTESTATIONS[name],
        }
        for name, identifier in validator.CANONICAL_DEPENDENCY_IDENTIFIERS.items()
    }
    metadata = {
        "dependency_mode": "CANONICAL",
        "canonical_dependencies": True,
        "dependency_provenance": provenance,
    }
    assert validator.check_dependency_consistency(metadata)["passed"] is True
    assert packager.check_dependency_consistency(metadata)["passed"] is True
    injected = json.loads(json.dumps(metadata))
    injected["dependency_provenance"]["evaluator"].update(
        {"mode": "injected_canonical", "attestation": "identity_verified"}
    )
    assert packager.check_dependency_consistency(injected)["passed"] is False
    assert validator.check_dependency_consistency(injected)["passed"] is False

    for name in validator.CANONICAL_DEPENDENCY_IDENTIFIERS:
        weak = json.loads(json.dumps(metadata))
        weak["dependency_provenance"][name]["attestation"] = "runner_default"
        assert packager.check_dependency_consistency(weak)["passed"] is False
        assert validator.check_dependency_consistency(weak)["passed"] is False


@pytest.fixture
def frozen_kimi_transition_case(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "project"
    metadata = json.loads((PROJECT_ROOT / "trajectories/kimi/metadata.json").read_text(encoding="utf-8"))
    relative_files = {
        "configs/model_transition.json",
        "configs/task.yaml",
        "configs/models.example.yaml",
        "prompts/agent_prompt.md",
        *metadata["hashes"]["sources"].keys(),
    }
    for relative in sorted(relative_files):
        source = PROJECT_ROOT / Path(*PurePosixPath(relative).parts)
        target = root / Path(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copytree(PROJECT_ROOT / "trajectories/kimi", root / "trajectories/kimi")
    return root, metadata


@pytest.mark.parametrize(
    "checker",
    (packager._verify_hash_evidence, validator._check_source_hashes),
)
def test_frozen_kimi_transition_is_byte_reversible_and_symmetric(
    frozen_kimi_transition_case: tuple[Path, dict[str, object]], checker: object
) -> None:
    root, metadata = frozen_kimi_transition_case
    # The v4 structural revision intentionally invalidates v3 frozen source
    # hashes; old trajectories remain evidence but cannot attest the new task.
    assert checker(root, metadata, expected_directory="kimi")


@pytest.mark.parametrize(
    "checker",
    (packager._verify_hash_evidence, validator._check_source_hashes),
)
def test_frozen_kimi_transition_cannot_be_reused_by_other_models(
    frozen_kimi_transition_case: tuple[Path, dict[str, object]], checker: object
) -> None:
    root, metadata = frozen_kimi_transition_case
    errors = checker(root, metadata, expected_directory="glm")
    assert "source hash mismatch: configs/models.example.yaml" in errors
    assert "source hash mismatch: framework/schemas.py" in errors


@pytest.mark.parametrize(
    "mutation",
    ("missing_manifest", "extra_manifest_key", "source_byte", "kimi_artifact", "metadata_hash", "identity"),
)
@pytest.mark.parametrize(
    "checker",
    (packager._verify_hash_evidence, validator._check_source_hashes),
)
def test_frozen_kimi_transition_rejects_tampering(
    frozen_kimi_transition_case: tuple[Path, dict[str, object]],
    checker: object,
    mutation: str,
) -> None:
    root, original_metadata = frozen_kimi_transition_case
    metadata = json.loads(json.dumps(original_metadata))
    manifest_path = root / "configs/model_transition.json"
    if mutation == "missing_manifest":
        manifest_path.unlink()
    elif mutation == "extra_manifest_key":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["unauthorized"] = True
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif mutation == "source_byte":
        path = root / "framework/schemas.py"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif mutation == "kimi_artifact":
        path = root / "trajectories/kimi/dialogue.md"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif mutation == "metadata_hash":
        metadata["hashes"]["sources"]["framework/schemas.py"] = "0" * 64
    elif mutation == "identity":
        metadata["endpoint"] = "https://example.invalid/v1"
    errors = checker(root, metadata, expected_directory="kimi")
    assert errors


@pytest.mark.parametrize(
    "checker",
    (packager._verify_hash_evidence, validator._check_source_hashes),
)
def test_frozen_kimi_transition_does_not_hide_third_source_mismatch(
    frozen_kimi_transition_case: tuple[Path, dict[str, object]], checker: object
) -> None:
    root, metadata = frozen_kimi_transition_case
    path = root / "evaluator/evaluate.py"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    errors = checker(root, metadata, expected_directory="kimi")
    assert "source hash mismatch: evaluator/evaluate.py" in errors
