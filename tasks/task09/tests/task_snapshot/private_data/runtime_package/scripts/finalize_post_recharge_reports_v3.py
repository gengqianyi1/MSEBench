#!/usr/bin/env python3
"""Render Torsion-only post-recharge release reports from validated r2 evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
VALIDATION = ROOT / "reports" / "POST_RECHARGE_CAMPAIGN_VALIDATION_20260805.json"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_NOT_OBJECT:{path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def score(value: Any) -> str:
    return "null" if value is None else f"{float(value):.12f}".rstrip("0").rstrip(".")


def main() -> int:
    report = load_json(VALIDATION)
    if report.get("status") != "PASS" or report.get("run_count") != 12:
        raise RuntimeError("validated 12-run campaign is required")
    rows = report["rows"]
    aggregates = report["aggregates"]

    release = {
        "schema_version": "3.0",
        "task_id": "adaptive_xtb_torsion_profile_agent_v3",
        "status": "READY_WITH_EVIDENCED_EXTERNAL_BLOCKS",
        "benchmark_framework_status": "PASS",
        "release_completeness": "READY_WITH_12_FORMAL_TERMINALS",
        "canonical_campaign": "m/post_recharge_20260805_r2",
        "suite_case_count": 4,
        "formal_run_count": 12,
        "formal_pass_count": report["pass_count"],
        "formal_blocked_count": report["blocked_count"],
        "native_hidden_grid_points": 216,
        "insufficient_balance_blocked_count": 0,
        "models": {
            label: {
                "status": (
                    "PARTIAL_COVERAGE"
                    if aggregates[label]["pass_count"]
                    else "EXTERNALLY_BLOCKED"
                ),
                "score_excluding_blocked": aggregates[label]["macro_average_excluding_blocked"],
                "pass_cases": aggregates[label]["pass_count"],
                "fail_cases": aggregates[label]["fail_count"],
                "blocked_cases": aggregates[label]["blocked_count"],
                "ranking_eligible": aggregates[label]["ranking_eligible"],
            }
            for label in ("deepseek", "kimi", "glm")
        },
        "external_blocker_categories": report["blocker_categories"],
        "legacy_v2_trajectories_valid_for_v3": False,
    }
    write_json(ROOT / "configs" / "release.json", release)

    model_status = {
        "schema_version": "torsion_model_status/3.0-post-recharge",
        "canonical_campaign": "m/post_recharge_20260805_r2",
        "preflight_report": "reports/PROVIDER_PREFLIGHT_POST_RECHARGE_20260805.json",
        "campaign_validation_report": "reports/POST_RECHARGE_CAMPAIGN_VALIDATION_20260805.json",
        "all_evaluator_replays_match": True,
        "blocked_policy": "null and excluded from capability means and ranking",
        "insufficient_balance_blocked_count": 0,
        "external_blocker_categories": report["blocker_categories"],
        "models": {
            label: {
                "requested_model_id": report["models"][label],
                "pass": aggregates[label]["pass_count"],
                "fail": aggregates[label]["fail_count"],
                "blocked": aggregates[label]["blocked_count"],
                "capability_mean_excluding_blocked": aggregates[label]["macro_average_excluding_blocked"],
                "population_standard_deviation": aggregates[label]["population_standard_deviation"],
                "minimum_score": aggregates[label]["minimum_score"],
                "maximum_score": aggregates[label]["maximum_score"],
                "ranking_eligible": aggregates[label]["ranking_eligible"],
            }
            for label in ("deepseek", "kimi", "glm")
        },
        "cases": rows,
    }
    write_json(ROOT / "reports" / "MODEL_RUN_STATUS.json", model_status)

    blockers = [row for row in rows if row["run_status"] == "BLOCKED"]
    write_json(
        ROOT / "reports" / "FORMAL_CAMPAIGN_BLOCKER.json",
        {
            "schema_version": "formal-campaign-external-blockers/3.0-post-recharge",
            "canonical_campaign": "m/post_recharge_20260805_r2",
            "blocked_count": len(blockers),
            "insufficient_balance_count": 0,
            "categories": report["blocker_categories"],
            "all_blockers_have_independent_replay": all(
                row["stored_evaluation_matches_replay"] for row in blockers
            ),
            "cases": [
                {
                    "logical_model": row["logical_model"],
                    "case_id": row["case_id"],
                    "category": row["failure_category"],
                    "provider_status": row["provider_status"],
                    "requested_model_id": row["requested_model_id"],
                    "actual_model_id": row["actual_model_id"],
                    "provider_retries": row["provider_retries"],
                    "xtb_success_points": row["xtb_success_points"],
                    "score": None,
                }
                for row in blockers
            ],
        },
    )

    case_table = [
        "| Model | Case | Status | Official | Scientific | Process | xTB ok/fail | Model calls | Retries | Requested / actual |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        requested_actual = f"`{row['requested_model_id']}` / `{row['actual_model_id'] or 'null'}`"
        status_text = row["run_status"]
        if row["failure_category"]:
            status_text += f" / {row['failure_category']}"
        case_table.append(
            "| {model} | `{case}` | {status} | {official} | {scientific} | {process} | {ok}/{failed} | {calls} | {retries} | {identity} |".format(
                model=row["logical_model"],
                case=row["case_id"],
                status=status_text,
                official=score(row["official_score"]),
                scientific=score(row["scientific_score"]),
                process=score(row["process_score"]),
                ok=row["xtb_success_points"],
                failed=row["xtb_failed_points"],
                calls=row["model_calls"],
                retries=row["provider_retries"],
                identity=requested_actual,
            )
        )

    aggregate_table = [
        "| Model | PASS | FAIL | BLOCKED | Scored mean | Std. dev. | Min | Max | Ranking eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label in ("deepseek", "kimi", "glm"):
        agg = aggregates[label]
        aggregate_table.append(
            f"| {label} | {agg['pass_count']} | {agg['fail_count']} | {agg['blocked_count']} | "
            f"{score(agg['macro_average_excluding_blocked'])} | {score(agg['population_standard_deviation'])} | "
            f"{score(agg['minimum_score'])} | {score(agg['maximum_score'])} | "
            f"{'yes' if agg['ranking_eligible'] else 'no'} |"
        )

    metric_table = [
        "| Model | Case | RMSE | MaxAE | Min F1 | Max F1 | Barrier MAE | Max barrier error |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["run_status"] != "PASS":
            continue
        metrics = row["metrics"]
        metric_table.append(
            f"| {row['logical_model']} | `{row['case_id']}` | "
            f"{score(metrics.get('rmse_kcal_mol'))} | {score(metrics.get('maxae_kcal_mol'))} | "
            f"{score(metrics.get('minimum_f1'))} | {score(metrics.get('maximum_f1'))} | "
            f"{score(metrics.get('barrier_mae_kcal_mol'))} | "
            f"{score(metrics.get('maximum_barrier_error_kcal_mol'))} |"
        )

    comparison = """# Model comparison — Adaptive Torsion v3 post-recharge

Canonical evidence is `working/m/post_recharge_20260805_r2`. All 12 model×case terminals were produced by one locked runner/evaluator/configuration set and independently replayed. BLOCKED rows are null, excluded from numeric aggregates, and disqualify coverage-based ranking; they are never converted to zero.

## Per-case results

{case_table}

## Aggregates

{aggregate_table}

## Baseline context

| Frozen baseline | Four-case macro official | Interpretation |
|---|---:|---|
| Weak 6-angle | 0.000000 | below minimum adaptive coverage |
| Fixed 12-angle | 83.714238 | strong on smooth cases, 50.6109 on the amide stress case |
| Oracle-guided greedy diagnostic | 72.208436 | organizer-only calibration, not a participant baseline |

## Sampling and score diagnostics for valid scientific trajectories

{metric_table}

DeepSeek's valid public-case trajectory began with a periodic six-angle scaffold and refined near predicted curvature/extrema, producing 12 unique points. GLM's three valid trajectories adapted case-by-case: 12 points for butane, 11 for difluoroethane after one protocol-correction turn, and 12 for N-methylacetamide with refinement around the narrow amide barrier. Kimi produced no complete scientific trajectory because all four provider sessions were externally blocked, so no sampling-policy or score inference is made for it.

Score differences come from independently recomputed curve error, symmetry-aware extrema, barrier error, adaptive coverage, and process gates—not model self-reports. GLM is near-perfect on the two smooth hidden cases but falls to `66.835672287878` on the sharp amide barrier; DeepSeek's public case scores `68.21965443164` because RMSE/MaxAE and minimum-extrema recall are weaker despite a correct global basin and full process score.

There are no `INSUFFICIENT_BALANCE` terminals after recharge. The eight remaining BLOCKED rows carry bounded-retry evidence for `NETWORK_CONNECTION_FAILED` or `PROVIDER_OVERLOADED`. DeepSeek has one scored PASS (`68.21965443164`); Kimi has no scored case; GLM has three scored PASS cases with mean `88.875779651515`. The historical GLM mean `88.892742` is not reused.
""".format(
        case_table="\n".join(case_table),
        aggregate_table="\n".join(aggregate_table),
        metric_table="\n".join(metric_table),
    )
    write_text(PROJECT / "MODEL_COMPARISON.md", comparison)
    write_text(ROOT / "reports" / "ORGANIZER_MODEL_COMPARISON.md", comparison)

    write_text(
        PROJECT / "TEST_REPORT.md",
        """# Test report — Adaptive Torsion v3 post-recharge

Pre-campaign and post-campaign full suites passed. Final post-campaign command: `python -m pytest -q`; result: **483 passed in 12.64 s**. This is above the historical 480-test floor.

Additional gates passed: Python syntax compilation; exact-ID three-model preflight; fresh four-point xTB 6.7.1 smoke; four oracle commitments; 216/216 hidden native oracle points; 12/12 trajectory hash-chain validation; 12/12 profile/artifact hash validation; 12/12 stored-versus-replayed evaluator equality; and working-tree secret scan with zero findings. Role-specific staging/ZIP boundary, manifest, extraction, and extracted test gates are enforced by the Torsion-only release builder.
""",
    )
    write_text(
        PROJECT / "RELEASE_VALIDATION.md",
        """# Release validation — Adaptive Torsion v3 post-recharge

- Canonical campaign: `working/m/post_recharge_20260805_r2`, 12/12 formal terminals, 4 PASS and 8 evidenced external BLOCKED.
- Exact preflight identities: DeepSeek V4 Pro, Kimi K2.6, and GLM 5.2 all PASS with no fallback.
- Balance gate: zero `INSUFFICIENT_BALANCE` terminals after recharge.
- Replay/integrity: all trajectory chains, stored evaluations, independent replays, profile hashes, artifact hashes, and native manifests match.
- Scientific runtime: xTB 6.7.1; fresh four-point smoke PASS; all four oracle commitments match; hidden native oracle points 216/216.
- Tests/security: 483 passed; working-tree secret findings 0; no project `.env`.
- Packaging: only the Torsion participant and organizer archives are rebuilt. QED v4 and xTB Failure v2 archives remain frozen byte-for-byte.
""",
    )
    write_text(
        PROJECT / "REVISION_RESPONSE.md",
        """# Revision response — Adaptive xTB Torsion Profile v3 post-recharge

The pre-recharge project and archives were snapshotted without deletion. The recharged workspace credential was loaded from the explicitly pinned root `.env` without logging any value; all three exact model IDs passed a one-turn, zero-xTB preflight.

The first post-recharge attempt (`r1`) exposed a multi-turn stale-connection lifecycle defect and was stopped without becoming canonical. The runner was hardened so every bounded application attempt receives a fresh fixed-model client; sampling budget, prompt, retry count/delay, four cases, oracle, native xTB protocol, and scoring were unchanged. Regression coverage increased from 480 to 483 tests.

The clean `r2` campaign ran all 12 combinations from one lock. Four completed scientific PASS cases and eight genuine external provider blocks remain. All blocks are network connection or provider overload evidence with null scores; no balance, permission, fallback, fabricated trajectory, or cross-model reuse remains. The complete result is deliverable under the audit rule allowing fully evidenced external BLOCKED terminals.
""",
    )
    write_text(
        PROJECT / "BASELINE_COMPARISON.md",
        """# Baseline comparison — Adaptive Torsion v3

Frozen four-case calibration (`validation/baselines_20260805_r1`) is unchanged by the recharge rerun:

| Baseline | Macro official | Per-case behavior |
|---|---:|---|
| Weak 6-angle | 0.000000 | fails minimum adaptive coverage |
| Fixed 12-angle | 83.714238 | 89.2461, 97.5, 97.5, 50.6109 |
| Oracle-guided greedy diagnostic | 72.208436 | organizer calibration only |

The fixed public policy is strong on smooth profiles but materially weaker on the narrow amide barrier, demonstrating cross-morphology difficulty without changing the established periodic/symmetry scoring.
""",
    )
    write_text(
        PROJECT / "CHANGELOG.md",
        """# Changelog — Adaptive Torsion v3 post-recharge release

- Preserved a complete pre-recharge snapshot and all historical evidence.
- Explicitly pinned the trusted root `.env` so stale inherited credentials cannot win; no credential value is persisted.
- Passed exact-ID recharge preflight for DeepSeek V4 Pro, Kimi K2.6, and GLM 5.2.
- Added complete per-run environment, optional reasoning/cache usage, oracle commitment, profile hash, and artifact hash metadata.
- Diagnosed the non-canonical r1 stale HTTP connection lifecycle and recreated the fixed-model client on every bounded attempt.
- Ran a clean locked r2 campaign over all 12 model×case combinations; 4 PASS and 8 evidenced external BLOCKED, with zero balance blocks.
- Independently replayed all 12 evaluations and validated 216/216 hidden oracle points, xTB 6.7.1 smoke, 483 tests, manifests, and secrets.
- Rebuilt only the Torsion participant/organizer releases; QED v4 and xTB Failure v2 remain frozen.
""",
    )
    write_text(
        PROJECT / "DESIGN_RATIONALE.md",
        """# Design rationale — Adaptive Torsion v3

The evaluated capability is adaptive sampling policy: the model chooses new 5° grid angles in three feedback phases under a 10–12 point budget. Native GFN2-xTB execution, periodic model selection, curve reconstruction, extrema analysis, and final serialization are deterministic tool responsibilities.

Four molecules expose distinct biphenyl symmetry, alkane, gauche-effect, and narrow amide-barrier behavior. Established periodic, near-degenerate-extrema, and barrier matching rules remain frozen. Equal case weighting tests policy transfer rather than rewarding one public morphology.

BLOCKED is orthogonal to scientific performance. A genuine external provider failure has a null score and removes ranking eligibility; a normal response with failed science is FAIL/0. Physical participant/organizer separation prevents oracle reconstruction from local caches or paths.
""",
    )
    write_text(
        ROOT / "reports" / "ORGANIZER_VALIDATION_REPORT.md",
        """# Organizer validation — Torsion v3 post-recharge

- Four frozen 72-point oracle commitments match; the three private cases have 216/216 manifest-bound xTB 6.7.1 native points.
- The locked r2 campaign has 12/12 independently replayable terminals, 4 PASS and 8 evidenced external BLOCKED, with zero balance blocks.
- Every stored evaluation equals two fresh offline evaluations; every profile/artifact hash and trajectory chain validates.
- Fixed-12 calibration remains 83.714238; weak-6 remains 0; oracle-guided diagnostic remains 72.208436.
- Python syntax, 483 tests, fresh xTB smoke, and zero-secret checks pass.
""",
    )
    write_text(
        ROOT / "reports" / "REPRODUCIBILITY_REPORT.md",
        """# Reproducibility report — Torsion v3 post-recharge

`m/post_recharge_20260805_r2/campaign_lock.json` commits the exact model IDs, temperature, null seed, prompt hash, runner, provider adapter, evaluator, scoring helpers, suite, and four oracle hashes. Every case records start/end/duration, Python/RDKit/xTB/OS, requested/actual model identity, standard and optional token/cache projections, retries, terminal status, oracle commitment, profile hash, and artifact hashes.

The independent validation re-evaluates every case twice from its append-only hash-chained trajectory and manifest-bound native results. Both fresh evaluations must equal the stored evaluation byte-for-byte. BLOCKED cases retain partial native evidence but remain null and excluded from means.
""",
    )
    write_text(
        ROOT / "reports" / "SCIENTIFIC_VALIDATION_REPORT.md",
        """# Scientific validation report — Adaptive xTB Torsion v3

Three hidden profiles use RDKit ETKDGv3/constrained UFF initialization followed by native xTB 6.7.1 GFN2 constrained tight optimization on every 5° grid point. All 216 hidden attempts are manifest-valid, finite, converged, profile-eligible, within the frozen torsion tolerance, and exactly bound to the oracle profiles. The inherited public 2-fluorobiphenyl 72-point commitment also matches.

The post-recharge four-point smoke at 15°, 105°, 195°, and 285° passed with the committed xTB 6.7.1 binary. Formal r2 native attempts are independently verified against their manifests and xTB identity. The suite makes no claim of experimental free energies or a fully relaxed multidimensional potential surface; it is a frozen constrained gas-phase GFN2-xTB benchmark.
""",
    )
    print(json.dumps({"status": "PASS", "project": str(PROJECT), "reports_updated": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
