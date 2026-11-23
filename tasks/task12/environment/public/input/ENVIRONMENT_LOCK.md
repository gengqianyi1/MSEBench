# Frozen runtime and replay evidence

Canonical runtime: Windows x86-64, Python 3.9.24, PyBaMM 25.4.2, NumPy 1.26.4, SciPy 1.13.1, CasADi 3.6.7, PyYAML 6.0.2. Telemetry is disabled; CasADi uses safe mode with `rtol=1e-7` and `atol=1e-7`. There is no scored solver fallback. All four experiment steps must complete; partial solutions are invalid.

The private reference candidate is `cc=4.0 C`, `cc/cv=4.15 V`, `cv cutoff=0.25 C`, negative/positive thickness `110/100 um`, porosity `0.32`, and particle radius `8.8 um`. At the canonical `1e-7` tolerances it completes all four steps and scores 100: charge 74.3760 min, discharge capacity 0.3258 Ah, voltage 2.5000-4.1500 V, and minimum negative-particle surface concentration 1402.1399 mol m-3.

The archived DeepSeek submission is a different flat candidate (`2.0 C`, `4.10/4.10 V`, `0.15 C`, `100/90 um`, `0.35`, `4.5 um`). The envelope-compatible grader correctly evaluates it scientifically instead of returning `INVALID_CANDIDATE`; its replay is recorded separately and it is not the reference candidate.

Sensitivity evidence is recorded in `revision_records/12_pybamm_solver_sensitivity.md`. At looser tolerances the same candidate either crosses a voltage gate or returns an incomplete/unstable solution; therefore the exact solver and tolerances are part of the scientific contract, and incomplete trajectories are now rejected. Any threshold, model, parameter, platform, or solver-setting change requires a new task version and full scientific rerun.

## Revision4 clean-room reference (2026-08-31)

The Harbor agent/verifier lockfiles are authoritative for replay:
`environment/environment.lock.yml` and `tests/environment.lock.yml`, both SHA-256
`03704a4b9218ccce46b23f54ef947bdd140233511d4054f3d4a4166f6ef0f4c8`.
The offline runtime matched Python 3.9.24, PyBaMM 25.4.2, NumPy 1.26.4,
SciPy 1.13.1 and Casadi 3.6.7. See the Harbor task's `CLEAN_ROOM_REPLAY.md`
and `tests/environment_fingerprint_20260831.json`. A Linux Docker image digest
and live same-model Harbor parity remain pending and are not claimed.
