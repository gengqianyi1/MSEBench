# Task 20 Harbor migration record

- Source execution mode: `not_applicable_new_task`.
- Legacy native runner: none; this is a new Harbor-native task.
- Harbor protocol: one protected case, at most five Agent turns, four diagnostic calls, then one terminal submission.
- Private boundary: mixed-mechanism truth, frozen diagnostic responses, regeneration calibration and Oracle remain inside the verifier snapshot.
- Grading: organizer-evidence replay only.
- Parity definition: exact equality between the author direct protocol driver and Harbor sidecar for identical requests and submission.
- Current real-model status: `REAL_MODEL_EVALUATION_PENDING`.

No legacy parity is claimed. Release gates stay false until the private image, leakage scan and three required model runs are complete.

## Author revision 2026-09-10

- Scientific grader, case, budgets and Oracle are unchanged.
- sidecar release_ready is false, consistent with task/contract release_eligible.
- Historical local image IDs are explicitly not deployable registry references.
- Organizer deployment uses scripts/isolated_sidecar.py: Docker network none,
  no published ports, fixed docker-exec relay bound only to organizer loopback.
  No shared Agent or Harbor core modifications are introduced by this revision.
- Native HTTP -> attested evidence -> private grader CLI regression is available
  in release_evidence/review_20260910, including five legal scientific errors.
- Docker relay isolation, Task20-specific Harbor E2E and fresh three-model
  evaluation remain NOT RUN because the Docker engine is unavailable.
- This is an author return for integration, not an unconditional release.
