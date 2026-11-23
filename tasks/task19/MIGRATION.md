# Task 19 Harbor migration record

- Source execution mode: `not_applicable_new_task`
- Legacy native runner: none; this is a new Harbor-native task.
- Harbor protocol: one protected case, at most five Agent turns, four experiment calls, then one terminal submission.
- Private boundary: experiment responses, correct identity, scoring features, utility calibration and Oracle remain inside the verifier snapshot.
- Grading: `protocol_grader.py` reconstructs measurements and scoring only from organizer evidence.
- Parity definition: exact equality between the author direct protocol driver and Harbor sidecar for the same requests, submission, evidence and score.
- Current real-model status: `REAL_MODEL_EVALUATION_PENDING`.

No legacy parity claim is made because no legacy runner exists. Release gates remain false until container build, leakage scan and all three required model runs are completed.
