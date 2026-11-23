#!/bin/bash
set -euo pipefail
python /workspace/chembench_protocol/evaluate_round.py \
  --round 4 \
  --candidate /workspace/rounds/round-3/candidate.json \
  --output /workspace/rounds/round-3/feedback.json \
  --state /workspace/rounds/public_protocol_state.json \
  --task-config /workspace/input/tasks/q1_lipophilic/task_config.yaml
