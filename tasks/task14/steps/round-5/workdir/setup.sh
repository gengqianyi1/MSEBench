#!/bin/bash
set -euo pipefail
python /workspace/chembench_protocol/evaluate_round.py --round 5 --candidate /workspace/rounds/round-4/candidate.json --output /workspace/rounds/round-4/feedback.json --task-config /workspace/input/tasks/q1_multiobjective/task_config.yaml --state /workspace/rounds/best_so_far.json
