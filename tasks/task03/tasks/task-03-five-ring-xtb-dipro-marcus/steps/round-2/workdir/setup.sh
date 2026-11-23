#!/bin/bash
set -euo pipefail
# Round feedback is produced by the private sidecar after the Agent writes its
# candidate.  There is intentionally no public evaluator or state file here.
mkdir -p /workspace/rounds
