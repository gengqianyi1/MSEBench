#!/bin/sh
set -eu
export CHEMBENCH_ORGANIZER_EVIDENCE=/workspace/cases/case-1/case_result.json
if [ ! -f "$CHEMBENCH_ORGANIZER_EVIDENCE" ]; then
  echo "missing declared verifier artifact: case_result.json" >&2
fi
exec python /tests/protocol_grader.py
