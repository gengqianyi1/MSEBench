#!/bin/bash
set -uo pipefail
if test -f /workspace/cases/case-5/case_result.json; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
