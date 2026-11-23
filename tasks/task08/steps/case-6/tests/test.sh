#!/bin/bash
set -uo pipefail
/tests/final_test.sh
test -f /logs/verifier/reward.json -o -f /logs/verifier/reward.txt || echo 0 > /logs/verifier/reward.txt
