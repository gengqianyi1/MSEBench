#!/bin/bash
set -euo pipefail
python /tests/final_grader.py
python /tests/run_legacy_grader.py
