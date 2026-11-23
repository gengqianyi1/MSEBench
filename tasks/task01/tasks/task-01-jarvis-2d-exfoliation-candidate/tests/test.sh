#!/bin/bash
set -euo pipefail
python /tests/test_llm_request_accounting.py
python /tests/run_legacy_grader.py
