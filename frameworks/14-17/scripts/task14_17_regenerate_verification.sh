#!/usr/bin/env bash
# Rebuild every Docker-free verification output into verification/, from the package as it stands.
# Run from the package root. Anything needing Docker (Harbor E2E, sidecar wiring) is not produced
# here on purpose -- see verification/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p verification
{ echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo; echo "\$ python scripts/task14_17_oracle_and_invalid_check.py"
  python scripts/task14_17_oracle_and_invalid_check.py; echo
  echo "\$ python scripts/task14_17_oracle_and_invalid_check.py --self-test"
  python scripts/task14_17_oracle_and_invalid_check.py --self-test
} > verification/oracle_and_invalid.txt 2>&1
{ echo "\$ python scripts/task14_17_boundary_check.py"; python scripts/task14_17_boundary_check.py; echo
  echo "\$ python scripts/task14_17_boundary_check.py --self-test"
  python scripts/task14_17_boundary_check.py --self-test
} > verification/boundary.txt 2>&1
{ echo "\$ python scripts/task14_17_shortcut_baseline_report.py"
  python scripts/task14_17_shortcut_baseline_report.py
} > verification/shortcut_baselines.txt 2>&1
{ echo "\$ python scripts/task14_17_package_self_contained_check.py"
  python scripts/task14_17_package_self_contained_check.py; echo
  echo "\$ python scripts/task14_17_package_self_contained_check.py --self-test"
  python scripts/task14_17_package_self_contained_check.py --self-test
} > verification/package_self_contained.txt 2>&1
echo "verification/ rebuilt:"; ls -1 verification/
