#!/bin/bash
set -euo pipefail
python - <<'PY'
import json
from pathlib import Path
p=Path("/workspace/rounds/round-2/candidate.json")
value=json.loads(p.read_text()) if p.is_file() else None
candidate=value.get("candidate", value) if isinstance(value, dict) else None
if not isinstance(candidate, dict) or not str(candidate.get("smiles", "")).strip():
    raise SystemExit("missing or invalid candidate SMILES")
PY
echo 1 > /logs/verifier/reward.txt
