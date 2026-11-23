#!/bin/bash
set -euo pipefail
python - <<'PY'
import json
from pathlib import Path
p=Path("/workspace/rounds/round-3/candidate.json"); v=json.loads(p.read_text()) if p.is_file() else {}; c=v.get("candidate") if isinstance(v,dict) else None
if not isinstance(c,dict) or not isinstance(c.get("formulation"),dict): raise SystemExit("invalid candidate")
PY
echo 1 > /logs/verifier/reward.txt
