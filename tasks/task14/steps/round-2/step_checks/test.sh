#!/bin/bash
set -uo pipefail
if python - <<'PY'
import json, sys
from pathlib import Path
p = Path("/workspace/rounds/round-2/candidate.json")
if not p.is_file():
    print("MISSING candidate.json"); sys.exit(1)
d = json.loads(p.read_text())
c = d.get("candidate", d)
if not isinstance(c, dict) or not str(c.get("smiles", "")).strip():
    print("UNPARSEABLE candidate"); sys.exit(1)
print("protocol step complete")
PY
then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
    exit 1
fi
