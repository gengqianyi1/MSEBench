#!/bin/bash
set -uo pipefail
if python - <<'PY'
import json
from pathlib import Path
p=Path("/workspace/rounds/round-5/candidate.json")
v=json.loads(p.read_text()) if p.is_file() else {}
c=v.get("candidate") if isinstance(v,dict) else None
required=['candidate_id', 'parent_candidate_id', 'proposal_mode', 'smiles', 'structural_modification', 'modification_reasoning']
if not isinstance(c,dict) or any(k not in c for k in required) or c.get("candidate_id")!="R5":
    raise SystemExit("invalid round candidate")
PY
then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
