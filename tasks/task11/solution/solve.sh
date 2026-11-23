#!/bin/bash
set -euo pipefail
cat > /workspace/final_answer.json <<'JSON'
{"smiles":"COCCOC(=O)C(F)OC","name":"deterministic lifecycle oracle","design_rationale":"A public-contract fixture satisfying the frozen descriptor constraints."}
JSON
