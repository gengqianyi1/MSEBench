#!/bin/bash
set -euo pipefail
cat > /workspace/final_answer.json <<'JSON'
{"candidate":{"cc_c_rate":2.0,"cc_until_voltage_V":4.1,"cv_voltage_V":4.1,"cv_until_current_C":0.15,"negative_electrode_thickness_um":100,"positive_electrode_thickness_um":90,"electrode_porosity":0.35,"particle_radius_um":4.5},"name":"public-envelope lifecycle oracle","rationale":"A deterministic in-range protocol that must be independently simulated by the verifier."}
JSON
