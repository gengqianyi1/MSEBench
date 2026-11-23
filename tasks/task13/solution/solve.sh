#!/bin/bash
set -euo pipefail
cat > /workspace/final_answer.json <<'JSON'
{"route":[
{"reaction":"suzuki_coupling","reactants":["COC(=O)c1cc(Br)cc(Br)c1","Cc1ccc(B(O)O)cc1"],"product":"COC(=O)c1cc(Br)cc(-c2ccc(C)cc2)c1"},
{"reaction":"suzuki_coupling","reactants":["COC(=O)c1cc(Br)cc(-c2ccc(C)cc2)c1","OB(O)c1ccccc1"],"product":"COC(=O)c1cc(-c2ccccc2)cc(-c2ccc(C)cc2)c1"},
{"reaction":"ester_hydrolysis","reactants":["COC(=O)c1cc(-c2ccccc2)cc(-c2ccc(C)cc2)c1"],"product":"Cc1ccc(-c2cc(C(=O)O)cc(-c3ccccc3)c2)cc1"},
{"reaction":"fischer_esterification","reactants":["Cc1ccc(-c2cc(C(=O)O)cc(-c3ccccc3)c2)cc1","OCCBr"],"product":"Cc1ccc(-c2cc(C(=O)OCCBr)cc(-c3ccccc3)c2)cc1"},
{"reaction":"sn2_n_alkylation","reactants":["NCc1ccccc1","Cc1ccc(-c2cc(C(=O)OCCBr)cc(-c3ccccc3)c2)cc1"],"product":"Cc1ccc(-c2cc(C(=O)OCCNCc3ccccc3)cc(-c3ccccc3)c2)cc1"},
{"reaction":"n_acylation","reactants":["Cc1ccc(-c2cc(C(=O)OCCNCc3ccccc3)cc(-c3ccccc3)c2)cc1","CC(=O)Cl"],"product":"CC(=O)N(CCOC(=O)c1cc(-c2ccccc2)cc(-c2ccc(C)cc2)c1)Cc1ccccc1"}
]}
JSON
