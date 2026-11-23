from __future__ import annotations
import importlib.util,json,sys
from pathlib import Path

TARGET='CC(=O)N(CCOC(=O)c1cc(-c2ccccc2)cc(-c2ccc(C)cc2)c1)Cc1ccccc1'
INVENTORY=['COC(=O)c1cc(Br)cc(Br)c1','Cc1ccc(B(O)O)cc1','OB(O)c1ccccc1','OCCBr','NCc1ccccc1','CC(=O)Cl','COC(=O)c1ccc(Br)cc1','COC(=O)c1cccc(Br)c1','COC(=O)c1ccccc1','COC(=O)c1ccc(Br)c(Br)c1','COC(=O)c1c(Br)cccc1Br','Cc1cccc(B(O)O)c1','Cc1ccccc1B(O)O','COc1ccc(B(O)O)cc1','Fc1ccc(B(O)O)cc1','Clc1ccc(B(O)O)cc1','Nc1ccccc1','NCCc1ccccc1','CNc1ccccc1','ClCc1ccccc1','BrCCBr','CI','CCO','CC(=O)Br','O=C(O)c1ccccc1']
CONSTRAINTS={'max_steps':{'min':2,'max':6},'atom_economy':{'min':0.50}}
SCORING={'total_points':100,'expected_route_length':6,'step_progress_points':35,'route_validity_points':25,'target_reached_points':40}

def _modules():
    core=Path(__file__).parent/'core'; sys.path.insert(0,str(core))
    def load(name):
        spec=importlib.util.spec_from_file_location(name,core/f'{name}.py'); module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module); return module
    chemistry=load('chemistry'); evaluator=load('evaluator'); scoring=load('scoring'); return evaluator,scoring

def grade(context):
    try:
        # 2026-09-27 fix: prefer canonical final_answer.json over route.json.
        # Old behavior (kept for audit): route.json was read first, so a
        # stale/draft route.json silently overrode the canonical
        # final_answer.json whenever the two files disagreed, contradicting
        # instruction.md ("the Harness saves canonical final_answer.json";
        # route.json is an optional audit artifact). New behavior:
        # final_answer.json is authoritative; route.json is consulted only as
        # a fallback when final_answer.json is absent.
        path=context.submission_dir/'final_answer.json'
        if not path.is_file(): path=context.submission_dir/'route.json'
        payload=json.loads(path.read_text(encoding='utf-8')); route=payload['route']
    except Exception:
        return {'score':0.0,'subscores':{},'failure_codes':['INVALID_ROUTE_JSON'],'evidence':{}}
    evaluator,scoring=_modules(); evaluated=evaluator.evaluate_route(route,TARGET,INVENTORY,CONSTRAINTS); scored=scoring.score_single_route(evaluated,SCORING)
    flags=evaluator.route_constraint_flags(evaluated); failures=[key for key,value in flags.items() if not value]
    if not evaluated.final_equals_target: failures.append('final_equals_target')
    return {'score':float(scored['total_route_score']),'subscores':{'verified_step_progress':scored['step_progress_points'],'process_validity':scored['route_validity_points'],'exact_target':scored['target_reached_points']},'failure_codes':failures,'evidence':{'n_steps':evaluated.n_steps,'process_flags':flags,'final_equals_target':evaluated.final_equals_target,'per_step':[step.__dict__ for step in evaluated.step_evals],'equivalence_contract':'RDKit canonical isomeric SMILES; salts remain separate fragments; no implicit heavy-atom reagents; declared reactants supply every product heavy atom'}}
