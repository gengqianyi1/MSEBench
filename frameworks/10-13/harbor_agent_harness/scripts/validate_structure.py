#!/usr/bin/env python3
"""Static structure validator: single-step and multi-step Harbor tasks.

Default root is `tasks`. For each task:
  - if `task.toml` has `[[steps]]` (multi-step), validate the multi-step layout;
  - otherwise validate the single-step layout.
"""
from pathlib import Path
import json, sys, tomllib, yaml

try:
    import jsonschema
    _SCHEMA_PATH = Path(__file__).resolve().parents[1] / 'schemas' / 'harness-contract.schema.json'
    _CONTRACT_SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding='utf-8'))
    _HAS_JSONSCHEMA = True
except ImportError:
    jsonschema = None
    _CONTRACT_SCHEMA = None
    _HAS_JSONSCHEMA = False

root = Path(sys.argv[1] if len(sys.argv) > 1 else 'tasks')
errs, tasks = [], []

for d in sorted(root.glob('task-*')):
    tasks.append(d.name)
    # common
    for f in ['task.toml', 'environment/Dockerfile', 'environment/environment.yaml', 'environment/environment.lock.yml']:
        if not (d / f).exists():
            errs.append(f'{d.name}: missing {f}')
    if (d / 'native_runner.py').exists() or list(d.rglob('native_runner.py')):
        errs.append(f'{d.name}: native_runner.py leaked into generated task')
    for sh in d.rglob('*.sh'):
        if b'\r\n' in sh.read_bytes():
            errs.append(f'{d.name}: CRLF in {sh.relative_to(d)}')
    try:
        x = tomllib.loads((d / 'task.toml').read_text())
    except Exception as e:
        errs.append(f'{d.name}: invalid TOML: {e}')
        continue
    if x.get('schema_version') != '1.4':
        errs.append(f'{d.name}: schema_version != 1.4')
    agent_cfg = d / 'environment' / 'public' / '.chembench_agent.json'
    if agent_cfg.is_file():
        try:
            protocol_mode = json.loads(agent_cfg.read_text()).get('protocol_mode')
        except Exception as e:
            errs.append(f'{d.name}: invalid .chembench_agent.json: {e}')
            protocol_mode = None
        if protocol_mode == 'private_response_round':
            deployment = d / 'environment' / 'public' / 'sidecar_deployment.yaml'
            if not deployment.is_file():
                errs.append(f'{d.name}: private-response protocol missing sidecar_deployment.yaml')
            else:
                spec = yaml.safe_load(deployment.read_text()) or {}
                required = set((spec.get('ownership') or {}).get('author_required_before_release') or [])
                if not {'private_image_digest', 'disclosure_justification', 'parity_evidence'} <= required:
                    errs.append(f'{d.name}: sidecar deployment lacks author-required release fields')

    steps = x.get('steps')
    if steps:
        # ---- multi-step ----
        if not x.get('multi_step_reward_strategy'):
            errs.append(f'{d.name}: multi-step missing multi_step_reward_strategy')
        names = [s.get('name') for s in steps]
        if len(names) != len(set(names)):
            errs.append(f'{d.name}: step names not unique')
        root_tests = (d / 'tests' / 'test.sh').exists()
        oracle_status = (x.get('metadata') or {}).get('oracle_status', 'missing')
        for s in steps:
            name = s.get('name')
            sd = d / 'steps' / name
            if oracle_status == 'verified_harbor':
                sol = sd / 'solution' / 'solve.sh'
                if not sol.exists():
                    errs.append(f'{d.name}: step {name} missing solution/solve.sh for oracle_status={oracle_status}')
            if not (sd / 'instruction.md').exists():
                errs.append(f'{d.name}: step {name} missing instruction.md')
            step_ts = (sd / 'tests' / 'test.sh') if (sd / 'tests' / 'test.sh').exists() else (d / 'tests' / 'test.sh')
            if not ((sd / 'tests' / 'test.sh').exists() or root_tests):
                errs.append(f'{d.name}: step {name} has no test.sh and no root fallback')
            elif step_ts.exists():
                txt = step_ts.read_text(encoding='utf-8')
                if 'reward' not in txt and 'run_legacy_grader' not in txt:
                    errs.append(f'{d.name}: step {name} verifier does not write reward file')
    else:
        # ---- single-step ----
        for f in ['instruction.md', 'tests/Dockerfile', 'tests/test.sh']:
            if not (d / f).exists():
                errs.append(f'{d.name}: missing {f}')
        if (x.get('verifier') or {}).get('environment_mode') != 'separate':
            errs.append(f'{d.name}: verifier.environment_mode != separate')

    try:
        yaml.safe_load((d / 'environment/environment.yaml').read_text())
        yaml.safe_load((d / 'environment/environment.lock.yml').read_text())
    except Exception as e:
        errs.append(f'{d.name}: invalid YAML: {e}')

    contract = d / 'harness_contract.yaml'
    if _HAS_JSONSCHEMA and contract.is_file():
        try:
            data = yaml.safe_load(contract.read_text(encoding='utf-8'))
            jsonschema.validate(data, _CONTRACT_SCHEMA)
        except Exception as e:
            errs.append(f'{d.name}: harness_contract.yaml fails schema: {e}')

report = {'dataset': str(root), 'task_count': len(tasks), 'errors': errs, 'status': 'PASS' if not errs else 'FAIL'}
print(json.dumps(report, indent=2))
sys.exit(1 if errs else 0)
