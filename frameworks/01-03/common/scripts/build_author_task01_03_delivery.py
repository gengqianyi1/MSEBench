"""Freeze organizer reference submissions and replay with current task sources.

No scientific approval is inferred from a packaging or verifier pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import os
OUT = ROOT / os.environ.get('AUTHOR_DELIVERY_OUT', 'author_delivery_task01_03_20260905')
CONFIG = {
    'task01': ('task-01-jarvis-2d-exfoliation-candidate', 'task01-author-chgnet-refinement-20260905', 'chembench/std-task01-verifier:2.1.0'),
    'task02': ('task-02-oligothiophene-smd-sapt', 'task02-deepseek-v4-flash-20260905', 'chembench/std-task02-verifier:2.1.2'),
    'task03': ('task-03-five-ring-xtb-dipro-marcus', None, 'chembench/std-task03-verifier:2.1.0'),
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def prepare(key):
    name, candidate, image = CONFIG[key]
    task, bundle = ROOT / 'tasks' / name, OUT / key
    if bundle.exists():
        raise FileExistsError(bundle)
    spec = tomllib.loads((task / 'task.toml').read_text(encoding='utf-8'))
    inspected = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image], text=True))[0]
    bundle.mkdir(parents=True)
    write(bundle / 'provenance/image_inspect.json', inspected)
    # Archive exact task implementation independently of the immutable base image.
    shutil.copytree(task / 'tests', bundle / 'runtime/tests', ignore=shutil.ignore_patterns('__pycache__', 'oracle_bundle'))
    shutil.copyfile(task / 'task.toml', bundle / 'runtime/task.toml')
    if candidate:
        source = ROOT / 'candidate_oracle_bundles' / candidate
        shutil.copytree(source / 'workspace', bundle / 'workspace')
        shutil.copytree(source / 'verifier', bundle / 'provenance/original_verifier')
        shutil.copyfile(source / 'manifest.json', bundle / 'provenance/original_manifest.json')
        for file in source.glob('*llm_call_summary.json'):
            shutil.copyfile(file, bundle / 'provenance' / file.name)
        score, reward = read(source / 'manifest.json')['score'], read(source / 'manifest.json')['reward']
        if key == 'task01':
            refine = ROOT / 'scripts/refine_task01_reference.py'
            command = ['docker', 'run', '--rm', '--network', 'none', '--user', '0:0',
                '--mount', f'type=bind,source={(bundle / "workspace").resolve().as_posix()},target=/workspace',
                '--mount', f'type=bind,source={(task / "environment/public/input/original").resolve().as_posix()},target=/public_original,readonly',
                '--mount', f'type=bind,source={refine.resolve().as_posix()},target=/refine.py,readonly',
                '--entrypoint', '/opt/conda/envs/chembench/bin/python', image, '/refine.py']
            subprocess.run(command, check=True)
            write(bundle / 'provenance/public_artifact_refinement_command.json', command)
            # The source generator wrote input_files as a path->digest mapping,
            # while the frozen scoring contract requires a list of records.
            generation_manifest = bundle / 'workspace/generator/generation_manifest.json'
            gm = read(generation_manifest)
            if isinstance(gm.get('input_files'), dict):
                gm['input_files'] = [
                    {'path': path, 'sha256': digest}
                    for path, digest in sorted(gm['input_files'].items())
                ]
                write(generation_manifest, gm)
        if key == 'task02':
            # The source run recorded this endpoint under the ALPB optimization log;
            # materialize the declared selected endpoint path for replay lineage.
            endpoint = bundle / 'workspace/structures/endpoint_n_hexane_selected.xyz'
            shutil.copyfile(bundle / 'workspace/logs/alpb/n_hexane_start0/xtbopt.xyz', endpoint)
            fragments = bundle / 'workspace/structures/fragments'
            if not fragments.exists() or not any(fragments.iterdir()):
                run_fragments = ROOT / 'candidate_oracle_runs/task02-deepseek-v4-flash-20260905/evidence-task02'
                matches = [p for p in run_fragments.rglob('workspace/structures/fragments') if any(p.iterdir())]
                if matches:
                    shutil.copytree(matches[0], fragments, dirs_exist_ok=True)
            # The candidate-export layer omitted two small Stage-A evidence files.
            # Reconstruct the mapping from the validated SDF atom order (0-based):
            # S4-C0-C8-S9 is the inter-ring torsion and C3/C5 are the terminal
            # alpha carbons.  This restores the evidence declared by stage_A.
            analysis = bundle / 'workspace/analysis'
            analysis.mkdir(parents=True, exist_ok=True)
            write(analysis / 'atom_mapping.json', {
                'index_base': 0,
                'source_structure': 'structures/solute_validated.sdf',
                'inter_repeat_bond': [0, 8],
                'inter_repeat_dihedral': [4, 0, 8, 9],
                'sulfur_atoms': [4, 9],
                'terminal_carbons': [3, 5],
            })
            accepted = {}
            for rel in ('instance.json', 'protocol.yaml', 'structures/solute_validated.sdf',
                        'structures/solute_validated.xyz'):
                p = bundle / 'workspace' / rel
                if p.is_file():
                    accepted[rel] = {'sha256': sha(p)}
            write(analysis / 'accepted_input_hashes.json', accepted)
    else:
        for step in sorted((task / 'steps').iterdir()):
            if step.name.startswith('round-'):
                dest = bundle / 'workspace/rounds' / step.name
                dest.mkdir(parents=True)
                shutil.copyfile(step / 'solution/candidate.json', dest / 'candidate.json')
        shutil.copyfile(task / 'steps/recommendation/solution/final_answer.json', bundle / 'workspace/final_answer.json')
        shutil.copytree(task / 'steps', bundle / 'oracle_submission/steps')
        score = reward = None
    declared = task / 'environment/public/sidecar_deployment.yaml'
    if declared.exists():
        shutil.copyfile(declared, bundle / 'provenance/original_sidecar_deployment.yaml')
    manifest = {
        'schema_version': 'chembench-author-reference-bundle/1.0',
        'status': 'PREPARED_PENDING_CURRENT_SOURCE_REPLAY',
        'task_name': spec['task']['name'], 'task_version': spec['task']['version'],
        'reference_kind': 'scored_reference_submission' if candidate else 'frozen_oracle_submission',
        'sidecar_image_digest': inspected['RepoDigests'][0].split('@')[-1] if inspected.get('RepoDigests') else inspected['Id'],
        'image_id': inspected['Id'], 'image_tag': image,
        'source_candidate': candidate, 'score': score, 'reward': reward,
        'scientific_approval': 'PENDING_AUTHOR_REVIEW',
        'production_release_approval': 'PENDING_AUTHOR_REVIEW',
        'required_artifacts': spec['metadata']['required_artifacts'],
        'runtime_file_hashes': {p.relative_to(bundle / 'runtime').as_posix(): sha(p) for p in (bundle / 'runtime').rglob('*') if p.is_file()},
    }
    write(bundle / 'manifest.json', manifest)
    print(json.dumps({'prepared': key, 'image_id': inspected['Id']}), flush=True)


def replay(key, label='replay'):
    bundle = OUT / key
    manifest = read(bundle / 'manifest.json')
    logs = bundle / f'{label}_logs'
    if logs.exists():
        raise FileExistsError('Replay is append-only; preserve existing evidence: ' + str(logs))
    logs.mkdir()
    work = bundle / f'{label}_workspace'
    shutil.copytree(bundle / 'workspace', work)
    command = ['docker', 'run', '--rm', '--user', '0:0', '--network', 'none', '--cpus', '4',
               '-e', 'OMP_NUM_THREADS=1', '-e', 'MKL_NUM_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1']
    for source, target, readonly in (
        (work, '/workspace', False), (logs, '/logs', False), (bundle / f'{label}_run', '/run', False),
        (bundle / 'runtime/tests', '/tests', True),
        (bundle / 'runtime/tests/task_snapshot', '/task', True),
    ):
        source.mkdir(parents=True, exist_ok=True)
        command += ['--mount', f'type=bind,source={source.resolve().as_posix()},target={target}' + (',readonly' if readonly else '')]
    programs = ['/tests/final_grader.py', '/tests/run_legacy_grader.py'] if key == 'task03' else ['/tests/run_legacy_grader.py']
    # A constant Python program sequences task-owned entrypoints, without shell interpolation.
    runner = ('import runpy\n' + f'for program in {programs!r}:\n'
              '    try:\n        runpy.run_path(program, run_name="__main__")\n'
              '    except SystemExit as exc:\n        if exc.code not in (None, 0): raise\n')
    command += [manifest['image_id'], 'micromamba', 'run', '-n', 'chembench', 'python', '-c', runner]
    write(bundle / f'provenance/{label}_command.json', command)
    with (logs / 'console.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    write(logs / 'process.json', {'returncode': result.returncode, 'current_source_mounted': True, 'network': 'none'})
    if result.returncode:
        raise RuntimeError(f'{key} replay failed; see {logs}')
    details, reward = read(logs / 'verifier/details.json'), read(logs / 'verifier/reward.json')
    shutil.copytree(logs / 'verifier', bundle / 'verifier', dirs_exist_ok=True)
    if key == 'task03':
        # final_grader generated these values by xTB/DIPRO recomputation.
        shutil.copyfile(work / 'result.json', bundle / 'workspace/result.json')
        shutil.copytree(work / 'verifier_compute', bundle / f'{label}_computation_evidence')
    manifest.update(status='REPLAY_VERIFIED_PENDING_AUTHOR_REVIEW',
                    score=details.get('scientific_score', details['legacy_result']['score']),
                    reward=reward['reward'], failure_codes=details['legacy_result'].get('failure_codes', []),
                    artifacts={p.relative_to(bundle / 'workspace').as_posix(): sha(p) for p in (bundle / 'workspace').rglob('*') if p.is_file()})
    write(bundle / 'manifest.json', manifest)
    print(json.dumps({'task': key, 'score': manifest['score'], 'reward': manifest['reward'], 'failure_codes': manifest['failure_codes']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'replay'])
    parser.add_argument('task', choices=CONFIG)
    parser.add_argument('--label', default='replay')
    args = parser.parse_args()
    if not args.label.replace('_', '').isalnum():
        parser.error('label must be alphanumeric with optional underscores')
    if args.action == 'prepare': prepare(args.task)
    else: replay(args.task, args.label)
