"""Repair reference reporting from existing public artifacts and regenerate library.

Run in the locked scientific environment with /public_original and /workspace.
Never reads a private label or adjusts model predictions.
"""
import csv
import gzip
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path('/workspace')
PUBLIC = Path('/public_original')
sys.path.insert(0, str(PUBLIC / 'src'))
from evaluator import evaluate_result


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


start = time.monotonic()
result = read(WORK / 'result.json')
log = read(WORK / 'training_log.json')
versions = {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'pymatgen', 'joblib']}
# Supply the omitted public constraint asset so public hashes and generator replay work.
shutil.copyfile(PUBLIC / 'data/candidate_generation_constraints.json', WORK / 'data/candidate_generation_constraints.json')
metadata = read(WORK / 'data/dataset_metadata.json')
result['dataset']['checksums_verified'] = bool(metadata.get('public_file_sha256')) and all(
    (WORK / 'data' / name).is_file() and sha(WORK / 'data' / name) == expected
    for name, expected in metadata.get('public_file_sha256', {}).items())
with (WORK / 'predictions.csv').open() as stream:
    predictions = sorted(csv.DictReader(stream), key=lambda row: (float(row['predicted_exfoliation_energy_mev_atom']), row['sample_id']))[:10]
old_by_id = {str(item['sample_id']): item for item in result['recommended_candidates']}
result['recommended_candidates'] = [dict(old_by_id[row['sample_id']], rank=i,
    predicted_exfoliation_energy_mev_atom=float(row['predicted_exfoliation_energy_mev_atom'])) for i, row in enumerate(predictions, 1)]
# The former value was within-group prediction SD, not joint invariance MAE.
result['validation'].update({
    'mae_mev_atom': log['validation_mae_mev_atom'],
    'rmse_mev_atom': log['validation_rmse_mev_atom'],
    'spearman': log['validation_spearman'],
    'constant_baseline_mae_mev_atom': log['constant_baseline_mae_mev_atom'],
    'constant_baseline_rmse_mev_atom': log['constant_baseline_rmse_mev_atom'],
    'invariance_mae_mev_atom': log['invariance_diagnostic']['joint_invariance_mae_mev_atom'],
})
result['scientific_interpretation']['vacuum_sensitivity_assessment'] = (
    'The feature design omits explicit vacuum-cell height. The archived perturbation diagnostic reports '
    'mean within-group prediction SD of 0.122069 meV/atom and joint invariance MAE of 31.663169 meV/atom; '
    'these are distinct metrics. Exact invariance to every representation or vacuum perturbation is not established.')
log.update(package_versions=versions, random_seeds={'model': log['random_seed'], 'generator': 42},
    data_hashes={p.name: sha(p) for p in (WORK / 'data').iterdir() if p.is_file()},
    validation_metrics={k: v for k, v in result['validation'].items() if isinstance(v, (float, int))},
    selected_hyperparameters=log['model']['hyperparameters'], elapsed_seconds=None,
    provenance_note='Validation metrics are preserved from the source training log, not newly recomputed. '
    'Package versions describe this artifact replay environment. Original training elapsed time is unknown; '
    'runtime_seconds in the source log resembles a Unix timestamp and is not treated as duration.')
write(WORK / 'training_log.json', log)
generated = WORK / 'generator/replayed_library.json.gz'
command = [sys.executable, str(WORK / 'generator/generate.py'), '--input', str(WORK / 'data/train.json.gz'),
    '--constraints', str(WORK / 'data/candidate_generation_constraints.json'), '--config', str(WORK / 'generator/config.json'),
    '--output', str(generated)]
subprocess.run(command, check=True)
with gzip.open(generated, 'rt') as stream:
    actual = [json.loads(line) for line in stream if line.strip()]
with gzip.open(WORK / 'candidate_library.json.gz', 'rt') as stream:
    expected = [json.loads(line) for line in stream if line.strip()]
if actual != expected:
    raise RuntimeError('Deterministic generator differs from submitted library; refuse to certify reproducibility')
write(WORK / 'generator/generation_manifest.json', {
    'random_seed': 42, 'input_files': {'data/train.json.gz': sha(WORK / 'data/train.json.gz'),
        'data/candidate_generation_constraints.json': sha(WORK / 'data/candidate_generation_constraints.json')},
    'library_versions': versions, 'generation_parameters': {'config': 'generator/config.json', 'sha256': sha(WORK / 'generator/config.json')},
    'submitted_count': len(actual), 'interface_version': 'candidate-generator-v1',
    'replay': {'recordwise_exact_match': True, 'elapsed_seconds': time.monotonic() - start,
               'command_argv': command, 'comparison': 'decoded JSON records; gzip timestamps excluded'},
})
write(WORK / 'result.json', result)
evaluation = evaluate_result(result, WORK, PUBLIC / 'tasks/jarvis_2d_exfoliation/result_schema.json')
report = read(WORK / 'native_final_report.json')
report['evaluation'] = evaluation
report['summary'] = ('Public artifact reporting repaired and the 1000-record generator replayed exactly. '
    'Model predictions and five selected candidate IDs are unchanged. Validation metrics are archived estimates; '
    'CHGNet relaxation success is a local force convergence check, not proof of synthesizability or dynamical stability.')
write(WORK / 'native_final_report.json', report)
write(WORK / 'reference_public_audit.json', evaluation)
print(json.dumps({'public_errors': evaluation['errors'], 'compliance_penalty': evaluation['compliance_penalty'],
                  'generator_recordwise_match': True}))
