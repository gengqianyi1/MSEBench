from __future__ import annotations
import importlib.util, json, subprocess, sys
from pathlib import Path

def _invalid(code, **evidence):
    return {
        'score': 0.0,
        'subscores': {'property_prediction': 0.0, 'candidate_library': 0.0},
        'failure_codes': [code],
        'evidence': evidence,
    }

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _independent_workflow_report(context, result, report):
    source_root = context.task_dir / 'input' / 'original' / 'src'
    sys.path.insert(0, str(source_root))
    try:
        evaluator = _load_module('task01_public_evaluator', source_root / 'evaluator.py')
        evaluation = evaluator.evaluate_result(result, context.submission_dir)
    except Exception as exc:
        evaluation = {'recomputed_error': f'{type(exc).__name__}: {exc}'}
    payload = {'evaluation': evaluation}
    trusted = None
    trusted_path = context.submission_dir / 'chgnet_tool_attempts.json'
    if trusted_path.is_file():
        try:
            data = json.loads(trusted_path.read_text(encoding='utf-8'))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            data = {}
        if data.get('counting_authority') == 'task01_sidecar' and data.get('chgnet_tool_attempts') is not None:
            trusted = int(data['chgnet_tool_attempts'])
    if trusted is not None:
        payload['chgnet_tool_attempts'] = trusted
    elif isinstance(report, dict) and report.get('chgnet_tool_attempts') is not None:
        payload['chgnet_tool_attempts'] = report.get('chgnet_tool_attempts')
    report_path = context.run_dir / 'independent_workflow_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding='utf-8')
    return report_path

def grade(context):
    result_path=context.submission_dir/'result.json'
    report_path=context.submission_dir/'native_final_report.json'
    if not result_path.is_file():
        return _invalid('MISSING_RESULT_JSON', path='result.json')
    if not report_path.is_file():
        return _invalid('MISSING_WORKFLOW_REPORT', path='native_final_report.json')
    try:
        result=json.loads(result_path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _invalid('INVALID_RESULT_JSON', error=str(exc))
    try:
        report=json.loads(report_path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _invalid('INVALID_WORKFLOW_REPORT', error=str(exc))
    evaluation=report.get('evaluation', report) if isinstance(report, dict) else {}
    if isinstance(evaluation, dict) and evaluation.get('integrity_violation'):
        return _invalid('INTEGRITY_VIOLATION', workflow_evaluation=evaluation)
    independent_report=_independent_workflow_report(context, result, report)
    output=context.run_dir/'native_private_score.json'
    command=[
        sys.executable,
        str(context.task_dir/'input'/'original'/'scripts'/'score_submission.py'),
        '--workspace', str(context.submission_dir),
        '--labels', str(context.private_data_dir/'test_labels_private.json.gz'),
        '--metadata', str(context.private_data_dir/'dataset_metadata.json'),
        '--workflow-report', str(independent_report),
        '--output', str(output),
    ]
    proc=subprocess.run(command,text=True,capture_output=True,timeout=3600)
    if proc.returncode:
        return _invalid(
            'NATIVE_SCORER_REJECTED_SUBMISSION',
            returncode=proc.returncode,
            stderr=proc.stderr[-4000:],
        )
    data=json.loads(output.read_text(encoding='utf-8'))
    return {'score':float(data['score']),'subscores':{'property_prediction':float(data['property_prediction']['numeric_score']),'candidate_library':float(data['candidate_library']['candidate_score'])},'failure_codes':[] if data.get('score_eligible') else ['NATIVE_SCORE_INELIGIBLE'],'evidence':data}
