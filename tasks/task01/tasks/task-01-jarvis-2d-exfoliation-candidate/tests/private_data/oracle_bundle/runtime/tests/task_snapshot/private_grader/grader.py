from __future__ import annotations
import json, subprocess, sys

def _invalid(code, **evidence):
    return {
        'score': 0.0,
        'subscores': {'property_prediction': 0.0, 'candidate_library': 0.0},
        'failure_codes': [code],
        'evidence': evidence,
    }

def grade(context):
    result_path=context.submission_dir/'result.json'
    report_path=context.submission_dir/'native_final_report.json'
    if not result_path.is_file():
        return _invalid('MISSING_RESULT_JSON', path='result.json')
    if not report_path.is_file():
        return _invalid('MISSING_WORKFLOW_REPORT', path='native_final_report.json')
    try:
        json.loads(result_path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _invalid('INVALID_RESULT_JSON', error=str(exc))
    try:
        report=json.loads(report_path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _invalid('INVALID_WORKFLOW_REPORT', error=str(exc))
    evaluation=report.get('evaluation', report) if isinstance(report, dict) else {}
    if isinstance(evaluation, dict) and evaluation.get('integrity_violation'):
        return _invalid('INTEGRITY_VIOLATION', workflow_evaluation=evaluation)
    output=context.run_dir/'native_private_score.json'
    command=[
        sys.executable,
        str(context.task_dir/'input'/'original'/'scripts'/'score_submission.py'),
        '--workspace', str(context.submission_dir),
        '--labels', str(context.private_data_dir/'test_labels_private.json.gz'),
        '--metadata', str(context.private_data_dir/'dataset_metadata.json'),
        '--workflow-report', str(context.submission_dir/'native_final_report.json'),
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
