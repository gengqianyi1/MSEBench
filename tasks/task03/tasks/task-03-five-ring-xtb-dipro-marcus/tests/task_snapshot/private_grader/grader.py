from __future__ import annotations
import json

def _invalid(code, **evidence):
    return {
        'score': 0.0,
        'subscores': {'recommended_rate_score': 0.0},
        'failure_codes': [code],
        'evidence': evidence,
    }

def grade(context):
    result_path=context.submission_dir/'result.json'
    answer_path=context.submission_dir/'final_answer.json'
    if not result_path.is_file(): return _invalid('MISSING_RESULT_JSON', path='result.json')
    if not answer_path.is_file(): return _invalid('MISSING_FINAL_ANSWER_JSON', path='final_answer.json')
    try:
        report=json.loads(result_path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _invalid('INVALID_RESULT_JSON', error=str(exc))
    try:
        final_answer=json.loads(answer_path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _invalid('INVALID_FINAL_ANSWER_JSON', error=str(exc))
    if not isinstance(report,dict) or not isinstance(final_answer,dict):
        return _invalid('INVALID_RESULT_SCHEMA', expected='JSON objects')
    checks=report.get('checks') or {}
    if not isinstance(checks,dict) or not checks.get('protocol_complete',False):
        return _invalid('INCOMPLETE_MULTI_ROUND', checks=checks)
    report_rec=(report.get('final_recommendation') or {}).get('recommended_candidate_id')
    answer_rec=final_answer.get('recommended_candidate_id')
    if report_rec != answer_rec:
        return {'score':0.0,'subscores':{'recommended_rate_score':0.0},'failure_codes':['FINAL_ANSWER_RESULT_MISMATCH'],'evidence':{'result_recommendation':report_rec,'final_answer_recommendation':answer_rec}}
    native=report.get('score') or {}
    if not isinstance(native,dict): return _invalid('INVALID_SCORE_SCHEMA')
    failures=[]
    if not native.get('recommendation_valid',False): failures.append('INVALID_FINAL_RECOMMENDATION')
    if not native.get('eligible',False): failures.append('RECOMMENDED_CANDIDATE_INELIGIBLE')
    if failures:
        return {'score':0.0,'subscores':{'recommended_rate_score':0.0},'failure_codes':failures,'evidence':{'native_score':native,'checks':checks,'final_recommendation':report.get('final_recommendation')}}
    try:
        score=float(native.get('total_score',0))
        rate_score=float(native.get('rate_performance_score',score))
    except (TypeError, ValueError) as exc:
        return _invalid('INVALID_SCORE_VALUE', error=str(exc))
    return {'score':score,'subscores':{'recommended_rate_score':rate_score},'failure_codes':[],'evidence':{'native_score':native,'checks':checks,'final_recommendation':report.get('final_recommendation')}}
