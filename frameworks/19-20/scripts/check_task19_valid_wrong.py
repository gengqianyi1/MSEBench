"""Private, offline scientific regression; NOT a Harbor or live-model score."""
import copy
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from task_paths import task_path
TESTS = task_path(19) / 'tests'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    adapter = load('review_adapter', TESTS / 'private_suite_adapter.py').create_adapter(TESTS / 'task_snapshot')
    grader = load('review_grader', TESTS / 'protocol_grader.py')
    oracle = json.loads((TESTS / 'fixtures/oracle.json').read_text(encoding='utf-8'))
    reference = json.loads((TESTS / 'task_snapshot/private_data/reference.json').read_text(encoding='utf-8'))
    calls = [('energy_resolved_msms', {'energy_eV': 45}),
             ('correlation_window', {'window': 'aromatic'}),
             ('correlation_window', {'window': 'sidechain'}),
             ('condition_perturbation', {'condition': 'pH4'})]
    state = adapter.start_session(None, 'review-events')
    events = [{'tool': tool, 'arguments': args, 'response': adapter.call_tool(0, tool, args, state)} for tool, args in calls]
    assert all(event['response']['ok'] for event in events)
    cases = {'oracle': copy.deepcopy(oracle)}
    no_claims = copy.deepcopy(oracle)
    no_claims['claims'] = []
    cases['valid_no_claims'] = no_claims
    wrong_claims = copy.deepcopy(oracle)
    for claim in wrong_claims['claims']:
        claim['relation'] = 'contradicts' if claim['relation'] == 'supports' else 'supports'
    cases['valid_wrong_claim_relations'] = wrong_claims
    true_ids = {m['candidate_id'] for m in oracle['members']}
    # Complete hypothesis space now admits legitimate zero-overlap alternatives.
    for overlap in (0, 1):
        pair, ids = next((pair, ids) for pair, ids in sorted(adapter.pairs.items()) if len(ids & true_ids) == overlap)
        candidate = copy.deepcopy(oracle)
        candidate['pair_hypothesis_id'] = pair
        candidate['members'] = [dict(candidate_id=cid, role='aromatic_oxidation_branch',
                                      abundance_interval=[0.4, 0.5], structural_features=sorted(adapter.features_by_candidate[cid])) for cid in sorted(ids)]
        candidate['pathways'] = [{'candidate_id': cid, 'events': ['PATH_OXIDATION']} for cid in sorted(ids)]
        candidate['excluded_pair_hypotheses'] = [p for p in sorted(adapter.pairs) if p != pair]
        cases[f'valid_wrong_pair_{overlap}_correct_members'] = candidate
        supported = copy.deepcopy(candidate)
        original_paths = {p['candidate_id']: p['events'] for p in oracle['pathways']}
        supported['pathways'] = [{'candidate_id': cid, 'events': original_paths.get(cid, ['PATH_OXIDATION'])} for cid in sorted(ids)]
        cases['valid_wrong_pair_gate_probe'] = supported
    records = []
    fixture_dir = TESTS / 'fixtures/valid_wrong'
    fixture_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='task19-review-') as temp:
        for name, candidate in cases.items():
            token = hashlib.sha256(name.encode()).hexdigest()[:32]
            session = adapter.start_session(temp, token)
            response = adapter.submit(0, candidate, events, session)
            assert response.get('ok') is True, (name, 'schema rejected')
            evidence = json.loads((Path(temp) / token / 'case-1-evidence.json').read_text(encoding='utf-8'))
            grader._validate_attestation(evidence, reference)
            result = grader.grade_evidence(evidence, reference)
            assert 0 <= result['score'] <= 100
            if name == 'oracle':
                assert result['score'] == 100, result
            else:
                assert result['score'] < 100, (name, result)
                (fixture_dir / (name + '.json')).write_text(json.dumps(candidate, indent=2) + '\n', encoding='utf-8')
            if 'correct_members' in name:
                ceiling = 30 if '_0_' in name else 49
                assert result['score'] <= ceiling
            if name == 'valid_wrong_pair_gate_probe':
                assert result['score'] == 49
                assert result['evidence']['joint_identity_gate_applied'] is True
            records.append(dict(fixture=name, schema_valid=True, attestation_valid=True,
                                score=result['score'], normalized_scientific_score=result['score']/100,
                                subscores=result['subscores'], evidence=result['evidence'], failure_codes=result['failure_codes']))
    out = ROOT / 'release_evidence/review_20260908'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'valid_wrong_results.json').write_text(json.dumps(dict(status='PASS', scope='offline signed-evidence scientific replay; not Harbor E2E or LLM evaluation', cases=records), indent=2) + '\n', encoding='utf-8')
    print(json.dumps([dict(fixture=r['fixture'], score=r['score'], schema_valid=True) for r in records], indent=2))


if __name__ == '__main__':
    main()
