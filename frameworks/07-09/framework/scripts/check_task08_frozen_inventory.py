"""Verify every original Task 08 frozen case file without altering commitments."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'tasks/task-08-xtb-failure-diagnosis-restart/tests/task_snapshot/private_data/runtime_package'

def main():
    suite=json.loads((RUNTIME/'organizer_private/suite.json').read_text())
    rows=[]
    for case in [suite['public_case'],*suite['hidden_cases']]:
        work=RUNTIME/case['workspace']
        manifest=json.loads((work/'case_manifest.json').read_text())
        failures=[]
        for item in manifest['files']:
            path=work/item['path']
            if not path.is_file(): failures.append({'path':item['path'],'error':'missing'})
            elif path.stat().st_size!=item['size_bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256'].lower():
                failures.append({'path':item['path'],'error':'size_or_sha256_mismatch'})
        rows.append({'case_id':case['case_id'],'files_checked':len(manifest['files']),'failures':failures})
    result={'status':'PASS' if not any(r['failures'] for r in rows) else 'FAIL','cases':rows}
    output=ROOT/'release_evidence/task08/frozen_inventory_check.json'
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
    return 0 if result['status']=='PASS' else 1

if __name__=='__main__': raise SystemExit(main())
