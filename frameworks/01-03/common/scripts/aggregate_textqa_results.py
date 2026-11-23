#!/usr/bin/env python3
"""Aggregate Harbor verifier details for one TextQA generation version."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('results_root',type=Path); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    rows=[]
    for path in args.results_root.rglob('details.json'):
        try:
            d=json.loads(path.read_text(encoding='utf-8'))
            if d.get('schema_version')!='textqa-verifier-details/1.0': continue
            item=d.get('private_result',{}); evidence=item.get('evidence',{})
            rows.append({'trial':str(path.parent.parent),'task_id':d.get('task_id'),'source_id':evidence.get('question_id'),'question_type':evidence.get('question_type'),'predicted_answer':evidence.get('predicted_answer'),'score':item.get('score',0),'failure_codes':item.get('failure_codes',[])})
        except Exception: continue
    by_type=defaultdict(list)
    for row in rows: by_type[row['question_type']].append(row['score'])
    report={'schema_version':'textqa-results-summary/1.0','question_count':len(rows),'mean_score':sum(r['score'] for r in rows)/len(rows) if rows else None,'mean_reward':sum(r['score'] for r in rows)/(100*len(rows)) if rows else None,'by_question_type':{key:{'count':len(values),'mean_score':sum(values)/len(values)} for key,values in by_type.items()},'rows':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps({k:report[k] for k in ('question_count','mean_score','mean_reward')},ensure_ascii=False))
if __name__=='__main__': raise SystemExit(main())
