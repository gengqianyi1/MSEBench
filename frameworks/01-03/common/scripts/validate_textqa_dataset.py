#!/usr/bin/env python3
"""Static privacy and structural checks for a generated TextQA version."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('dataset',type=Path); args=parser.parse_args()
    root=args.dataset.resolve(); manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    failures=[]
    if len(manifest['tasks']) != manifest['question_count']: failures.append('manifest count mismatch')
    names=set()
    for item in manifest['tasks']:
        task=root/item['task_directory']; name=item['task_name']
        if name in names: failures.append(f'duplicate task name {name}')
        names.add(name)
        required=[task/'task.toml',task/'instruction.md',task/'environment/Dockerfile',task/'environment/public/question.json',task/'tests/Dockerfile',task/'tests/grader.py',task/'tests/private_reference.json',task/'tests/test.sh']
        for path in required:
            if not path.is_file(): failures.append(f'missing {path.relative_to(root)}')
        public_question=json.loads((task/'environment/public/question.json').read_text(encoding='utf-8'))
        reference=json.loads((task/'tests/private_reference.json').read_text(encoding='utf-8'))
        forbidden={'answer','explanation'} & set(public_question)
        if forbidden: failures.append(f'private keys leaked in public question JSON: {name}: {sorted(forbidden)}')
        # Option labels are necessarily present in a multiple-choice question,
        # so matching an answer such as "A" in public text is a false positive.
        # Detect only a nontrivial verbatim explanation leak.
        if len(reference['explanation']) >= 40 and reference['explanation'] in (task/'instruction.md').read_text(encoding='utf-8'):
            failures.append(f'explanation leaked in instruction: {name}')
    if failures:
        print(json.dumps({'status':'FAIL','failures':failures},ensure_ascii=False)); return 1
    print(json.dumps({'status':'PASS','version':manifest['version'],'questions':manifest['question_count'],'source_sha256':manifest['source_sha256']},ensure_ascii=False)); return 0
if __name__=='__main__': raise SystemExit(main())
