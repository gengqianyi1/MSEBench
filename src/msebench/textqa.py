"""Bilingual TextQA evaluation, one question per streaming model call."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import unicodedata

from .config import load_profile, api_key
from .llm import LLMClient, LLMError
from .evaluator import textqa_match
from .utils import timestamp

SYSTEM_PROMPT = (
    'Answer only the current materials science question. One or more options may '
    'be correct. Output only all correct option letters separated by commas '
    '(for example A,C,F). Do not include explanations.'
)


def extract_choices(text):
    """Parse a complete answer, never collect letters from arbitrary prose."""
    if not isinstance(text, str):
        return None
    text = unicodedata.normalize('NFKC', text).strip()
    text = re.sub(r'^```(?:text)?\s*|\s*```$', '', text).strip()
    boxed = re.fullmatch(r'\\boxed\{([^{}]+)\}', text)
    if boxed:
        text = boxed.group(1)
    text = text.replace('**', '').strip()
    text = re.sub(r'^(?:(?:final\s+)?answer|(?:最终|正确)?答案|正确选项)\s*(?:is|为|是)?\s*[:：]?\s*', '', text, flags=re.I)
    text = text.strip().rstrip('.。').strip()
    if not re.fullmatch(r'[A-Za-z](?:[\s,，、;；/|]*[A-Za-z])*', text):
        return None
    letters = re.findall(r'[A-Za-z]', text.upper())
    if len(letters) != len(set(letters)):
        return None
    return frozenset(letters)


def score_answer(prediction, gold, qtype='multiple_choice'):
    """Full match: 1; nonempty proper subset: 0.5; otherwise: 0."""
    if qtype in {'multiple_choice', 'single_choice'}:
        got, want = extract_choices(prediction), extract_choices(gold)
        if got is None or want is None:
            return 0.0
        if got == want:
            return 1.0
        return 0.5 if got and got < want else 0.0
    return float(textqa_match(prediction, gold))


def match_answer(prediction, gold, qtype='multiple_choice'):
    """Whether the answer earns full credit (compatibility helper)."""
    return score_answer(prediction, gold, qtype) == 1.0


def read_items(path, language='ch'):
    items, ids = [], set()
    with Path(path).open(encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError('expected an object')
                bilingual = 'question_ch' in item or 'question_en' in item
                question = item.get('question_' + language) if bilingual else item.get('question')
                if not isinstance(question, str) or not question.strip():
                    raise ValueError('missing or empty question for selected language')
                if not isinstance(item.get('answer'), str) or not item['answer'].strip():
                    raise ValueError('missing or empty answer')
                item = dict(item, question=question, type='multiple_choice' if bilingual else item.get('type', 'text'))
                if item['type'] in {'multiple_choice', 'single_choice'} and extract_choices(item['answer']) is None:
                    raise ValueError('invalid reference option letters')
                item.setdefault('id', f'line_{line_no}')
                if not isinstance(item['id'], str) or not item['id'] or item['id'] in ids:
                    raise ValueError('id must be a unique nonempty string')
                ids.add(item['id'])
                items.append(item)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{path}:{line_no}: {exc}') from exc
    if not items:
        raise ValueError(f'{path}: no questions')
    return items


def run_one(client, item, index, language='ch', prediction=None):
    prompt = SYSTEM_PROMPT if item['type'] in {'multiple_choice', 'single_choice'} else 'Answer only the current question concisely.'
    record = {key: item.get(key) for key in ('id', 'field', 'material_category', 'type', 'category')}
    record.update(index=index, language=language, question=item['question'], gold=item['answer'])
    try:
        result = client.chat([{'role': 'system', 'content': prompt}, {'role': 'user', 'content': item['question']}]) if client else {'content': prediction}
        answer = result.get('content') or ''
        choices = extract_choices(answer) if item['type'] in {'multiple_choice', 'single_choice'} else None
        score = score_answer(answer, item['answer'], item['type'])
        record.update(answer=answer, score=score, extracted=','.join(sorted(choices)) if choices else None,
                      correct=(score == 1.0), error=None)
        for key in ('model', 'elapsed_seconds', 'finish_reason', 'reasoning_content'):
            if key in result:
                record[key] = result[key]
    except LLMError as exc:
        record.update(answer=None, extracted=None, score=0.0, correct=False, error=str(exc))
    return record


def metrics(records):
    total = len(records)
    correct = sum(bool(r['correct']) for r in records)
    total_score = sum(r['score'] for r in records)
    return dict(total=total, correct=correct,
                partial=sum(r['score'] == 0.5 for r in records),
                total_score=total_score,
                score_rate=round(total_score / total, 6) if total else 0.0,
                accuracy=round(correct / total, 6) if total else 0.0,
                errors=sum(bool(r['error']) for r in records))


def main(argv=None, *, default_input=None, default_results=Path('results')):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile')
    parser.add_argument('--config', type=Path, help='Model profile JSON')
    parser.add_argument('--input', type=Path, default=default_input, required=default_input is None)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--language', choices=('ch', 'en'), default='ch')
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--max-tokens', type=int)
    parser.add_argument('--predictions', type=Path, help='Offline JSONL: unique id and answer for every input question')
    args = parser.parse_args(argv)
    if args.concurrency < 1 or (args.max_tokens is not None and args.max_tokens < 1):
        parser.error('concurrency and max-tokens must be positive')
    try:
        items = read_items(args.input, args.language)
        predictions = {}
        if args.predictions:
            with args.predictions.open(encoding='utf-8') as handle:
                for n, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not isinstance(row.get('answer'), str):
                        raise ValueError(f'predictions line {n}: expected string id and answer')
                    if row['id'] in predictions:
                        raise ValueError(f'duplicate prediction id: {row["id"]}')
                    predictions[row['id']] = row['answer']
            if set(predictions) != {item['id'] for item in items}:
                raise ValueError('prediction IDs must match input IDs exactly')
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    cfg, client = {}, None
    if not args.predictions:
        cfg = load_profile(args.profile, args.config)
        if not cfg.get('api_base') or not cfg.get('model'):
            parser.error('model mode requires MSEBENCH_API_BASE and MSEBENCH_MODEL_ID or a configured profile')
        cfg['stream'] = True
        if args.max_tokens is not None:
            cfg['max_tokens'] = args.max_tokens
        cfg['api_key'] = api_key(cfg)
        client = LLMClient(cfg)
    results_root = Path(os.getenv('MSEBENCH_RESULTS_ROOT') or os.getenv('MATBENCH_RESULTS_ROOT') or default_results)
    output = args.output or results_root / 'textqa' / timestamp() / 'results.jsonl'
    summary_path = output.with_name('summary.json')
    inputs = {args.input.resolve(), args.predictions.resolve() if args.predictions else None}
    if output.resolve() in inputs or summary_path.resolve() in inputs or output.resolve() == summary_path.resolve():
        parser.error('output and summary must be distinct and must not overwrite inputs')
    output.parent.mkdir(parents=True, exist_ok=True)
    def evaluate(pair):
        index, item = pair
        return run_one(client, item, index, args.language, predictions.get(item['id']))
    records = []
    with output.open('w', encoding='utf-8') as handle, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for record in pool.map(evaluate, enumerate(items)):
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()
    summary = dict(task='textqa', scoring='wrong_zero_subset_half', input=str(args.input.resolve()), language=args.language,
                   mode='offline' if args.predictions else 'model', profile=cfg.get('profile'), model=cfg.get('model'),
                   **metrics(records))
    for dimension in ('field', 'material_category'):
        groups = defaultdict(list)
        for record in records:
            groups[record.get(dimension) or 'unknown'].append(record)
        summary['by_' + dimension] = {key: metrics(group) for key, group in sorted(groups.items())}
    for key in ('stream', 'temperature', 'top_p', 'max_tokens'):
        summary[key] = cfg.get(key)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if summary['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
