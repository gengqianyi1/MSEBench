"""Inspect public contract compliance without reading private labels."""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/public_src')
from evaluator import evaluate_result

report = evaluate_result(json.loads(Path('/workspace/result.json').read_text()), Path('/workspace'),
                         Path('/public_schema/result_schema.json'))
print(json.dumps({key: value for key, value in report.items() if key not in ('candidate_report',)}, indent=2))
