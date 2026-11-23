import json
from pathlib import Path

from jsonschema import Draft202012Validator


def test_example_result_matches_strict_schema():
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "tasks/jarvis_2d_exfoliation/result_schema.json").read_text())
    result = json.loads((root / "examples/example_result.json").read_text())
    errors = list(Draft202012Validator(schema).iter_errors(result))
    assert not errors, [error.message for error in errors]
