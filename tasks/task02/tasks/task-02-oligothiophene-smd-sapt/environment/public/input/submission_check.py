"""Pre-submit artifact check for Task 02 (run from the workspace cwd)."""
from pathlib import Path
import json
from jsonschema import Draft202012Validator

REQUIRED = ["final_answer.json", "result.json", "native_final_report.json", "stage_A_audit.json", "plan.json", "stage_B_bulk.json", "stage_C_local.json", "revisions.json"]
WRAPPER_SCHEMAS = {
    "final_answer.json": "input/schemas/final_answer.schema.json",
    "native_final_report.json": "input/schemas/native_final_report.schema.json",
    "revisions.json": "input/schemas/revisions.schema.json",
}

def main() -> int:
    root = Path.cwd()
    missing = [p for p in REQUIRED if not (root / p).is_file()]
    errors = []
    for name in REQUIRED:
        path = root / name
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                expected = (list,) if name == "revisions.json" else (dict,)
                if not isinstance(value, expected):
                    kind = "array" if name == "revisions.json" else "object"
                    errors.append(f"{name}: top-level JSON value must be an {kind}")
                schema_name = WRAPPER_SCHEMAS.get(name)
                if schema_name:
                    schema = json.loads((root / schema_name).read_text(encoding="utf-8"))
                    for failure in Draft202012Validator(schema).iter_errors(value):
                        location = "/".join(str(item) for item in failure.absolute_path) or "<root>"
                        errors.append(f"{name}:{location}: {failure.message}")
            except Exception as exc:
                errors.append(f"{name}: invalid JSON ({exc})")
    print(json.dumps({"ok": not missing and not errors, "missing": missing, "errors": errors}, ensure_ascii=False, indent=2))
    return 0 if not missing and not errors else 1

if __name__ == "__main__":
    raise SystemExit(main())
