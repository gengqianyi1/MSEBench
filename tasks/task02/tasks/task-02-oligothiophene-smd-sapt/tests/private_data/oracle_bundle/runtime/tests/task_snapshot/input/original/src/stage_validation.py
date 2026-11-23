from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

FORMAL_SYSTEMS = ["gas", "n-hexane", "1,4-dioxane", "chloroform", "water"]
ENDPOINTS = ["n-hexane", "water"]


def _load_json(path: Path) -> tuple[Any, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"{path.name} is not parseable JSON: {exc}"


def _resolve(workspace: Path, value: Any) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else workspace / p


def _metadata(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"\s*BENCHMARK_([A-Z0-9_]+)\s*:\s*(.*?)\s*$", line)
        if match:
            out[match.group(1).lower()] = match.group(2)
    return out


def _float(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def _schema_type_ok(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_schema_type_ok(value, item) for item in expected)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _schema_type_name(expected: Any) -> str:
    return "/".join(expected) if isinstance(expected, list) else str(expected)


def _minimal_schema_errors(data: Any, schema: Any, path: str = "$") -> list[str]:
    """Small JSON-schema subset used when jsonschema is unavailable."""
    if not isinstance(schema, dict):
        return []
    errors: list[str] = []

    expected_type = schema.get("type")
    if expected_type is not None and not _schema_type_ok(data, expected_type):
        return [f"{path} must be {_schema_type_name(expected_type)}"]

    if "const" in schema and data != schema["const"]:
        errors.append(f"{path} must equal {schema['const']!r}")
    if "enum" in schema and data not in schema["enum"]:
        errors.append(f"{path} must be one of {schema['enum']!r}")
    if isinstance(data, str):
        if "minLength" in schema and len(data) < int(schema["minLength"]):
            errors.append(f"{path} must have length >= {schema['minLength']}")
        pattern = schema.get("pattern")
        if pattern and re.match(str(pattern), data) is None:
            errors.append(f"{path} does not match required pattern")
    if isinstance(data, list):
        if "minItems" in schema and len(data) < int(schema["minItems"]):
            errors.append(f"{path} must contain at least {schema['minItems']} item(s)")
        if schema.get("uniqueItems") and len({json.dumps(item, sort_keys=True, default=str) for item in data}) != len(data):
            errors.append(f"{path} must contain unique items")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(data):
                errors.extend(_minimal_schema_errors(item, item_schema, f"{path}[{index}]"))
    if isinstance(data, dict):
        required = schema.get("required") or []
        for key in required:
            if key not in data:
                errors.append(f"{path}.{key} is required")
        properties = schema.get("properties") or {}
        for key, subschema in properties.items():
            if key in data:
                errors.extend(_minimal_schema_errors(data[key], subschema, f"{path}.{key}"))
        if schema.get("additionalProperties") is False:
            extra = sorted(set(data) - set(properties))
            for key in extra:
                errors.append(f"{path}.{key} is not an allowed property")
    return errors


def _validate_schema(data: Any, schema: Any, label: str) -> list[str]:
    try:
        import jsonschema
    except ModuleNotFoundError:
        return [f"{label} schema validation failed: {error}" for error in _minimal_schema_errors(data, schema)]
    except ImportError:
        return [f"{label} schema validation failed: {error}" for error in _minimal_schema_errors(data, schema)]
    try:
        jsonschema.validate(data, schema)
    except Exception as exc:
        return [f"{label} schema validation failed: {exc}"]
    return []


def _formal_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    meta = _metadata(text)
    energy = _float(meta.get("total_energy_hartree"))
    if energy is None:
        matches = re.findall(r"converged\s+SCF\s+energy\s*=\s*([-+0-9.eEdD]+)", text, flags=re.I)
        if matches:
            energy = _float(matches[-1].replace("D", "E").replace("d", "e"))
    converged = str(meta.get("converged", "")).strip().lower() in {"true", "1", "yes"}
    return {"metadata": meta, "energy_hartree": energy, "converged": converged}


def _read_xyz(path: Path) -> tuple[list[str], list[list[float]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError("XYZ has fewer than three lines")
    n = int(lines[0].strip())
    rows = lines[2:2 + n]
    if len(rows) != n:
        raise ValueError("XYZ atom count does not match file length")
    symbols: list[str] = []
    coords: list[list[float]] = []
    for row in rows:
        parts = row.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ line: {row}")
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, coords


def coordinate_hash(path: Path, digits: int) -> str:
    from rdkit.Chem import GetPeriodicTable

    symbols, coords = _read_xyz(path)
    table = GetPeriodicTable()
    rows = []
    for symbol, xyz in zip(symbols, coords):
        atomic_number = int(table.GetAtomicNumber(symbol))
        rows.append(f"{atomic_number},{xyz[0]:.{digits}f},{xyz[1]:.{digits}f},{xyz[2]:.{digits}f}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def validate_stage_A(workspace: Path) -> list[str]:
    required = ["stage_A_audit.json", "plan.json", "structures/solute_validated.sdf", "structures/solute_validated.xyz", "analysis/atom_mapping.json", "analysis/accepted_input_hashes.json"]
    errors = [f"missing required Stage A file: {rel}" for rel in required if not (workspace / rel).exists()]
    stage_path = workspace / "stage_A_audit.json"
    data, error = _load_json(stage_path) if stage_path.exists() else (None, None)
    if error:
        errors.append(error)
    elif isinstance(data, dict):
        schema_path = workspace / "stage_A_schema.json"
        schema, schema_error = _load_json(schema_path) if schema_path.exists() else (None, "stage_A_schema.json is missing from the workspace")
        if schema_error:
            errors.append(schema_error)
        else:
            errors.extend(_validate_schema(data, schema, "stage_A_audit.json"))
    elif stage_path.exists():
        errors.append("stage_A_audit.json must contain a JSON object")
    for rel in ("plan.json", "analysis/atom_mapping.json", "analysis/accepted_input_hashes.json"):
        path = workspace / rel
        if path.exists():
            _, parse_error = _load_json(path)
            if parse_error:
                errors.append(parse_error)
    return errors


def validate_stage_B(stage_file: str | Path, workspace: str | Path | None = None) -> list[str]:
    stage_path = Path(stage_file)
    if workspace is None:
        workspace_path = stage_path.parent.resolve()
    else:
        workspace_path = Path(workspace).resolve()
        if not stage_path.is_absolute():
            stage_path = workspace_path / stage_path
    errors: list[str] = []
    if not stage_path.exists():
        return ["missing required Stage B file: stage_B_bulk.json"]

    data, parse_error = _load_json(stage_path)
    if parse_error:
        return [parse_error]
    if not isinstance(data, dict):
        return ["stage_B_bulk.json must contain a JSON object"]

    schema_path = workspace_path / "stage_B_schema.json"
    if not schema_path.exists():
        errors.append("stage_B_schema.json is missing from the workspace")
    else:
        schema, schema_error = _load_json(schema_path)
        if schema_error:
            errors.append(schema_error)
        else:
            errors.extend(_validate_schema(data, schema, "stage_B_bulk.json"))

    instance_path = workspace_path / "instance.json"
    instance, instance_error = _load_json(instance_path)
    if instance_error:
        errors.append(instance_error)
        instance = {}
    permitted = set((instance or {}).get("sensitivity_options") or [])
    digits = int(((instance or {}).get("coordinate_hash_spec") or {}).get("round_digits", 10))

    search = data.get("gas_conformer_search") or {}
    selected = search.get("selected_R_g") or {}
    geometry_rel = selected.get("geometry_file")
    geometry_path = _resolve(workspace_path, geometry_rel) if geometry_rel else workspace_path / "structures/R_g.xyz"
    if not geometry_path.exists():
        errors.append(f"selected R_g geometry does not exist: {geometry_rel or 'structures/R_g.xyz'}")
        computed_hash = None
    else:
        try:
            computed_hash = coordinate_hash(geometry_path, digits)
        except Exception as exc:
            computed_hash = None
            errors.append(f"could not compute R_g coordinate hash: {exc}")
    reported_hash = selected.get("coordinate_sha256")
    if computed_hash and reported_hash != computed_hash:
        errors.append("gas_conformer_search.selected_R_g.coordinate_sha256 does not match structures/R_g.xyz")

    hash_path = workspace_path / "analysis/formal_coordinate_hashes.json"
    hashes, hash_error = _load_json(hash_path) if hash_path.exists() else (None, "analysis/formal_coordinate_hashes.json is missing")
    if hash_error:
        errors.append(hash_error)
    elif not isinstance(hashes, dict):
        errors.append("analysis/formal_coordinate_hashes.json must contain an object")
    else:
        if hashes.get("geometry_file") != "structures/R_g.xyz":
            errors.append("formal_coordinate_hashes.geometry_file must be structures/R_g.xyz")
        if computed_hash and hashes.get("coordinate_sha256") != computed_hash:
            errors.append("formal_coordinate_hashes.coordinate_sha256 does not match structures/R_g.xyz")
        systems = hashes.get("systems")
        if not isinstance(systems, dict):
            errors.append("formal_coordinate_hashes.systems must be an object")
        else:
            for name in FORMAL_SYSTEMS:
                if systems.get(name) != computed_hash:
                    errors.append(f"formal_coordinate_hashes.systems.{name} must equal the common R_g coordinate hash")

    formal = data.get("formal_calculations") or {}
    energies = formal.get("energies_hartree") or {}
    logs = formal.get("log_files") or {}
    base_log_paths: set[Path] = set()
    for system in FORMAL_SYSTEMS:
        reported_energy = _float(energies.get(system))
        if reported_energy is None:
            errors.append(f"formal_calculations.energies_hartree.{system} is missing or non-numeric")
        log_value = logs.get(system)
        if not log_value:
            errors.append(f"formal_calculations.log_files.{system} is missing")
            continue
        log_path = _resolve(workspace_path, log_value)
        base_log_paths.add(log_path.resolve())
        if not log_path.exists():
            errors.append(f"formal log does not exist for {system}: {log_value}")
            continue
        parsed = _formal_log(log_path)
        if parsed["energy_hartree"] is None:
            errors.append(f"formal log has no parseable energy for {system}: {log_value}")
        elif reported_energy is not None and abs(parsed["energy_hartree"] - reported_energy) > 1.0e-8:
            errors.append(f"reported energy for {system} does not match its raw log")
        if not parsed["converged"]:
            errors.append(f"formal log is not marked converged for {system}: {log_value}")
        meta = parsed["metadata"]
        if str(meta.get("solvent", "")) != system:
            errors.append(f"formal log BENCHMARK_SOLVENT mismatch for {system}: {log_value}")
        if str(meta.get("geometry_file", "")) != "structures/R_g.xyz":
            errors.append(f"formal log BENCHMARK_GEOMETRY_FILE must be structures/R_g.xyz for {system}")

    sensitivity = data.get("sensitivity_calculation") or {}
    sensitivity_type = sensitivity.get("type")
    if sensitivity_type not in permitted:
        errors.append(f"sensitivity type is not permitted by instance.json: {sensitivity_type!r}")
    sensitivity_logs = sensitivity.get("log_files") or []
    parseable_sensitivity = 0
    for value in sensitivity_logs:
        path = _resolve(workspace_path, value)
        if path.resolve() in base_log_paths:
            errors.append(f"sensitivity log must be distinct from common-reference logs: {value}")
            continue
        if not path.exists():
            errors.append(f"sensitivity log does not exist: {value}")
            continue
        if _formal_log(path).get("energy_hartree") is None:
            errors.append(f"sensitivity log has no parseable formal energy: {value}")
        else:
            parseable_sensitivity += 1
    if parseable_sensitivity < 1:
        errors.append("at least one distinct parseable sensitivity log is required")

    ranking = data.get("ranking") or {}
    if not isinstance(ranking.get("lowest_solvation_free_energy_solvent"), str) or not ranking.get("lowest_solvation_free_energy_solvent"):
        errors.append("ranking.lowest_solvation_free_energy_solvent is required")
    if not isinstance(ranking.get("ranking_robust_under_sensitivity"), bool):
        errors.append("ranking.ranking_robust_under_sensitivity must be boolean")
    if not isinstance(ranking.get("near_tie"), bool):
        errors.append("ranking.near_tie must be boolean")

    return errors


def validate_stage_C(workspace: Path) -> list[str]:
    path = workspace / "stage_C_local.json"
    if not path.exists():
        return ["missing required Stage C file: stage_C_local.json"]
    data, error = _load_json(path)
    if error:
        return [error]
    if not isinstance(data, dict):
        return ["stage_C_local.json must contain a JSON object"]
    schema_path = workspace / "stage_C_schema.json"
    schema, schema_error = _load_json(schema_path) if schema_path.exists() else (None, "stage_C_schema.json is missing from the workspace")
    if schema_error:
        return [schema_error]
    errors = _validate_schema(data, schema, "stage_C_local.json")
    for endpoint in ENDPOINTS:
        item = (data.get("endpoints") or {}).get(endpoint) or {}
        for key in ("selected_structure_file", "optimization_log"):
            value = item.get(key)
            if value and not _resolve(workspace, value).exists():
                errors.append(f"{endpoint}.{key} does not exist: {value}")
        sapt = item.get("sapt") or {}
        if sapt.get("raw_log") and not _resolve(workspace, sapt["raw_log"]).exists():
            errors.append(f"{endpoint}.sapt.raw_log does not exist")
    return errors


def validate_stage_D(workspace: Path) -> list[str]:
    errors: list[str] = []
    result_path = workspace / "result.json"
    revisions_path = workspace / "revisions.json"
    if not result_path.exists():
        errors.append("missing required Stage D file: result.json")
    if not revisions_path.exists():
        errors.append("missing required Stage D file: revisions.json")
    result, result_error = _load_json(result_path) if result_path.exists() else (None, None)
    revisions, revisions_error = _load_json(revisions_path) if revisions_path.exists() else (None, None)
    if result_error:
        errors.append(result_error)
    if revisions_error:
        errors.append(revisions_error)
    if result is not None:
        schema_path = workspace / "result_schema.json"
        schema, schema_error = _load_json(schema_path) if schema_path.exists() else (None, "result_schema.json is missing")
        if schema_error:
            errors.append(schema_error)
        else:
            errors.extend(_validate_schema(result, schema, "result.json"))
    if revisions is not None and not isinstance(revisions, (list, dict)):
        errors.append("revisions.json must contain a JSON array or object")
    return errors


def validate_stage(workspace: str | Path, stage_id: str) -> list[str]:
    root = Path(workspace).resolve()
    normalized = stage_id.strip().upper().replace("STAGE_", "")
    if normalized == "A":
        return validate_stage_A(root)
    if normalized == "B":
        return validate_stage_B(root / "stage_B_bulk.json", root)
    if normalized == "C":
        return validate_stage_C(root)
    if normalized == "D":
        return validate_stage_D(root)
    return [f"unknown stage id: {stage_id}"]
