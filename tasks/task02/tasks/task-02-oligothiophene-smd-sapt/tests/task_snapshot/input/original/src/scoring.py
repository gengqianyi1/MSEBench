from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


STAGE_RANK = {"stage_A": 1, "stage_B": 2, "stage_C": 3, "stage_D": 4}
STAGE_FILE_RANK = {
    "stage_A_audit.json": 1,
    "stage_B_bulk.json": 2,
    "stage_C_local.json": 3,
    "result.json": 4,
}
REQUIRED_STAGE_FILES = ["stage_A_audit.json", "plan.json", "stage_B_bulk.json", "stage_C_local.json", "result.json", "revisions.json"]
FORMAL_SOLVENTS = ["gas", "n-hexane", "1,4-dioxane", "chloroform", "water"]
ENDPOINTS = ["n-hexane", "water"]


def _load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            rows.append({"_malformed": line})
    return rows


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return None


def _norm_path(value: Any) -> str | None:
    if not value:
        return None
    return str(value).replace("\\", "/").lstrip("./")


def _resolve(workspace: Path, value: Any) -> Path | None:
    rel = _norm_path(value)
    if not rel:
        return None
    p = Path(rel)
    return p if p.is_absolute() else workspace / p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _recursive_values(obj: Any, keys: set[str]) -> List[Any]:
    out: List[Any] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) in keys:
                out.append(v)
            out.extend(_recursive_values(v, keys))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_recursive_values(item, keys))
    return out


def _text_blob(*objs: Any) -> str:
    chunks: List[str] = []
    for obj in objs:
        if isinstance(obj, str):
            chunks.append(obj)
        elif obj is not None:
            try:
                chunks.append(json.dumps(obj, ensure_ascii=False))
            except Exception:
                chunks.append(str(obj))
    return " ".join(chunks).lower()


def _construct_expected_molecule(instance: Dict[str, Any]):
    from rdkit import Chem

    cfg = instance["molecule"]
    repeat = cfg["repeat_unit_smiles"]
    length = int(cfg["oligomer_length"])
    if length != 2:
        raise ValueError("Bundled scorer currently supports length 2")
    units = [Chem.MolFromSmiles(repeat, sanitize=False) for _ in range(2)]
    if any(m is None for m in units):
        raise ValueError("repeat parse failed")
    rw = Chem.RWMol(Chem.CombineMols(units[0], units[1]))
    offset = units[0].GetNumAtoms()

    def neighbor(mol, isotope: int, off: int) -> int:
        atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0 and a.GetIsotope() == isotope]
        if len(atoms) != 1 or len(list(atoms[0].GetNeighbors())) != 1:
            raise ValueError("connection labels invalid")
        return list(atoms[0].GetNeighbors())[0].GetIdx() + off

    rw.AddBond(neighbor(units[0], 1, 0), neighbor(units[1], 2, offset), Chem.BondType.SINGLE)
    for idx in sorted([a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0], reverse=True):
        rw.RemoveAtom(idx)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def _candidate_structure_path(workspace: Path, result: Dict[str, Any], stage_a: Dict[str, Any]) -> Path | None:
    candidates = []
    model = result.get("molecular_model") or {}
    for key in ("structure_sdf", "validated_structure_sdf", "sdf_file"):
        candidates.append(model.get(key))
    candidates.extend(model.get("structure_files") or [])
    candidates.extend(_recursive_values(stage_a, {"structure_sdf", "validated_structure_sdf", "sdf_file"}))
    candidates.append("structures/solute_validated.sdf")
    for value in candidates:
        p = _resolve(workspace, value)
        if p and p.exists() and p.suffix.lower() in {".sdf", ".mol"}:
            return p
    return None


def _parse_molecule(path: Path):
    from rdkit import Chem

    if path.suffix.lower() == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
        return next((m for m in supplier if m is not None), None)
    return Chem.MolFromMolFile(str(path), removeHs=False, sanitize=True)


def _read_xyz(path: Path) -> Tuple[List[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError("empty xyz")
    n = int(lines[0].strip())
    atom_lines = lines[2:2 + n]
    if len(atom_lines) != n:
        raise ValueError("truncated xyz")
    symbols: List[str] = []
    coords: List[List[float]] = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError("invalid xyz line")
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.asarray(coords, dtype=float)


def _coordinate_hash(path: Path, digits: int = 10) -> str:
    from rdkit.Chem import GetPeriodicTable

    symbols, coords = _read_xyz(path)
    table = GetPeriodicTable()
    records = []
    for symbol, xyz in zip(symbols, coords):
        z = int(table.GetAtomicNumber(symbol))
        records.append(f"{z},{xyz[0]:.{digits}f},{xyz[1]:.{digits}f},{xyz[2]:.{digits}f}")
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def _same_xyz_evidence(
    workspace: Path,
    first: Any,
    second: Any,
    *,
    atol_angstrom: float = 1.0e-6,
) -> bool:
    if _norm_path(first) == _norm_path(second):
        return True
    first_path = _resolve(workspace, first)
    second_path = _resolve(workspace, second)
    if not first_path or not second_path or not first_path.exists() or not second_path.exists():
        return False
    try:
        first_symbols, first_coords = _read_xyz(first_path)
        second_symbols, second_coords = _read_xyz(second_path)
    except (OSError, ValueError):
        return False
    return (
        first_symbols == second_symbols
        and first_coords.shape == second_coords.shape
        and bool(np.allclose(first_coords, second_coords, rtol=0.0, atol=atol_angstrom))
    )


def _parse_metadata(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"\s*BENCHMARK_([A-Z0-9_]+)\s*:\s*(.*?)\s*$", line)
        if match:
            out[match.group(1).lower()] = match.group(2)
    return out


def _last_float(patterns: Iterable[str], text: str) -> float | None:
    found: List[float] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I | re.M):
            try:
                found.append(float(match.group(1).replace("D", "E").replace("d", "e")))
            except Exception:
                pass
    return found[-1] if found else None


def _parse_formal_log(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    metadata = _parse_metadata(text)
    energy = _last_float(
        [
            r"BENCHMARK_TOTAL_ENERGY_HARTREE\s*:\s*([-+0-9.eEdD]+)",
            r"converged\s+SCF\s+energy\s*=\s*([-+0-9.eEdD]+)",
            r"FINAL\s+SINGLE\s+POINT\s+ENERGY\s+([-+0-9.eEdD]+)",
            r"total\s+energy\s*[=:]\s*([-+0-9.eEdD]+)",
        ],
        text,
    )
    converged = bool(re.search(r"converged\s+SCF\s+energy|SCF\s+converged|BENCHMARK_CONVERGED\s*:\s*(true|yes|1)", text, flags=re.I))
    orbital = {
        "homo_hartree": _last_float([r"BENCHMARK_HOMO_HARTREE\s*:\s*([-+0-9.eEdD]+)"], text),
        "lumo_hartree": _last_float([r"BENCHMARK_LUMO_HARTREE\s*:\s*([-+0-9.eEdD]+)"], text),
        "dipole_x_debye": _last_float([r"BENCHMARK_DIPOLE_X_DEBYE\s*:\s*([-+0-9.eEdD]+)"], text),
        "dipole_y_debye": _last_float([r"BENCHMARK_DIPOLE_Y_DEBYE\s*:\s*([-+0-9.eEdD]+)"], text),
        "dipole_z_debye": _last_float([r"BENCHMARK_DIPOLE_Z_DEBYE\s*:\s*([-+0-9.eEdD]+)"], text),
        "dipole_magnitude_debye": _last_float([r"BENCHMARK_DIPOLE_MAGNITUDE_DEBYE\s*:\s*([-+0-9.eEdD]+)"], text),
    }
    combined_dipole = re.search(
        r"BENCHMARK_DIPOLE_DEBYE_XYZ\s*:\s*"
        r"([-+0-9.eEdD]+)\s*,\s*([-+0-9.eEdD]+)\s*,\s*([-+0-9.eEdD]+)",
        text,
        flags=re.I,
    )
    if combined_dipole:
        for key, value in zip(
            ("dipole_x_debye", "dipole_y_debye", "dipole_z_debye"),
            combined_dipole.groups(),
        ):
            if orbital[key] is None:
                orbital[key] = float(value.replace("D", "E").replace("d", "e"))
    return {"path": path, "text": text, "metadata": metadata, "energy_hartree": energy, "converged": converged, **orbital}


def _entry_name(entry: Dict[str, Any]) -> str:
    return str(entry.get("solvent") or entry.get("name") or entry.get("label") or "").strip().lower()


def _formal_entries(result: Dict[str, Any], stage_b: Dict[str, Any], workspace: Path) -> Dict[str, Dict[str, Any]]:
    bulk = result.get("bulk_solvation") or {}
    out: Dict[str, Dict[str, Any]] = {}
    canonical = stage_b.get("formal_calculations") or {}
    canonical_logs = canonical.get("log_files") or {}
    canonical_energies = canonical.get("energies_hartree") or {}
    canonical_geometry = canonical.get("geometry_file")
    if isinstance(canonical_logs, dict):
        for name in FORMAL_SOLVENTS:
            log_value = canonical_logs.get(name)
            if not log_value:
                continue
            entry = {
                "solvent": name,
                "energy_hartree": canonical_energies.get(name),
                "log_file": log_value,
                "geometry_file": canonical_geometry,
            }
            out[name] = {
                "entry": entry,
                "log_path": _resolve(workspace, log_value),
                "geometry_path": _resolve(workspace, canonical_geometry),
            }
    sources = [
        bulk.get("formal_calculations"), bulk.get("calculations"), bulk.get("solvation_results"),
        stage_b.get("formal_calculations"), stage_b.get("calculations"), stage_b.get("solvation_results"),
    ]
    entries: List[Dict[str, Any]] = []
    for source in sources:
        if isinstance(source, list):
            entries.extend(x for x in source if isinstance(x, dict))
        elif isinstance(source, dict):
            for key, value in source.items():
                if isinstance(value, dict):
                    value = dict(value)
                    value.setdefault("solvent", key)
                    entries.append(value)
    gas_ref = bulk.get("gas_reference") or stage_b.get("gas_reference")
    if isinstance(gas_ref, dict):
        gas_ref = dict(gas_ref)
        gas_ref.setdefault("solvent", "gas")
        entries.append(gas_ref)

    for entry in entries:
        name = _entry_name(entry)
        if name in {"hexane", "nhexane"}:
            name = "n-hexane"
        if name in {"dioxane", "1,4 dioxane"}:
            name = "1,4-dioxane"
        if name not in FORMAL_SOLVENTS:
            continue
        log_value = next((entry.get(k) for k in ("raw_log", "log_file", "output_file", "evidence_file") if entry.get(k)), None)
        geom_value = next((entry.get(k) for k in ("geometry_file", "input_geometry", "xyz_file") if entry.get(k)), None)
        log_path = _resolve(workspace, log_value)
        geom_path = _resolve(workspace, geom_value)
        out.setdefault(name, {"entry": entry, "log_path": log_path, "geometry_path": geom_path})

    for path in sorted((workspace / "logs" / "formal").glob("**/*")) if (workspace / "logs" / "formal").exists() else []:
        if not path.is_file():
            continue
        try:
            parsed = _parse_formal_log(path)
        except Exception:
            continue
        name = str(parsed["metadata"].get("solvent", "")).lower()
        if name in FORMAL_SOLVENTS and name not in out:
            geom = _resolve(workspace, parsed["metadata"].get("geometry_file"))
            out[name] = {"entry": {}, "log_path": path, "geometry_path": geom}
    return out


def _reported_energy(entry: Dict[str, Any]) -> float | None:
    for key in ("energy_hartree", "total_energy_hartree", "electronic_energy_hartree", "energy_au"):
        value = _as_float(entry.get(key))
        if value is not None:
            return value
    return None


def _reported_delta(entry: Dict[str, Any]) -> float | None:
    for key in ("delta_g_solv_kcal_mol", "solvation_energy_kcal_mol", "delta_kcal_mol", "delta_G_solv_SMD_kcal_mol"):
        value = _as_float(entry.get(key))
        if value is not None:
            return value
    return None


def _extract_ranking_value(bulk: Dict[str, Any], key: str) -> Any:
    candidates = (key,)
    for container in (bulk, bulk.get("ranking") if isinstance(bulk.get("ranking"), dict) else {}):
        for k in candidates:
            if k in container:
                return container.get(k)
    return None


def _kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    p0 = p - p.mean(axis=0)
    q0 = q - q.mean(axis=0)
    h = p0.T @ q0
    u, _, vt = np.linalg.svd(h)
    d = np.linalg.det(vt.T @ u.T)
    if d < 0:
        vt[-1, :] *= -1
    r = vt.T @ u.T
    return float(np.sqrt(np.mean(np.sum((p0 @ r - q0) ** 2, axis=1))))


def _dihedral(coords: np.ndarray, idx: List[int]) -> float:
    p0, p1, p2, p3 = [coords[i] for i in idx]
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def _selected_endpoint_files(result: Dict[str, Any], stage_c: Dict[str, Any], workspace: Path) -> Dict[str, Path]:
    conf = result.get("conformational_analysis") or stage_c.get("conformational_analysis") or stage_c.get("conformational_probe") or {}
    solvents = stage_c.get("endpoints") or conf.get("solvents") or conf.get("endpoints") or {}
    out: Dict[str, Path] = {}
    if isinstance(solvents, dict):
        for endpoint in ENDPOINTS:
            item = solvents.get(endpoint) or solvents.get("hexane" if endpoint == "n-hexane" else endpoint) or {}
            if isinstance(item, dict):
                value = item.get("selected_structure_file") or item.get("xyz_file") or item.get("structure_file")
                p = _resolve(workspace, value)
                if p and p.exists():
                    out[endpoint] = p
    for endpoint in ENDPOINTS:
        if endpoint in out:
            continue
        patterns = [f"**/*{endpoint}*selected*.xyz"]
        if endpoint == "n-hexane":
            patterns.append("**/*hexane*selected*.xyz")
        for pattern in patterns:
            matches = list(workspace.glob(pattern))
            if matches:
                out[endpoint] = matches[0]
                break
    return out


def _mapping_indices(workspace: Path) -> Dict[str, Any]:
    data = _load_json(workspace / "analysis" / "atom_mapping.json", {}) or {}
    return data if isinstance(data, dict) else {}


def _index_list(mapping: Dict[str, Any], keys: Iterable[str], length: int | None = None) -> List[int] | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, list) and all(isinstance(x, int) for x in value):
            if length is None or len(value) == length:
                return value
    return None


def _parse_sapt_log(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    meta = _parse_metadata(text)
    patterns = {
        "electrostatics": [r"BENCHMARK_SAPT_ELST_KCAL_MOL\s*:\s*([-+0-9.eEdD]+)", r"^\s*Electrostatics\s+[-+0-9.eEdD]+\s+\[mEh\]\s+([-+0-9.eEdD]+)\s+\[kcal/mol\]"],
        "exchange": [r"BENCHMARK_SAPT_EXCH_KCAL_MOL\s*:\s*([-+0-9.eEdD]+)", r"^\s*Exchange\s+[-+0-9.eEdD]+\s+\[mEh\]\s+([-+0-9.eEdD]+)\s+\[kcal/mol\]"],
        "induction": [r"BENCHMARK_SAPT_IND_KCAL_MOL\s*:\s*([-+0-9.eEdD]+)", r"^\s*Induction\s+[-+0-9.eEdD]+\s+\[mEh\]\s+([-+0-9.eEdD]+)\s+\[kcal/mol\]"],
        "dispersion": [r"BENCHMARK_SAPT_DISP_KCAL_MOL\s*:\s*([-+0-9.eEdD]+)", r"^\s*Dispersion\s+[-+0-9.eEdD]+\s+\[mEh\]\s+([-+0-9.eEdD]+)\s+\[kcal/mol\]"],
        "total": [r"BENCHMARK_SAPT_TOTAL_KCAL_MOL\s*:\s*([-+0-9.eEdD]+)", r"^\s*Total\s+SAPT0\s+[-+0-9.eEdD]+\s+\[mEh\]\s+([-+0-9.eEdD]+)\s+\[kcal/mol\]", r"SAPT0\s+total\s*[=:]\s*([-+0-9.]+)"],
        "ssapt0_total": [r"BENCHMARK_SSAPT0_TOTAL_KCAL_MOL\s*:\s*([-+0-9.eEdD]+)", r"^\s*Total\s+sSAPT0\s+[-+0-9.eEdD]+\s+\[mEh\]\s+([-+0-9.eEdD]+)\s+\[kcal/mol\]", r"sSAPT0\s+total\s*[=:]\s*([-+0-9.]+)"],
    }
    parsed = {key: _last_float(value, text) for key, value in patterns.items()}
    return {"path": path, "text": text, "metadata": meta, **parsed}


def _sapt_items(result: Dict[str, Any], stage_c: Dict[str, Any], workspace: Path) -> Dict[str, Dict[str, Any]]:
    local = result.get("local_interactions") or stage_c.get("local_interactions") or stage_c.get("explicit_complex_analysis") or {}
    solvents = stage_c.get("endpoints") or local.get("solvents") or local.get("endpoints") or {}
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(solvents, dict):
        for endpoint in ENDPOINTS:
            item = solvents.get(endpoint) or solvents.get("hexane" if endpoint == "n-hexane" else endpoint) or {}
            if not isinstance(item, dict) or not item:
                continue
            sapt = item.get("sapt") if isinstance(item.get("sapt"), dict) else item
            log_value = next((sapt.get(k) for k in ("raw_log", "log_file", "output_file", "psi4_output_file") if sapt.get(k)), None)
            log_path = _resolve(workspace, log_value)
            out[endpoint] = {"item": item, "sapt": sapt, "log_path": log_path}
    for path in sorted((workspace / "logs" / "sapt").glob("**/*")) if (workspace / "logs" / "sapt").exists() else []:
        if not path.is_file():
            continue
        try:
            parsed = _parse_sapt_log(path)
        except Exception:
            continue
        endpoint = str(parsed["metadata"].get("solvent", "")).lower()
        if endpoint == "hexane":
            endpoint = "n-hexane"
        if endpoint in ENDPOINTS and endpoint not in out:
            out[endpoint] = {"item": {}, "sapt": {}, "log_path": path}
    return out


def _dominant_attractive(parsed: Dict[str, Any]) -> str | None:
    attractive = {k: abs(v) for k in ("electrostatics", "induction", "dispersion") if (v := _as_float(parsed.get(k))) is not None and v < 0}
    return max(attractive, key=attractive.get) if attractive else None


def _reported_dominance(item: Dict[str, Any]) -> str | None:
    for key in ("largest_attractive_component", "dominant_attractive_component", "component_dominance"):
        value = item.get(key)
        if isinstance(value, str):
            low = value.lower()
            token = low.strip().replace("_", "-")
            if "electro" in token or token.startswith(("elst", "elec")):
                return "electrostatics"
            if "induct" in token or token.startswith(("ind", "pol")):
                return "induction"
            if "dispers" in token or token.startswith("disp"):
                return "dispersion"
    return None


def _stage_mutations(run_dir: Path) -> List[Dict[str, Any]]:
    rows = _read_jsonl(run_dir / "stage_integrity.jsonl")
    mutations: List[Dict[str, Any]] = []
    for file_name, file_rank in STAGE_FILE_RANK.items():
        frozen_hash: str | None = None
        frozen_at: int | None = None
        appeared = False
        for row in rows:
            rank = STAGE_RANK.get(str(row.get("runner_stage")), 0)
            snap = row.get("snapshot") or {}
            item = snap.get(file_name) if isinstance(snap, dict) else None
            if isinstance(item, dict):
                appeared = True
                hash_value = item.get("canonical_json_sha256") or item.get("sha256")
                if rank <= file_rank:
                    frozen_hash = hash_value
                    frozen_at = row.get("tool_index")
                elif frozen_hash is not None and hash_value != frozen_hash:
                    mutations.append({"file":file_name,"kind":"semantic_change","frozen_at_tool":frozen_at,"changed_at_tool":row.get("tool_index")})
                    break
            elif rank > file_rank and appeared and frozen_hash is not None:
                mutations.append({"file":file_name,"kind":"deleted","frozen_at_tool":frozen_at,"changed_at_tool":row.get("tool_index")})
                break
    return mutations


def _anomaly_contaminated(result: Dict[str, Any], stage_a: Dict[str, Any], stage_b: Dict[str, Any], stage_c: Dict[str, Any], anomaly_path: str) -> bool:
    accepted = _recursive_values(stage_a, {"accepted_inputs", "accepted_files", "accepted_artifacts"})
    used_inputs = _recursive_values(stage_b, {
        "input_geometry", "starting_geometry_file", "seed_geometry_file",
        "source_endpoint_geometry", "selected_structure_file", "geometry_file",
        "complex_geometry_file", "monomer_a_xyz", "monomer_b_xyz",
    }) + _recursive_values(stage_c, {
        "input_geometry", "starting_geometry_file", "seed_geometry_file",
        "source_endpoint_geometry", "selected_structure_file", "geometry_file",
        "complex_geometry_file", "monomer_a_xyz", "monomer_b_xyz",
    })
    downstream_text = _text_blob(accepted, used_inputs)
    norm = anomaly_path.lower().replace("\\", "/")
    base = Path(norm).name
    return norm in downstream_text or base in downstream_text




def _infer_anomaly_truth(workspace: Path, instance: Dict[str, Any]) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    metadata_path = workspace / "seed_evidence" / "structure_metadata.json"
    sdf_path = workspace / "seed_evidence" / "provided_structure.sdf"
    if metadata_path.exists() and sdf_path.exists():
        metadata = _load_json(metadata_path, {}) or {}
        if metadata.get("sha256") and str(metadata.get("sha256")) != _sha256(sdf_path):
            candidates.append({"artifact":"seed_evidence/structure_metadata.json","anomaly_type":"stale_hash","anomaly_reason":"recorded SDF hash mismatch"})
        try:
            mol = _parse_molecule(sdf_path)
            if mol is not None and int(metadata.get("explicit_atom_count")) != mol.GetNumAtoms():
                candidates.append({"artifact":"seed_evidence/structure_metadata.json","anomaly_type":"atom_count_mismatch","anomaly_reason":"recorded atom count mismatch"})
        except Exception:
            pass
    solvent_path = workspace / "seed_evidence" / "solvent_probe.json"
    if solvent_path.exists():
        probe = _load_json(solvent_path, {}) or {}
        water_cfg = (instance.get("solvents") or {}).get("water") or {}
        if probe.get("smiles") == water_cfg.get("smiles"):
            if probe.get("solvent_name") != "water" or probe.get("smd_name") != water_cfg.get("smd_name"):
                candidates.append({"artifact":"seed_evidence/solvent_probe.json","anomaly_type":"solvent_label_mismatch","anomaly_reason":"water identity conflicts with label or SMD name"})
            elif probe.get("xtb_alpb_name") != water_cfg.get("xtb_alpb_name"):
                candidates.append({"artifact":"seed_evidence/solvent_probe.json","anomaly_type":"solvent_alias_mismatch","anomaly_reason":"water identity conflicts with ALPB alias"})
    return {"anomalies": candidates}


def _counterfactual_ok(result: Dict[str, Any]) -> bool:
    item = result.get("counterfactual_next_calculation") or {}
    if not isinstance(item, dict):
        return False
    has_calc = any(item.get(k) for k in ("calculation", "proposal", "method", "description"))
    has_quant = any(_as_float(item.get(k)) is not None for k in ("decision_threshold", "target_uncertainty", "expected_information_gain", "quantitative_criterion"))
    return has_calc and has_quant


def _states_evidence_limit(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "does not",
            "cannot",
            "not establish",
            "insufficient",
            "would require",
            "requires additional",
            "requires further",
            "static local",
        )
    )



def _canonical_solvent_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    smiles = str(entry.get("smiles") or "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False}
    return {
        "valid": True,
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "formula": rdMolDescriptors.CalcMolFormula(Chem.AddHs(mol)),
        "smd_name": str(entry.get("smd_name") or entry.get("solvent_name") or ""),
        "xtb_alpb_name": str(entry.get("xtb_alpb_name") or entry.get("alpb_alias") or ""),
    }


def _reported_solvent_audit(stage_a: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    for obj in (stage_a.get("solvent_identity_audit"), result.get("solvent_identity_audit"), (result.get("molecular_model") or {}).get("solvents")):
        if isinstance(obj, dict):
            return obj
    return {}


def _anomaly_findings(stage_a: Dict[str, Any], result: Dict[str, Any]) -> List[Dict[str, Any]]:
    audit = result.get("anomaly_audit") or stage_a.get("anomaly_audit") or stage_a.get("identified_anomaly") or {}
    if isinstance(audit, dict):
        for key in ("findings", "anomalies", "items"):
            if isinstance(audit.get(key), list):
                return [x for x in audit[key] if isinstance(x, dict)]
        if audit.get("artifact") or audit.get("anomalous_artifact"):
            return [audit]
    if isinstance(audit, list):
        return [x for x in audit if isinstance(x, dict)]
    return []


def _truth_anomalies(instance_truth: Dict[str, Any]) -> List[Dict[str, Any]]:
    values = instance_truth.get("anomalies")
    if isinstance(values, list):
        return [x for x in values if isinstance(x, dict)]
    if instance_truth.get("anomaly_artifact"):
        return [{"artifact":instance_truth.get("anomaly_artifact"),"anomaly_type":instance_truth.get("anomaly_type"),"anomaly_reason":instance_truth.get("anomaly_reason")}]
    return []


def _normalise_anomaly_type(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _reference_adjacency(mol) -> Dict[int, set[int]]:
    out: Dict[int, set[int]] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        out[atom.GetIdx()] = {n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() != 1}
    return out


def _xyz_heavy_adjacency(path: Path) -> tuple[List[str], Dict[int, set[int]]]:
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    block = path.read_text(encoding="utf-8", errors="replace")
    mol = Chem.MolFromXYZBlock(block)
    if mol is None:
        raise ValueError("RDKit could not parse XYZ")
    rdDetermineBonds.DetermineConnectivity(mol)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    return symbols, _reference_adjacency(mol)


def _endpoint_connectivity(reference_sdf: Path, endpoint_xyz: Path) -> bool:
    ref = _parse_molecule(reference_sdf)
    if ref is None:
        return False
    ref_symbols = [a.GetSymbol() for a in ref.GetAtoms()]
    xyz_symbols, xyz_adj = _xyz_heavy_adjacency(endpoint_xyz)
    if ref_symbols != xyz_symbols:
        return False
    return _reference_adjacency(ref) == xyz_adj


def _circular_sector_count(angles: List[float], sector_width: float = 60.0) -> int:
    sectors = {int(((a + 180.0) % 360.0) // sector_width) for a in angles}
    return len(sectors)


def _stage_bypass_events(tool_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for idx, row in enumerate(tool_rows, start=1):
        rank = STAGE_RANK.get(str(row.get("stage")), 0)
        result = row.get("result") or {}
        changed = set(result.get("created_files") or []) | set(result.get("modified_files") or [])
        future = [name for name, file_rank in STAGE_FILE_RANK.items() if name in changed and file_rank > rank]
        if future:
            events.append({"tool_index":idx,"runner_stage":row.get("stage"),"future_stage_files":future})
    return events


def score_rules(
    run_dir: Path,
    workspace: Path,
    result: Dict[str, Any] | None = None,
    *,
    instance_truth: Dict[str, Any] | None = None,
    task_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    result = result or {}
    instance = _load_json(workspace / "instance.json", {}) or {}
    stage_a = _load_json(workspace / "stage_A_audit.json", {}) or {}
    stage_b = _load_json(workspace / "stage_B_bulk.json", {}) or {}
    stage_c = _load_json(workspace / "stage_C_local.json", {}) or {}
    plan = _load_json(workspace / "plan.json", {}) or {}
    revisions = _load_json(workspace / "revisions.json", None)
    task_cfg = task_cfg or {}
    instance_truth = instance_truth or _infer_anomaly_truth(workspace, instance)
    reference = instance_truth.get("scoring_reference") or {}

    breakdown: Dict[str, float] = {
        "input_audit_and_identity": 0.0,
        "conformer_search_and_design": 0.0,
        "fixed_geometry_smd": 0.0,
        "endpoint_geometry_and_lineage": 0.0,
        "sapt_evidence_and_dominance": 0.0,
        "workflow_integrity_and_reproducibility": 0.0,
    }
    details: Dict[str, Any] = {}
    warnings: List[str] = []
    errors: List[str] = []
    fabrication_signals: List[str] = []

    missing_required = [name for name in REQUIRED_STAGE_FILES if not (workspace / name).exists()]
    schema_errors: List[str] = []
    if result and (workspace / "result_schema.json").exists():
        try:
            import jsonschema
            jsonschema.validate(result, _load_json(workspace / "result_schema.json", {}))
        except Exception as exc:
            schema_errors.append(str(exc))

    # A1. Oligomer identity — 3 points.
    structure_path = _candidate_structure_path(workspace, result, stage_a)
    mapping = _mapping_indices(workspace)
    mol_detail: Dict[str, Any] = {}
    candidate = None
    expected = None
    try:
        expected = _construct_expected_molecule(instance)
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        expected_smiles = Chem.MolToSmiles(expected, canonical=True)
        expected_formula = rdMolDescriptors.CalcMolFormula(Chem.AddHs(expected))
        if structure_path:
            candidate = _parse_molecule(structure_path)
        if candidate is not None:
            cand_smiles = Chem.MolToSmiles(Chem.RemoveHs(candidate), canonical=True)
            cand_formula = rdMolDescriptors.CalcMolFormula(candidate)
            graph_ok = cand_smiles == expected_smiles and cand_formula == expected_formula
            identity_ok = Chem.GetFormalCharge(candidate) == int(instance.get("molecule", {}).get("charge", 0)) and int(instance.get("molecule", {}).get("multiplicity", 1)) == 1 and len(Chem.GetMolFrags(candidate)) == 1 and all(a.GetAtomicNum() != 0 for a in candidate.GetAtoms())
            if graph_ok:
                breakdown["input_audit_and_identity"] += 1.5
            if identity_ok:
                breakdown["input_audit_and_identity"] += 1.5
            mol_detail = {"candidate_smiles":cand_smiles,"expected_smiles":expected_smiles,"candidate_formula":cand_formula,"expected_formula":expected_formula,"graph_ok":graph_ok,"charge_connectivity_ok":identity_ok}
    except Exception as exc:
        errors.append(f"molecular identity check failed: {exc}")
    details["molecular_identity"] = mol_detail

    # A2. Mapping — 3 points.
    if candidate is not None:
        n = candidate.GetNumAtoms()
        inter = _index_list(mapping, ["inter_repeat_bond", "inter_repeat_atoms"], 2)
        dih = _index_list(mapping, ["inter_repeat_dihedral", "s_c_c_s_dihedral", "sccs_dihedral"], 4)
        sulf = _index_list(mapping, ["sulfur_atoms"], 2)
        term = _index_list(mapping, ["terminal_carbons", "terminal_carbon_atoms"], 2)
        if all(x is not None and all(0 <= i < n for i in x) for x in (inter, dih, sulf, term)):
            breakdown["input_audit_and_identity"] += 1
            try:
                bond = candidate.GetBondBetweenAtoms(inter[0], inter[1]) if inter else None
                if bond is not None:
                    breakdown["input_audit_and_identity"] += 1
                if sulf and all(candidate.GetAtomWithIdx(i).GetAtomicNum() == 16 for i in sulf) and term and all(candidate.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in term):
                    breakdown["input_audit_and_identity"] += 1
            except Exception:
                pass

    # A3. Four solvent identities — 2 points, 0.5 each.
    solvent_audit = _reported_solvent_audit(stage_a, result)
    solvent_scores: Dict[str, Any] = {}
    ref_solvents = reference.get("solvents") or {}
    for name in FORMAL_SOLVENTS[1:]:
        entry = solvent_audit.get(name) if isinstance(solvent_audit, dict) else None
        ref = ref_solvents.get(name) or {}
        score = 0.0
        observed: Dict[str, Any] = {}
        if isinstance(entry, dict) and ref:
            try:
                observed = _canonical_solvent_entry(entry)
                if observed.get("valid") and observed.get("canonical_smiles") == ref.get("canonical_smiles") and observed.get("formula") == ref.get("formula"):
                    score += 0.25
                if observed.get("smd_name") == ref.get("smd_name") and observed.get("xtb_alpb_name") == ref.get("xtb_alpb_name"):
                    score += 0.25
            except Exception as exc:
                observed = {"error":str(exc)}
        breakdown["input_audit_and_identity"] += score
        solvent_scores[name] = {"score":score,"observed":observed,"expected":ref}
    details["solvent_identity"] = solvent_scores

    # A4. Hidden multi-anomaly audit — 8 points.
    truth_anomalies = _truth_anomalies(instance_truth)
    findings = _anomaly_findings(stage_a, result)
    matched_findings: set[int] = set()
    anomaly_rows: List[Dict[str, Any]] = []
    artifact_hits = type_hits = evidence_hits = isolated_hits = 0
    for truth in truth_anomalies:
        ta = _norm_path(truth.get("artifact"))
        tt = _normalise_anomaly_type(truth.get("anomaly_type"))
        match_idx = None
        for idx, finding in enumerate(findings):
            fa = _norm_path(finding.get("artifact") or finding.get("anomalous_artifact"))
            if fa and ta and (fa == ta or Path(fa).name == Path(ta).name):
                match_idx = idx; break
        row = {"truth":truth,"matched":False}
        if match_idx is not None:
            matched_findings.add(match_idx); artifact_hits += 1
            finding = findings[match_idx]
            ft = _normalise_anomaly_type(finding.get("anomaly_type") or finding.get("type"))
            if ft == tt:
                type_hits += 1
            evidence_text = _text_blob(finding.get("evidence"), finding.get("reason"), finding.get("observation"))
            if len(evidence_text.strip()) >= 20:
                evidence_hits += 1
            contaminated = _anomaly_contaminated(result, stage_a, stage_b, stage_c, ta or "")
            action_text = _text_blob(finding.get("action"), finding)
            if not contaminated and any(x in action_text for x in ("exclude","repair","regener","reject","not use","quarantine")):
                isolated_hits += 1
            row.update({"matched":True,"finding":finding,"type_match":ft==tt,"contaminated":contaminated})
        anomaly_rows.append(row)
    denom = max(len(truth_anomalies), 1)
    breakdown["input_audit_and_identity"] += 2.0 * artifact_hits / denom
    breakdown["input_audit_and_identity"] += 2.0 * type_hits / denom
    breakdown["input_audit_and_identity"] += 2.0 * evidence_hits / denom
    breakdown["input_audit_and_identity"] += 1.0 * isolated_hits / denom
    false_positives = max(0, len(findings) - len(matched_findings))
    if truth_anomalies and false_positives == 0:
        breakdown["input_audit_and_identity"] += 1.0
    details["anomaly_audit"] = {"truth_count":len(truth_anomalies),"reported_count":len(findings),"artifact_hits":artifact_hits,"type_hits":type_hits,"evidence_hits":evidence_hits,"isolated_hits":isolated_hits,"false_positives":false_positives,"rows":anomaly_rows}

    # B. Conformer search and design — 12 points.
    search = stage_b.get("gas_conformer_search") or {}
    attempts = search.get("attempts") or []
    seed_angles: List[float] = []
    actual_seed_files = 0
    if isinstance(attempts, list):
        for item in attempts:
            if not isinstance(item, dict): continue
            path = _resolve(workspace, item.get("starting_geometry_file") or item.get("input_geometry") or item.get("seed_geometry_file"))
            angle = _as_float(item.get("initial_torsion_deg") or item.get("torsion_start_deg"))
            if path and path.exists(): actual_seed_files += 1
            if angle is not None: seed_angles.append(angle)
    if actual_seed_files >= 4:
        breakdown["conformer_search_and_design"] += 1
    if len(seed_angles) >= 4 and _circular_sector_count(seed_angles) >= 4:
        breakdown["conformer_search_and_design"] += 2
    retained = search.get("retained_conformers") or []
    if isinstance(retained, list) and len(retained) >= 2:
        breakdown["conformer_search_and_design"] += 1.5
        selected = search.get("selected_R_g") or {}
        if selected.get("geometry_file") and (stage_b.get("sensitivity_calculation") or {}).get("type"):
            breakdown["conformer_search_and_design"] += 1.5
    sensitivity = stage_b.get("sensitivity_calculation") or {}
    sens_logs = [_resolve(workspace, p) for p in (sensitivity.get("log_files") or [])]
    sens_logs = [p for p in sens_logs if p and p.exists()]
    rationale = _text_blob(sensitivity.get("rationale"), sensitivity)
    if sens_logs:
        breakdown["conformer_search_and_design"] += 1
        if any(x in rationale for x in ("conformer","grid","guess","density","uncert","ranking")):
            breakdown["conformer_search_and_design"] += 2
    resource_text = _text_blob(plan, stage_a)
    resource_tokens = sum(x in resource_text for x in ("thread","memory","parallel","timeout","scratch","failure","independent job","separate directory"))
    if resource_tokens >= 3: breakdown["conformer_search_and_design"] += 1.5
    if resource_tokens >= 5: breakdown["conformer_search_and_design"] += 1.5
    details["conformer_design"] = {"actual_seed_files":actual_seed_files,"seed_angles":seed_angles,"sector_count":_circular_sector_count(seed_angles) if seed_angles else 0,"retained_count":len(retained) if isinstance(retained,list) else 0,"sensitivity_logs":[str(p.relative_to(workspace)) for p in sens_logs],"resource_tokens":resource_tokens}

    # C. Fixed common-geometry SMD — 30 points.
    formal = _formal_entries(result, stage_b, workspace)
    parsed_formal: Dict[str, Dict[str, Any]] = {}
    digits = int(instance.get("coordinate_hash_spec", {}).get("round_digits", 10))
    rg_path = workspace / "structures" / "R_g.xyz"
    rg_hash = _coordinate_hash(rg_path, digits) if rg_path.exists() else None
    geometry_hashes: Dict[str, str] = {}
    metadata_ok: Dict[str, bool] = {}
    for name in FORMAL_SOLVENTS:
        info = formal.get(name) or {}
        lp = info.get("log_path")
        if lp and lp.exists():
            try:
                parsed = _parse_formal_log(lp); parsed_formal[name] = parsed
                meta = parsed.get("metadata") or {}
                metadata_ok[name] = bool(parsed.get("converged") and str(meta.get("program","")).lower()=="pyscf" and str(meta.get("functional","")).lower().replace("-","")=="m052x" and str(meta.get("basis","")).lower().replace(" ","")=="6-31g*" and str(meta.get("solvent","")).lower()==name)
                gp = _resolve(workspace, meta.get("geometry_file")) or info.get("geometry_path")
                if gp and gp.exists(): geometry_hashes[name] = _coordinate_hash(gp,digits)
            except Exception as exc:
                warnings.append(f"formal parse failed for {name}: {exc}")
    if rg_hash and len(geometry_hashes)==5 and set(geometry_hashes.values())=={rg_hash}:
        breakdown["fixed_geometry_smd"] += 2
    if sum(metadata_ok.values()) == 5:
        breakdown["fixed_geometry_smd"] += 2
    smd_ref = reference.get("smd") or {}
    energy_ref = smd_ref.get("energy_hartree") or {}
    energy_tol = float(smd_ref.get("energy_tolerance_hartree",5e-6))
    energy_matches: Dict[str, bool] = {}
    for name in FORMAL_SOLVENTS:
        observed = _as_float((parsed_formal.get(name) or {}).get("energy_hartree"))
        expected_value = _as_float(energy_ref.get(name))
        ok = observed is not None and expected_value is not None and metadata_ok.get(name,False) and abs(observed-expected_value)<=energy_tol
        energy_matches[name]=ok
        if observed is not None:
            breakdown["fixed_geometry_smd"] += 1.2
    gas = _as_float((parsed_formal.get("gas") or {}).get("energy_hartree"))
    factor = float(instance.get("constants",{}).get("hartree_to_kcal_mol",627.509474))
    deltas: Dict[str,float] = {}
    if gas is not None:
        for name in FORMAL_SOLVENTS[1:]:
            e = _as_float((parsed_formal.get(name) or {}).get("energy_hartree"))
            if e is not None: deltas[name]=(e-gas)*factor
    delta_ref = smd_ref.get("delta_kcal_mol") or {}
    delta_tol = float(smd_ref.get("delta_tolerance_kcal_mol",0.03))
    delta_matches: Dict[str,bool] = {}
    for name in FORMAL_SOLVENTS[1:]:
        ok = name in deltas and _as_float(delta_ref.get(name)) is not None and abs(deltas[name]-float(delta_ref[name]))<=delta_tol
        delta_matches[name]=ok
        if ok: breakdown["fixed_geometry_smd"] += 2
    observed_order = sorted(deltas,key=deltas.get) if len(deltas)==4 else []
    expected_order = list(smd_ref.get("ascending_delta_order") or [])
    if observed_order == expected_order:
        breakdown["fixed_geometry_smd"] += 8
    bulk = result.get("bulk_solvation") or stage_b.get("ranking") or stage_b
    reported_lowest = _extract_ranking_value(bulk,"lowest_solvation_free_energy_solvent")
    expected_lowest = str(smd_ref.get("lowest_solvation_free_energy_solvent") or "")
    orbital_complete = sum(all(_as_float((parsed_formal.get(n) or {}).get(k)) is not None for k in ("homo_hartree","lumo_hartree")) for n in FORMAL_SOLVENTS)
    dipole_complete = sum(all(_as_float((parsed_formal.get(n) or {}).get(k)) is not None for k in ("dipole_x_debye","dipole_y_debye","dipole_z_debye")) for n in FORMAL_SOLVENTS)
    breakdown["fixed_geometry_smd"] += 2.0 * orbital_complete / 5.0
    breakdown["fixed_geometry_smd"] += 2.0 * dipole_complete / 5.0
    details["formal_recompute"]={"energies_hartree":{n:(parsed_formal.get(n) or {}).get("energy_hartree") for n in FORMAL_SOLVENTS},"energy_matches_diagnostic_only":energy_matches,"deltas_kcal_mol":deltas,"delta_matches":delta_matches,"observed_order":observed_order,"expected_order":expected_order,"reported_lowest_solvation_free_energy_solvent":reported_lowest,"expected_lowest_solvation_free_energy_solvent":expected_lowest,"coordinate_hashes":geometry_hashes,"metadata_ok":metadata_ok}

    # D. Endpoint geometry and evidence lineage — 10 points.
    selected_files = _selected_endpoint_files(result, stage_c, workspace)
    ref_sdf = structure_path or workspace/"structures/solute_validated.sdf"
    connectivity: Dict[str,bool] = {}
    for endpoint in ENDPOINTS:
        p = selected_files.get(endpoint)
        ok = False
        if p and p.exists() and ref_sdf.exists():
            try: ok = _endpoint_connectivity(ref_sdf,p)
            except Exception as exc: warnings.append(f"{endpoint} connectivity check failed: {exc}")
        connectivity[endpoint]=ok
        if ok: breakdown["endpoint_geometry_and_lineage"] += 1.5
    geometry_recompute: Dict[str,Any] = {}
    if all(e in selected_files for e in ENDPOINTS):
        try:
            sh,ch=_read_xyz(selected_files["n-hexane"]); sw,cw=_read_xyz(selected_files["water"])
            heavy=[i for i,s in enumerate(sh) if s.upper()!="H"]
            dihedral_idx=_index_list(mapping,["inter_repeat_dihedral","s_c_c_s_dihedral","sccs_dihedral"],4)
            terminal_idx=_index_list(mapping,["terminal_carbons","terminal_carbon_atoms"],2)
            if sh==sw and dihedral_idx and terminal_idx:
                ph=_dihedral(ch,dihedral_idx); pw=_dihedral(cw,dihedral_idx)
                th=min(abs(ph),abs(180-abs(ph))); tw=min(abs(pw),abs(180-abs(pw)))
                geometry_recompute={"heavy_atom_rmsd_angstrom":_kabsch_rmsd(ch[heavy],cw[heavy]),"twist_difference_deg":abs(tw-th),"end_to_end_difference_angstrom":abs(float(np.linalg.norm(cw[terminal_idx[0]]-cw[terminal_idx[1]]))-float(np.linalg.norm(ch[terminal_idx[0]]-ch[terminal_idx[1]])))}
                breakdown["endpoint_geometry_and_lineage"] += 1
                reported = stage_c.get("endpoint_comparison") or (result.get("conformational_analysis") or {}).get("comparison") or {}
                match=0
                for k,v in geometry_recompute.items():
                    rv=_as_float(reported.get(k))
                    tol=1.0 if "deg" in k else 0.02
                    if rv is not None and abs(rv-v)<=tol: match+=1
                breakdown["endpoint_geometry_and_lineage"] += 2.0*match/3.0
        except Exception as exc: warnings.append(f"endpoint metric check failed: {exc}")
    endpoints = stage_c.get("endpoints") or {}
    lineage_ok=0; diversity_ok=0
    for endpoint in ENDPOINTS:
        item=endpoints.get(endpoint) or {}
        selected_rel=_norm_path(item.get("selected_structure_file"))
        complexes=item.get("explicit_complexes") or []
        if selected_rel and isinstance(complexes,list) and complexes and all(_norm_path(c.get("source_endpoint_geometry"))==selected_rel for c in complexes if isinstance(c,dict)):
            lineage_ok+=1
        files=[]; separated=False
        for c in complexes if isinstance(complexes,list) else []:
            if not isinstance(c,dict): continue
            p=_resolve(workspace,c.get("geometry_file"))
            if p and p.exists(): files.append(_coordinate_hash(p,digits))
            if _as_float(c.get("frozen_interaction_energy_kcal_mol")) is not None and _as_float(c.get("deformation_energy_kcal_mol")) is not None: separated=True
        if len(set(files))>=3 and separated: diversity_ok+=1
    breakdown["endpoint_geometry_and_lineage"] += 1.0*lineage_ok/2.0*2.0
    breakdown["endpoint_geometry_and_lineage"] += 1.0*diversity_ok/2.0*2.0
    details["endpoint_geometry"]={"connectivity":connectivity,"geometry_recompute":geometry_recompute,"lineage_ok_count":lineage_ok,"diversity_and_energy_separation_count":diversity_ok}

    # E. SAPT dominant-attraction analysis — 14 points.
    # No SAPT component or total is compared with a private numerical tolerance.
    # Protocol, fragment identity, and geometry lineage are eligibility gates only;
    # the sole scored result is the correctly supported dominant attractive component.
    sapt_items = _sapt_items(result,stage_c,workspace)
    sapt_ref = reference.get("sapt") or {}
    sapt_details: Dict[str,Any] = {}
    for endpoint in ENDPOINTS:
        info=sapt_items.get(endpoint) or {}; lp=info.get("log_path")
        endpoint_detail={"evidence_eligible":False,"dominance_score":0.0}
        if lp and lp.exists():
            try:
                parsed=_parse_sapt_log(lp); meta=parsed.get("metadata") or {}
                components=[_as_float(parsed.get(k)) for k in ("electrostatics","exchange","induction","dispersion")]
                parsed_ok=all(v is not None for v in components) and _as_float(parsed.get("total")) is not None
                protocol_ok=(
                    str(meta.get("program","")).lower()=="psi4"
                    and str(meta.get("method","")).lower()=="sapt0"
                    and str(meta.get("basis","")).lower()=="jun-cc-pvdz"
                    and str(meta.get("solvent","")).lower() in {endpoint,"hexane" if endpoint=="n-hexane" else endpoint}
                )
                gp=_resolve(workspace,meta.get("geometry_file")); fa=_resolve(workspace,meta.get("fragment_a_geometry")); fb=_resolve(workspace,meta.get("fragment_b_geometry"))
                rep=((stage_c.get("endpoints") or {}).get(endpoint) or {}).get("representative") or {}
                lineage_ok=(
                    gp is not None and gp.exists() and fa is not None and fa.exists() and fb is not None and fb.exists()
                    and _same_xyz_evidence(workspace, rep.get("complex_geometry_file"), meta.get("geometry_file"))
                    and _same_xyz_evidence(workspace, rep.get("monomer_a_xyz"), meta.get("fragment_a_geometry"))
                    and _same_xyz_evidence(workspace, rep.get("monomer_b_xyz"), meta.get("fragment_b_geometry"))
                )
                endpoint_detail["evidence_eligible"]=bool(parsed_ok and protocol_ok and lineage_ok)
                raw_dom=_dominant_attractive(parsed)
                reported_dom=_reported_dominance(info.get("sapt") or {}) or _reported_dominance(info.get("item") or {})
                attractive = {
                    k: abs(float(parsed[k])) for k in ("electrostatics", "induction", "dispersion")
                    if _as_float(parsed.get(k)) is not None and float(parsed[k]) < 0
                }
                accepted_dominance = {raw_dom} if raw_dom else set()
                if attractive:
                    best = max(attractive.values())
                    accepted_dominance |= {k for k, magnitude in attractive.items() if best - magnitude <= 0.5}
                if endpoint_detail["evidence_eligible"] and reported_dom in accepted_dominance:
                    endpoint_detail["dominance_score"]=7.0
                if reported_dom and raw_dom and reported_dom!=raw_dom:
                    fabrication_signals.append(f"{endpoint}: reported SAPT dominance contradicts raw components")
                endpoint_detail.update({
                    "parsed_ok":parsed_ok,"protocol_ok":protocol_ok,"lineage_ok":lineage_ok,
                    "parsed_components":{k:parsed.get(k) for k in ("electrostatics","exchange","induction","dispersion","total")},
                    "raw_dominance":raw_dom,"reported_dominance":reported_dom,"accepted_dominance":sorted(accepted_dominance),
                })
            except Exception as exc: warnings.append(f"SAPT parse failed for {endpoint}: {exc}")
        breakdown["sapt_evidence_and_dominance"] += endpoint_detail["dominance_score"]
        sapt_details[endpoint]=endpoint_detail
    details["sapt_recompute"]=sapt_details

    # F. Workflow integrity and reproducibility — 8 points.
    tool_rows=_read_jsonl(run_dir/"tool_calls.jsonl")
    mutations=_stage_mutations(run_dir)
    bypass=_stage_bypass_events(tool_rows)
    provenance_text=_text_blob(stage_a.get("provenance"),stage_b.get("provenance"),stage_c.get("provenance"),result.get("provenance"),_load_json(workspace/"analysis/accepted_input_hashes.json",{}))
    if "sha256" in provenance_text or "hash" in provenance_text: breakdown["workflow_integrity_and_reproducibility"] += 1
    if any(x in provenance_text for x in ("protocol","version","program")): breakdown["workflow_integrity_and_reproducibility"] += 1
    if not mutations and (workspace/"stage_B_bulk.json").exists(): breakdown["workflow_integrity_and_reproducibility"] += 2
    if isinstance(revisions,list):
        if all(isinstance(x,dict) for x in revisions): breakdown["workflow_integrity_and_reproducibility"] += 1
    if not missing_required and not schema_errors: breakdown["workflow_integrity_and_reproducibility"] += 1
    model_errors=sum(bool((r.get("result") or {}).get("model_caused_error_suspected") or (r.get("result") or {}).get("error") or (r.get("result") or {}).get("protected_file_violation")) for r in tool_rows)
    if model_errors==0 and tool_rows: breakdown["workflow_integrity_and_reproducibility"] += 2
    elif model_errors==1 and not missing_required: breakdown["workflow_integrity_and_reproducibility"] += 1
    details["workflow_integrity"]={"stage_mutations":mutations,"stage_bypass_events":bypass,"model_errors":model_errors,"missing_required_files":missing_required,"schema_errors":schema_errors}

    legacy_maxima={"input_audit_and_identity":16.0,"conformer_search_and_design":12.0,"fixed_geometry_smd":30.0,"endpoint_geometry_and_lineage":10.0,"sapt_evidence_and_dominance":14.0,"workflow_integrity_and_reproducibility":8.0}
    hard_maxima={"input_audit_and_identity":10.0,"conformer_search_and_design":10.0,"fixed_geometry_smd":20.0,"endpoint_geometry_and_lineage":8.0,"sapt_evidence_and_dominance":12.0,"workflow_integrity_and_reproducibility":5.0}
    for k,m in hard_maxima.items():
        legacy=min(legacy_maxima[k],max(0.0,breakdown[k])); breakdown[k]=round(m*legacy/legacy_maxima[k],3)
    ensemble_score=0.0; ensemble_details={}; stage_c_endpoints=(stage_c.get("endpoints") or {}) if isinstance(stage_c,dict) else {}
    for endpoint in ENDPOINTS:
        entries=(stage_c_endpoints.get(endpoint) or {}).get("sapt_ensemble") or []; valid_entries=[]; geometries=set()
        for entry in entries if isinstance(entries,list) else []:
            raw=_resolve(workspace,entry.get("raw_log")); geometry=str(entry.get("complex_geometry_file") or "")
            if raw is None or not raw.is_file() or not geometry or geometry in geometries: continue
            parsed=_parse_sapt_log(raw); reported=_reported_dominance(entry)
            if _dominant_attractive(parsed)==reported: valid_entries.append(entry); geometries.add(geometry)
        points=10.0 if len(valid_entries)>=2 else 0.0; ensemble_score+=points; ensemble_details[endpoint]={"valid_distinct_entries":len(valid_entries),"score":points}
    breakdown["multi_configuration_sapt"]=ensemble_score; details["multi_configuration_sapt"]=ensemble_details
    cross=result.get("cross_layer_analysis") or {}; cross_keys=("fixed_vs_relaxed_geometry","implicit_vs_explicit_solvent","frozen_vs_deformation_energy","sapt_geometry_dependence","experimental_scope_limit")
    cross_hits=sum(isinstance(cross.get(k),str) and len(cross.get(k).strip())>=40 for k in cross_keys)
    breakdown["cross_layer_consistency"]=3.0*cross_hits; details["cross_layer_consistency"]={"satisfied_items":cross_hits,"required_items":len(cross_keys)}

    # Hard caps are integrity safeguards, not duplicate positive scores.
    hard_caps: List[Dict[str,Any]]=[]
    if schema_errors: hard_caps.append({"cap":30,"reason":"result.json does not validate against the bundled schema"})
    if fabrication_signals: hard_caps.append({"cap":25,"reason":"raw evidence contradicts one or more reported claims"})
    contaminated_truth = any(row.get("contaminated") for row in anomaly_rows)
    if result.get("task_complete") and contaminated_truth: hard_caps.append({"cap":55,"reason":"a hidden anomalous artifact contaminated a downstream branch claimed as validated"})
    if mutations: hard_caps.append({"cap":60,"reason":"an earlier stage file was semantically modified after a later stage began"})
    if bypass: hard_caps.append({"cap":60,"reason":"future-stage completion output was created before the runner entered that stage"})
    # Microtest ordering remains an integrity gate but has no positive points.
    micro_at: Dict[str,int]={}; prod_at: Dict[str,int]={}
    for idx,row in enumerate(tool_rows,start=1):
        rr=row.get("result") or {}
        for tool in rr.get("micro_tools") or []: micro_at.setdefault(tool,idx)
        for tool in rr.get("production_tools") or []: prod_at.setdefault(tool,idx)
    tool_order_ok=all(tool in micro_at and micro_at[tool]<idx for tool,idx in prod_at.items())
    compute_budget=instance.get("compute_budget") or {}
    budget_enabled=bool(compute_budget.get("enabled",False)); tool_count=len(tool_rows)
    preferred=compute_budget.get("preferred_tool_calls"); maximum=compute_budget.get("max_tool_calls")
    if not tool_order_ok: hard_caps.append({"cap":70,"reason":"production API used before its micro-test"})

    subtotal=round(sum(breakdown.values()),3)
    penalties=compute_budget.get("tool_call_penalties") or {}
    penalty=0.0
    if budget_enabled and preferred is not None and tool_count>int(preferred):
        penalty=float(penalties.get(str(tool_count),penalties.get(tool_count,0.0)))
    after_penalty=round(max(0.0,subtotal-penalty),3)
    cap=min((x["cap"] for x in hard_caps),default=100)
    score=min(after_penalty,cap)
    details["fabrication_signals"]=fabrication_signals
    details["tool_calls"]={"count":tool_count,"preferred":preferred,"maximum":maximum,"penalty":penalty,"microtest_at":micro_at,"production_at":prod_at,"order_ok":tool_order_ok}
    return {"score":score,"max_score":100,"raw_score_before_caps":after_penalty,"raw_score_before_tool_call_penalty":subtotal,"tool_call_penalty":penalty,"breakdown":breakdown,"hard_caps":hard_caps,"errors":errors,"warnings":warnings,"details":details}
