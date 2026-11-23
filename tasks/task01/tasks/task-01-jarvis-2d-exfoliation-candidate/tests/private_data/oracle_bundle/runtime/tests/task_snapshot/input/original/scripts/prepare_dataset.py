from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data
from pymatgen.core import Structure


def valid_number(value: Any) -> bool:
    try:
        return value not in (None, "", "na", "nan") and np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def exact_structure_hash(structure_dict: dict) -> str:
    payload = json.dumps(structure_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def serialize_structure(atoms_dict: dict) -> dict:
    atoms = Atoms.from_dict(atoms_dict)
    return atoms.pymatgen_converter().as_dict()


def translated_reordered_variant(structure_dict: dict) -> dict:
    s = Structure.from_dict(structure_dict)
    s.translate_sites(range(len(s)), [0.173, 0.291, 0.0], frac_coords=True, to_unit_cell=True)
    order = list(reversed(range(len(s))))
    site_properties = {key: [values[i] for i in order] for key, values in s.site_properties.items()}
    out = Structure(
        s.lattice,
        [s[i].species for i in order],
        [s[i].frac_coords for i in order],
        coords_are_cartesian=False,
        site_properties=site_properties or None,
    )
    return out.as_dict()


def in_plane_supercell_variant(structure_dict: dict) -> dict:
    s = Structure.from_dict(structure_dict).copy()
    s.make_supercell([[2, 0, 0], [0, 1, 0], [0, 0, 1]])
    return s.as_dict()


def choose_ood_systems(df: pd.DataFrame, target_rows: int, rng: np.random.Generator) -> set[str]:
    counts = df.groupby("chemical_system").size().to_dict()
    systems = np.array(sorted(counts), dtype=object)
    rng.shuffle(systems)
    selected: set[str] = set()
    current = 0
    for system in systems:
        if current >= target_rows:
            break
        selected.add(str(system))
        current += int(counts[str(system)])
    return selected


def choose_group_holdout_systems(df: pd.DataFrame, target_rows: int, rng: np.random.Generator) -> set[str]:
    """Choose complete chemical-system groups for group-OOD holdout."""
    counts = df.groupby("chemical_system").size().sort_values()
    systems = np.array(list(counts.index), dtype=object)
    rng.shuffle(systems)
    selected: set[str] = set()
    current = 0
    for system in systems:
        count = int(counts.loc[system])
        if current >= target_rows and selected:
            break
        selected.add(str(system))
        current += count
    return selected


def choose_iid_rows_with_train_support(df: pd.DataFrame, target_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Choose IID rows while leaving at least one same-system row outside holdout."""
    counts = df.groupby("chemical_system").size()
    eligible_systems = np.array(sorted(counts[counts >= 2].index), dtype=object)
    rng.shuffle(eligible_systems)
    chosen: list[int] = []
    for system in eligible_systems:
        if len(chosen) >= target_rows:
            break
        idx = df.index[df.chemical_system == system].to_numpy()
        chosen.append(int(rng.choice(idx)))
    if len(chosen) < target_rows:
        raise SystemExit(
            f"could not construct IID split with train support: requested {target_rows}, got {len(chosen)}"
        )
    return np.array(chosen, dtype=int)


def assign_opaque_test_ids(test_public: pd.DataFrame, labels: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace all public test IDs with deterministic opaque IDs."""
    test_public = test_public.copy()
    labels = labels.copy()
    rng = np.random.default_rng(seed)
    row_uids = test_public["_row_uid"].astype(str).to_numpy()
    order = np.arange(len(row_uids))
    rng.shuffle(order)
    opaque_by_uid = {
        str(row_uids[i]): f"J2D-T-{rank + 1:06d}"
        for rank, i in enumerate(order)
    }
    test_public["sample_id"] = test_public["_row_uid"].astype(str).map(opaque_by_uid)
    labels["sample_id"] = labels["_row_uid"].astype(str).map(opaque_by_uid)
    if test_public["sample_id"].isna().any() or labels["sample_id"].isna().any():
        raise RuntimeError("internal opaque sample_id mapping failed")
    return test_public.drop(columns=["_row_uid"]), labels.drop(columns=["_row_uid"])



def build_candidate_constraints(elements: list[str]) -> dict:
    return {
        "version": "candidate-library-score-v2-chgnet-selected",
        "required_candidate_count": 1000,
        "candidate_id_pattern": "^CAND-[0-9]{6}$",
        "min_sites": 2,
        "max_sites": 40,
        "max_distinct_elements": 4,
        "ordered_structures_only": True,
        "allowed_elements": sorted(elements),
        "allowed_elements_provenance": {
            "source": "union of elements in the frozen JARVIS 2D source corpus before benchmark splitting",
            "selection_rule": "element must occur in the source corpus and be supported by the frozen CHGNet model",
        },
        "pathology": {
            "minimum_absolute_distance_angstrom": 0.6,
            "minimum_covalent_radius_ratio": 0.5,
            "maximum_lattice_condition_number": 200.0,
            "coordination_cutoff_angstrom": 5.0,
            "maximum_neighbors_per_site": 80,
            "minimum_vacuum_gap_angstrom": 6.0,
        },
        "structure_matcher": {
            "ltol": 0.2, "stol": 0.3, "angle_tol": 5.0,
            "primitive_cell": True, "scale": True, "attempt_supercell": True,
            "allow_subset": False,
        },
        "geometry": {
            "covalent_contact_ratio_zero": 0.50,
            "covalent_contact_ratio_full": 0.65,
            "vacuum_gap_zero_angstrom": 6.0,
            "vacuum_gap_full_angstrom": 10.0,
            "layer_thickness_full_max_angstrom": 15.0,
            "layer_thickness_zero_angstrom": 25.0,
            "lattice_length_min_angstrom": 1.5,
            "lattice_length_max_angstrom": 60.0,
            "lattice_angle_min_degree": 20.0,
            "lattice_angle_max_degree": 160.0,
            "condition_number_full_max": 50.0,
            "condition_number_zero": 200.0,
            "bond_radius_scale": 1.25,
            "connectivity_fraction_zero": 0.50,
            "connectivity_fraction_full": 0.90,
        },
        "chemistry": {
            "multi_element_max_atomic_fraction_full": 0.80,
            "multi_element_max_atomic_fraction_zero": 0.95,
            "substitution_radius_ratio_min": 0.70,
            "substitution_radius_ratio_max": 1.30,
            "shared_oxidation_state_or_same_group_is_compatible": True,
            "oxidation_state_failure_uncertain_score": 0.50,
            "all_metal_oxidation_fallback_score": 0.80,
        },
        "diversity": {
            "chemical_system_target": 100,
            "reduced_formula_target": 300,
            "prototype_target": 50,
            "chemical_system_points": 0.5,
            "reduced_formula_points": 0.25,
            "prototype_points": 0.75,
            "entropy_points": 0.5,
        },
        "reproducibility": {
            "interface_version": "candidate-generator-v1",
            "float_round_decimals": 8,
            "timeout_seconds": 300,
        },
        "scoring_weights": {
            "combined_structural_quality": 8.0,
            "internal_uniqueness": 3.0,
            "composition_structural_novelty": 3.0,
            "anonymous_prototype_novelty": 8.0,
            "composition_structure_diversity": 3.0,
            "generation_reproducibility": 5.0,
            "selected_chgnet_stability": 10.0,
            "model_prediction_traceability": 5.0,
            "uncertainty_and_applicability": 5.0,
        },
        "selected_chgnet": {
            "protocol_version": "chgnet-selected-relax-v1",
            "package_version": "0.4.2",
            "pretrained_model_version": "0.3.0",
            "required_candidate_count": 5,
            "maximum_tool_calls": 1,
            "maximum_steps": 200,
            "fmax_ev_ang": 0.05,
            "relax_cell": False,
            "low_prediction_fraction": 0.1,
            "minimum_prediction_std_mev_atom": 1e-6,
            "minimum_unique_prediction_values": 20,
        },
        "schema_failure_candidate_score_cap": 35.0,
        "score_caps": {
            "valid_below_700": 25.0,
            "unique_below_500": 20.0,
            "novel_below_300": 15.0,
        },
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--private-label-dir", required=True, help="Directory outside the agent-visible project in production.")
    ap.add_argument("--seed", type=int, default=20260715)
    ap.add_argument("--base-test-fraction", type=float, default=0.20)
    ap.add_argument("--validation-fraction", type=float, default=0.10)
    ap.add_argument("--ood-share-of-base-test", type=float, default=0.70)
    ap.add_argument("--invariance-source-fraction", type=float, default=0.25)
    ap.add_argument("--scoring-reference-json", help="Measured anchors generated by scripts/calibrate_scoring.py.")
    args = ap.parse_args()

    for name, value in {
        "base_test_fraction": args.base_test_fraction,
        "validation_fraction": args.validation_fraction,
        "ood_share_of_base_test": args.ood_share_of_base_test,
        "invariance_source_fraction": args.invariance_source_fraction,
    }.items():
        if not 0 < value < 1:
            raise SystemExit(f"{name} must be in (0, 1)")

    out = Path(args.output_dir)
    private = Path(args.private_label_dir)
    out.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    raw = data("dft_2d")
    rows = []
    excluded = {"missing_structure_or_target": 0, "conversion_failure": 0}
    for rec in raw:
        atoms_dict = rec.get("atoms")
        target = rec.get("exfoliation_energy")
        if atoms_dict is None or not valid_number(target):
            excluded["missing_structure_or_target"] += 1
            continue
        try:
            structure_dict = serialize_structure(atoms_dict)
            structure = Structure.from_dict(structure_dict)
        except Exception:
            excluded["conversion_failure"] += 1
            continue
        jid = str(rec.get("jid") or f"row-{len(rows):06d}")
        chemsys = "-".join(sorted(el.symbol for el in structure.composition.elements))
        rows.append({
            "sample_id": jid,
            "structure": structure_dict,
            "structure_hash": exact_structure_hash(structure_dict),
            "exfoliation_energy_mev_atom": float(target),
            "chemical_system": chemsys,
        })

    df = pd.DataFrame(rows)
    before = len(df)
    df = df.drop_duplicates("sample_id", keep="first")
    duplicate_id_removed = before - len(df)
    before = len(df)
    df = df.drop_duplicates("structure_hash", keep="first").reset_index(drop=True)
    exact_structure_duplicate_removed = before - len(df)
    if len(df) < 30:
        raise SystemExit("too few valid unique records to create a benchmark split")

    rng = np.random.default_rng(args.seed)
    desired_base_test = max(10, int(round(len(df) * args.base_test_fraction)))
    desired_ood = max(1, int(round(desired_base_test * args.ood_share_of_base_test)))
    ood_systems = choose_group_holdout_systems(df, desired_ood, rng)
    ood_pool = df[df.chemical_system.isin(ood_systems)].copy()
    if len(ood_pool) > desired_base_test - 1:
        # Extremely large systems cannot be a bounded group-OOD split; choose the smallest systems first.
        counts = df.groupby("chemical_system").size().sort_values()
        ood_systems = set()
        running = 0
        for system, count in counts.items():
            if running >= desired_ood:
                break
            if running + int(count) <= desired_base_test - 1:
                ood_systems.add(str(system))
                running += int(count)
        ood_pool = df[df.chemical_system.isin(ood_systems)].copy()

    remaining = df[~df.chemical_system.isin(ood_systems)].copy()
    iid_needed = desired_base_test - len(ood_pool)
    if iid_needed < 1 or iid_needed >= len(remaining):
        raise SystemExit("could not construct bounded IID/OOD test split; adjust fractions")
    iid_idx = choose_iid_rows_with_train_support(remaining, iid_needed, rng)
    iid = remaining.loc[iid_idx].copy()
    development = remaining.drop(index=iid_idx).copy()

    base_test = pd.concat([
        iid.assign(split_type="iid"),
        ood_pool.assign(split_type="chemical_system_ood"),
    ], ignore_index=True)
    base_test = base_test.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    val_n = max(5, int(round(len(development) * args.validation_fraction)))
    val_n = min(val_n, len(development) - 1)
    val_idx = choose_iid_rows_with_train_support(development, val_n, rng)
    validation = development.loc[val_idx].copy().reset_index(drop=True)
    train = development.drop(index=val_idx).copy().reset_index(drop=True)

    if set(iid.chemical_system) - set(train.chemical_system):
        raise RuntimeError("IID test split contains systems not present in train")
    if set(validation.chemical_system) - set(train.chemical_system):
        raise RuntimeError("validation split contains systems not present in train")
    if set(ood_pool.chemical_system) & (set(train.chemical_system) | set(validation.chemical_system)):
        raise RuntimeError("OOD split overlaps train/validation chemical systems")

    n_inv_sources = max(1, int(round(len(base_test) * args.invariance_source_fraction)))
    inv_source_idx = rng.choice(base_test.index.to_numpy(), size=min(n_inv_sources, len(base_test)), replace=False)
    variant_rows = []
    inv_group_by_id: dict[str, str] = {}
    for idx in inv_source_idx:
        row = base_test.loc[idx]
        group = f"inv::{len(inv_group_by_id) + 1:06d}"
        base_uid = f"base::{idx}"
        inv_group_by_id[base_uid] = group
        variants = [
            ("translated_reordered", translated_reordered_variant(row.structure)),
            ("in_plane_supercell_2x1", in_plane_supercell_variant(row.structure)),
        ]
        for variant_type, structure_dict in variants:
            variant_uid = f"variant::{idx}::{variant_type}"
            variant_rows.append({
                "_row_uid": variant_uid,
                "structure": structure_dict,
                "chemical_system": row.chemical_system,
                "true_exfoliation_energy_mev_atom": float(row.exfoliation_energy_mev_atom),
                "split_type": "invariance",
                "score_regression": False,
                "invariance_group": group,
                "variant_type": variant_type,
            })

    base_test = base_test.copy()
    base_test["_row_uid"] = [f"base::{i}" for i in base_test.index]
    base_public = base_test[["_row_uid", "structure", "chemical_system"]].copy()
    variants_public = pd.DataFrame(variant_rows)[["_row_uid", "structure", "chemical_system"]]
    test_public = pd.concat([base_public, variants_public], ignore_index=True)
    test_public = test_public.sample(frac=1.0, random_state=args.seed + 1).reset_index(drop=True)

    labels = base_test[["_row_uid", "exfoliation_energy_mev_atom", "split_type"]].rename(
        columns={"exfoliation_energy_mev_atom": "true_exfoliation_energy_mev_atom"}
    )
    labels["score_regression"] = True
    labels["invariance_group"] = labels["_row_uid"].astype(str).map(inv_group_by_id)
    labels["variant_type"] = "original"
    labels = pd.concat([
        labels,
        pd.DataFrame(variant_rows)[[
            "_row_uid", "true_exfoliation_energy_mev_atom", "split_type",
            "score_regression", "invariance_group", "variant_type",
        ]],
    ], ignore_index=True)
    test_public, labels = assign_opaque_test_ids(test_public, labels, args.seed + 2)

    public_cols = ["sample_id", "structure", "chemical_system", "exfoliation_energy_mev_atom"]
    gzip_compression = {"method": "gzip", "mtime": 0}
    train[public_cols].to_json(out / "train.json.gz", orient="records", lines=True, compression=gzip_compression)
    validation[public_cols].to_json(out / "validation.json.gz", orient="records", lines=True, compression=gzip_compression)
    test_public.to_json(out / "test_public.json.gz", orient="records", lines=True, compression=gzip_compression)
    label_path = private / "test_labels_private.json.gz"
    labels.to_json(label_path, orient="records", lines=True, compression=gzip_compression)

    all_elements = sorted({
        element
        for frame in (train, validation, test_public)
        for system in frame["chemical_system"].dropna().astype(str)
        for element in system.split("-")
        if element
    })
    candidate_constraints = build_candidate_constraints(all_elements)
    (out / "candidate_generation_constraints.json").write_text(
        json.dumps(candidate_constraints, indent=2) + "\n", encoding="utf-8"
    )

    public_hashes = {
        name: sha256(out / name)
        for name in (
            "train.json.gz", "validation.json.gz", "test_public.json.gz",
            "candidate_generation_constraints.json",
        )
    }
    snapshot_material = json.dumps(public_hashes, sort_keys=True).encode("utf-8")
    snapshot_id = "jarvis2d-exfoliation-" + hashlib.sha256(snapshot_material).hexdigest()[:16]

    if args.scoring_reference_json:
        scoring_reference = json.loads(Path(args.scoring_reference_json).read_text(encoding="utf-8"))
    else:
        scoring_reference = {
            "status": "provisional_unmeasured",
            "version": None,
            "strong_anchor_fraction": 0.65,
            "metrics": {},
            "note": "Run official floor/strong/ceiling submissions and scripts/calibrate_scoring.py before scoring.",
        }

    metadata = {
        "name": "jarvis_dft_2d_exfoliation_frozen",
        "snapshot_id": snapshot_id,
        "source": "NIST JARVIS-DFT dft_2d via jarvis-tools",
        "download_date": date.today().isoformat(),
        "jarvis_tools_version": importlib.metadata.version("jarvis-tools"),
        "target": "exfoliation_energy_mev_atom",
        "target_unit": "meV/atom",
        "file_format": {
            "train.json.gz": "gzip-compressed JSON Lines; one JSON object per line; pandas orient='records', lines=True",
            "validation.json.gz": "gzip-compressed JSON Lines; one JSON object per line; pandas orient='records', lines=True",
            "test_public.json.gz": "gzip-compressed JSON Lines; one JSON object per line; pandas orient='records', lines=True",
            "candidate_generation_constraints.json": "JSON object containing frozen candidate-generation and scoring thresholds",
        },
        "seed": args.seed,
        "counts": {
            "raw_records": len(raw),
            "valid_unique": len(df),
            "train": len(train),
            "validation": len(validation),
            "base_test": len(base_test),
            "test_public_total": len(test_public),
            "test_iid": int((base_test.split_type == "iid").sum()),
            "test_chemical_system_ood": int((base_test.split_type == "chemical_system_ood").sum()),
            "invariance_source_groups": len(inv_source_idx),
            "invariance_variant_rows": len(variant_rows),
        },
        "excluded": {
            **excluded,
            "duplicate_id_removed": duplicate_id_removed,
            "exact_structure_duplicate_removed": exact_structure_duplicate_removed,
        },
        "split_parameters": {
            "base_test_fraction": args.base_test_fraction,
            "validation_fraction": args.validation_fraction,
            "ood_share_of_base_test": args.ood_share_of_base_test,
            "invariance_source_fraction": args.invariance_source_fraction,
        },
        "permitted_input_columns": ["sample_id", "structure", "chemical_system"],
        "forbidden_auxiliary_dft_features": True,
        "benchmark_version": "1.0",
        "candidate_generation": {
            "required_count": 1000,
            "scoring_version": "candidate-library-score-v1",
            "property_used_for_candidate_scoring": False,
        },
        "public_file_sha256": public_hashes,
        "scoring_reference": scoring_reference,
    }
    (out / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (out / "checksums.sha256").write_text(
        "\n".join(f"{digest}  {name}" for name, digest in public_hashes.items()) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**metadata, "private_label_path": str(label_path)}, indent=2))


if __name__ == "__main__":
    main()
