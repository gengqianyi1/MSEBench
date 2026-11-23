#!/usr/bin/env python
"""Deterministic 2D-monolayer candidate generator for the JARVIS-2D exfoliation benchmark.

Reads data/train.json.gz (novelty reference) and the public candidate constraints,
constructs periodic monolayer structures on fixed templates, sizes the in-plane
lattice from covalent contacts, adds a 2D vacuum, validates every candidate against
the public pathology checks, de-duplicates with the frozen StructureMatcher policy
and writes exactly `required_candidate_count` gzip JSONL records together with
model-predicted exfoliation energies (embedded forest predictor).
"""
from __future__ import annotations
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- features ----
FEATURE_IMPL = r"""
import numpy as np
from pymatgen.core import Structure, Element

ELEM_ATTRS = ["Z","atomic_mass","X","atomic_radius","group","row"]

def _elem_attr(sym, attr):
    try:
        return float(getattr(Element(sym), attr))
    except Exception:
        return None

def comp_stats(comp):
    d = comp.get_el_amt_dict()
    elems = list(d.keys())
    fracs = np.array([d[e] for e in elems], dtype=float)
    fracs = fracs / fracs.sum()
    out = []
    for attr in ELEM_ATTRS:
        vals = np.array([_elem_attr(e, attr) if _elem_attr(e, attr) is not None else np.nan for e in elems], dtype=float)
        m = np.isfinite(vals)
        if m.sum() == 0:
            out += [np.nan] * 5
            continue
        w = fracs[m]; w = w / w.sum(); v = vals[m]
        mu = float((v * w).sum())
        out += [mu, float(v.min()), float(v.max()), float(np.sqrt((w * ((v - mu) ** 2)).sum())), float(v.max() - v.min())]
    return out

def _nearest_stats(st):
    n = len(st)
    try:
        ii, jj, offs, dd = st.get_neighbor_list(6.5)
    except Exception:
        return [5.0, 5.0, 0.0]
    best = np.full(n, np.inf); has = np.zeros(n, bool)
    for i0, j0, d in zip(ii, jj, dd):
        i, j = int(i0), int(j0)
        if i == j:
            continue
        if d < best[i]:
            best[i] = float(d); has[i] = True
    for i0, j0, d in zip(ii, jj, dd):
        i, j = int(i0), int(j0)
        if i == j and (not has[i]) and d < best[i]:
            best[i] = float(d); has[i] = True
    best = np.where(np.isfinite(best), best, 5.0)
    return [float(best.min()), float(best.mean()), float(best.std())]

def structure_feats(st):
    n = len(st)
    lat = st.lattice.matrix
    norms = np.linalg.norm(lat, axis=1)
    order = np.argsort(norms)
    area_pa = float(np.linalg.norm(np.cross(lat[order[0]], lat[order[1]]))) / max(n, 1)
    red_n = int(st.composition.reduced_composition.num_atoms)
    nn = _nearest_stats(st)
    return [red_n, area_pa] + nn

def make_features(sd):
    st = Structure.from_dict(sd)
    return comp_stats(st.composition.reduced_composition) + structure_feats(st)

FEATURE_NAMES = ([f"{a}_{s}" for a in ELEM_ATTRS for s in ["wmean","min","max","wstd","range"]]
                 + ["n_sites_reduced","area_per_atom","dmin","mean_nn","std_nn"])

def robust_matrix(structs):
    X = []
    for sd in structs:
        f = make_features(sd)
        X.append([x if np.isfinite(x) else 0.0 for x in f])
    return np.array(X, dtype=float)
"""
feature_ns = {}
exec(compile(FEATURE_IMPL, "features_impl", "exec"), feature_ns)
make_features = feature_ns["make_features"]
FEATURE_NAMES = feature_ns["FEATURE_NAMES"]

# --------------------------------------------------------- embedded predictor --
def predict_forest_matrix(X, trees):
    """ExtraTrees regression replicating sklearn traversal semantics (per tree)."""
    n = len(X)
    nt = len(trees)
    leaf_val = np.zeros((nt, n), dtype=np.float64)
    for ti, t in enumerate(trees):
        cl = np.asarray(t["children_left"], dtype=np.int64)
        cr = np.asarray(t["children_right"], dtype=np.int64)
        feat = np.asarray(t["feature"], dtype=np.int64)
        thr = np.asarray(t["threshold"], dtype=np.float64)
        node = np.zeros(n, dtype=np.int64)
        mask = cl[node] != -1
        while mask.any():
            idx = np.nonzero(mask)[0]
            go_left = X[idx, feat[node[idx]]] <= thr[node[idx]]
            node[idx[go_left]] = cl[node[idx[go_left]]]
            node[idx[~go_left]] = cr[node[idx[~go_left]]]
            mask = cl[node] != -1
        vals = np.asarray(t["value"], dtype=np.float64)
        leaf_val[ti] = vals[node]
    return leaf_val.mean(axis=0), leaf_val.std(axis=0)

# ------------------------------------------------------------- structure tools --
def _element_radius(symbol):
    try:
        from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
        value = CovalentRadius.radius.get(symbol)
        if value is not None and np.isfinite(float(value)):
            return float(value)
    except Exception:
        pass
    from pymatgen.core import Element
    el = Element(symbol)
    for attr in ("atomic_radius", "atomic_radius_calculated", "van_der_waals_radius"):
        value = getattr(el, attr, None)
        if value is not None:
            try:
                number = float(value)
            except Exception:
                continue
            if np.isfinite(number) and number > 0:
                return number
    return 1.0


def _largest_gap(frac_col):
    values = np.mod(np.asarray(frac_col, dtype=float), 1.0)
    ordered = np.sort(values)
    nxt = np.r_[ordered[1:], ordered[0] + 1.0]
    gaps = nxt - ordered
    index = int(np.argmax(gaps))
    return float(gaps[index]), float(nxt[index] % 1.0)


def _vacuum_axis(structure):
    lengths = np.asarray(structure.lattice.abc, dtype=float)
    best = None
    for axis in range(3):
        gapf, _ = _largest_gap(structure.frac_coords[:, axis])
        phys = gapf * lengths[axis]
        if best is None or phys > best[1]:
            best = (axis, phys)
    return best[0], best[1]


def pathology_reason(structure, constraints):
    cfg = constraints["pathology"]
    matrix = np.asarray(structure.lattice.matrix, dtype=float)
    if float(np.linalg.cond(matrix)) > float(cfg["maximum_lattice_condition_number"]):
        return "degenerate_lattice"
    close_i, close_j, _, close_d = structure.get_neighbor_list(float(cfg["minimum_absolute_distance_angstrom"]))
    if any(int(i) != int(j) or float(d) > 1e-8 for i, j, d in zip(close_i, close_j, close_d)):
        return "absolute_atomic_overlap"
    if len(structure) >= 2:
        dm = np.asarray(structure.distance_matrix, dtype=float)
        for i in range(len(structure)):
            for j in range(i + 1, len(structure)):
                rs = _element_radius(structure[i].specie.symbol) + _element_radius(structure[j].specie.symbol)
                if rs > 0 and float(dm[i, j] / rs) < float(cfg["minimum_covalent_radius_ratio"]):
                    return "covalent_radius_overlap"
    ni = structure.get_neighbor_list(float(cfg["coordination_cutoff_angstrom"]))[0]
    if len(ni):
        counts = np.bincount(np.asarray(ni, dtype=int), minlength=len(structure))
        if int(np.max(counts)) > int(cfg["maximum_neighbors_per_site"]):
            return "implausible_periodic_coordination"
    ax, vgap = _vacuum_axis(structure)
    if float(vgap) < float(cfg["minimum_vacuum_gap_angstrom"]):
        return "insufficient_2d_vacuum"
    bond_scale = float(constraints["geometry"]["bond_radius_scale"])
    max_cut = 2.0 * max(_element_radius(s.specie.symbol) for s in structure) * bond_scale
    ni, nj, offsets, nd = structure.get_neighbor_list(max_cut)
    for i, j, off, d in zip(ni, nj, offsets, nd):
        rs = _element_radius(structure[int(i)].specie.symbol) + _element_radius(structure[int(j)].specie.symbol)
        if abs(int(off[ax])) > 0 and float(d) <= bond_scale * rs:
            return "bonded_across_vacuum"
    return None


def normalize_2d_vacuum(structure):
    """Translate layer so the largest empty gap starts at fraction 0 (canonical origin)."""
    from pymatgen.core import Structure
    ax, _ = _vacuum_axis(structure)
    _, occ_start = _largest_gap(structure.frac_coords[:, ax])
    frac = np.mod(np.asarray(structure.frac_coords, dtype=float), 1.0)
    f = frac.copy()
    f[:, ax] = np.mod(f[:, ax] - occ_start, 1.0)
    return Structure(structure.lattice, structure.species, f, coords_are_cartesian=False,
                     site_properties=dict(structure.site_properties))


# ------------------------------------------------------------- motif library ----
def hex_lattice(a):
    return [[a, 0.0, 0.0], [a / 2.0, a * np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, 0.0]]

def sq_lattice(a):
    return [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, 0.0]]

def build_motif(motif, metal, nonmetal, config):
    from pymatgen.core import Lattice, Structure
    ra = _element_radius(metal)
    rb = _element_radius(nonmetal)
    if ra <= 0 or rb <= 0:
        return None
    d = float(config["bond_radius_scale"]) * (ra + rb)
    vac = float(config["vacuum_gap_angstrom"])
    frac_b = float(config["in_plane_bond_fraction"].get(motif, 1.0))
    vert = float(config["sandwich_vertical_fraction"].get(motif, 0.0))
    species = []
    frac_coords = []
    lat = None
    if motif == "hex_AB":
        a = np.sqrt(3.0) * d
        lat = hex_lattice(a)
        species = [metal, nonmetal]
        frac_coords = [[0.0, 0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0, 0.0]]
    elif motif == "sq_AB":
        a = np.sqrt(2.0) * d
        lat = sq_lattice(a)
        species = [metal, nonmetal]
        frac_coords = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]
    elif motif == "sq_AB2":
        a = 2.0 * d
        lat = sq_lattice(a)
        species = [metal, nonmetal, nonmetal]
        frac_coords = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]
    elif motif == "sq_AB3":
        a = 2.0 * d
        lat = sq_lattice(a)
        species = [metal, nonmetal, nonmetal, nonmetal]
        frac_coords = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.5, 0.0]]
    elif motif == "hex_AB_b":
        a = np.sqrt(3.0) * d
        lat = hex_lattice(a)
        w = float(config.get("buckle_angstrom", 0.5))
        species = [metal, nonmetal]
        frac_coords = [[0.0, 0.0, w], [1.0 / 3.0, 1.0 / 3.0, -w]]
    elif motif == "sq_AB2_b":
        a = 2.0 * d
        lat = sq_lattice(a)
        w = float(config.get("buckle_angstrom", 0.5))
        species = [metal, nonmetal, nonmetal]
        frac_coords = [[0.0, 0.0, 0.0], [0.5, 0.0, w], [0.0, 0.5, -w]]
    elif motif == "hex_AB2":
        d0 = frac_b * d
        if d0 >= d:
            return None
        u = float(np.sqrt(max(d * d - d0 * d0, 1e-6)))
        a = np.sqrt(3.0) * d0
        lat = hex_lattice(a)
        species = [metal, nonmetal, nonmetal]
        frac_coords = [[0.0, 0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0, u], [1.0 / 3.0, 1.0 / 3.0, -u]]
    else:
        return None
    zs = np.array([fc[2] for fc in frac_coords], dtype=float)
    h = float(max(zs.max() - zs.min(), 1e-6)) + 1.0
    c = h + vac
    mat = np.asarray(lat, dtype=float)
    mat[2][2] = c
    zmin = float(zs.min())
    final_frac = []
    for fc in frac_coords:
        t = (fc[2] - zmin) / max(h - 1.0, 1e-12)
        z = (t * h) / c
        final_frac.append([fc[0], fc[1], z + (1.0 - h / c) / 2.0])
    try:
        st = Structure(Lattice(mat), species, final_frac, coords_are_cartesian=False)
    except Exception:
        return None
    return st


# ---------------------------------------------------------------- main flow ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--constraints", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    constraints = json.loads(Path(args.constraints).read_text(encoding="utf-8"))
    required = int(constraints["required_candidate_count"])
    allowed = set(constraints["allowed_elements"])

    import pandas as pd
    from pymatgen.core import Structure, Element
    from pymatgen.core.structure_matcher import ElementComparator, StructureMatcher

    seed = int(config["seed"])

    train_formulas = set()
    train_elements = set()
    frame = pd.read_json(args.input, orient="records", lines=True, compression="infer")
    for raw, cs in zip(frame["structure"], frame["chemical_system"].fillna("").astype(str)):
        try:
            st = Structure.from_dict(raw)
            train_formulas.add(str(st.composition.reduced_formula))
        except Exception:
            pass
        for part in str(cs).split("-"):
            if part:
                train_elements.add(part)

    metals, nonmetals = [], []
    for sym in sorted(allowed):
        try:
            if Element(sym).is_metal:
                metals.append(sym)
            else:
                nonmetals.append(sym)
        except Exception:
            nonmetals.append(sym)

    matcher = StructureMatcher(
        ltol=float(constraints["structure_matcher"]["ltol"]),
        stol=float(constraints["structure_matcher"]["stol"]),
        angle_tol=float(constraints["structure_matcher"]["angle_tol"]),
        primitive_cell=bool(constraints["structure_matcher"]["primitive_cell"]),
        scale=bool(constraints["structure_matcher"]["scale"]),
        attempt_supercell=bool(constraints["structure_matcher"]["attempt_supercell"]),
        allow_subset=False,
        comparator=ElementComparator(),
    )

    trees = config["predictor"]["trees"]
    model_sha = config["prediction_model_sha256"]
    pred_round = int(config["prediction_round_decimals"])
    struct_round = int(config["structure_round_decimals"])

    pending = []            # (quantized structure dict, motif, metal, nonmetal)
    kept_by_formula = {}
    motif_order = config["motif_order"]
    seen = set()

    def quant(v, dec):
        if isinstance(v, float):
            return round(v, dec)
        if isinstance(v, list):
            return [quant(x, dec) for x in v]
        if isinstance(v, dict):
            return {k: quant(x, dec) for k, x in v.items()}
        return v

    import random as _random
    _rng = _random.Random(seed)
    motif_caps = config.get("motif_caps") or {}
    motif_order = list(config["motif_order"])
    # per-motif shuffled pair lists
    motif_pairs = {}
    for motif in motif_order:
        pairs = [(m, n) for m in metals for n in nonmetals]
        _rng.shuffle(pairs)
        motif_pairs[motif] = pairs
    pending = []
    kept_by_formula = {}
    seen = set()

    def quant(v, dec):
        if isinstance(v, float):
            return round(v, dec)
        if isinstance(v, list):
            return [quant(x, dec) for x in v]
        if isinstance(v, dict):
            return {k: quant(x, dec) for k, x in v.items()}
        return v

    got = {m: 0 for m in motif_order}
    for motif in motif_order:
        cap = min(int(motif_caps.get(motif, 10 ** 9)), required - len(pending))
        if cap <= 0:
            continue
        for metal, nonmetal in motif_pairs[motif]:
            if len(pending) >= required or got[motif] >= cap:
                break
            key = (motif, metal, nonmetal)
            if key in seen:
                continue
            seen.add(key)
            st = build_motif(motif, metal, nonmetal, config)
            if st is None:
                continue
            n = len(st)
            if n < int(constraints["min_sites"]) or n > int(constraints["max_sites"]):
                continue
            if len({s.specie.symbol for s in st}) > int(constraints["max_distinct_elements"]):
                continue
            if not {s.specie.symbol for s in st}.issubset(allowed):
                continue
            if not st.is_ordered:
                continue
            if float(st.volume) <= 1e-8:
                continue
            if pathology_reason(st, constraints) is not None:
                continue
            formula = str(st.composition.reduced_formula)
            norm = normalize_2d_vacuum(st)
            dup = False
            for kept in kept_by_formula.get(formula, []):
                try:
                    if matcher.fit(norm, kept):
                        dup = True
                        break
                except Exception:
                    dup = True
                    break
            if dup:
                continue
            kept_by_formula.setdefault(formula, []).append(norm)
            sd = quant(st.as_dict(), struct_round)
            try:
                qs = Structure.from_dict(sd)
            except Exception:
                continue
            pending.append((sd, qs, motif, metal, nonmetal))
            got[motif] += 1

    if len(pending) != required:
        sys.stderr.write("generated %d/%d records\n" % (len(pending), required))
        sys.exit(3)

    records = []
    for start in range(0, len(pending), 250):
        chunk = pending[start:start + 250]
        X = np.asarray([make_features(sd) for sd, _, _, _, _ in chunk], dtype=float)
        X = np.where(np.isfinite(X), X, 0.0)
        mean, std = predict_forest_matrix(X, trees)
        for k, (sd, qs, motif, metal, nonmetal) in enumerate(chunk):
            n = len(qs)
            pred = float(mean[k])
            unc = float(std[k])
            if not (np.isfinite(pred) and np.isfinite(unc) and unc > 0):
                unc = 1.0
            pred = max(0.0, pred)
            els = {s.specie.symbol for s in qs}
            el_known = float(np.mean([1.0 if e in train_elements else 0.0 for e in els]))
            dm = np.asarray(qs.distance_matrix, dtype=float)
            np.fill_diagonal(dm, np.inf)
            nn = dm.min(axis=1)
            ratios = [float(nn[i]) / max(_element_radius(qs[i].specie.symbol), 1e-9) for i in range(n)]
            ratio_q = float(np.mean(np.clip(1.0 - np.abs(np.asarray(ratios) - 1.0) / 0.45, 0.0, 1.0)))
            ad = float(np.clip(0.5 * el_known + 0.5 * ratio_q, 0.0, 1.0))
            d_bond = float(config["bond_radius_scale"]) * (_element_radius(metal) + _element_radius(nonmetal))
            ops = [
                {"name": "choose_template",
                 "parameters": {"motif": motif, "n_sites": int(n),
                                "stoichiometry": "AB" if motif.split("_")[1] == "AB" else motif.split("_")[1]}},
                {"name": "assign_species",
                 "parameters": {"metal": metal, "nonmetal": nonmetal,
                                "target_bond_angstrom": round(d_bond, 4)}},
                {"name": "size_lattice_to_contacts",
                 "parameters": {"bravais": motif.split("_")[0],
                                "vacuum_gap_angstrom": float(config["vacuum_gap_angstrom"])}},
            ]
            index = start + k
            record = {
                "candidate_id": "CAND-%06d" % index,
                "generation_index": index,
                "random_seed": seed,
                "predicted_exfoliation_energy_mev_atom": round(pred, pred_round),
                "prediction_uncertainty_mev_atom": round(float(max(unc, 1e-3)), pred_round),
                "applicability_domain_score": round(ad, 4),
                "prediction_model_sha256": model_sha,
                "structure": sd,
                "provenance": {
                    "strategy": "deterministic template construction with covalent-contact lattice sizing and fixed 2D vacuum",
                    "parent_ids": [],
                    "operations": ops,
                },
                "notes": "de novo monolayer template candidate; no public structure used as parent",
            }
            records.append(record)

    out = Path(args.output)
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.write(json.dumps({"generated": len(records), "output": str(out)}))


if __name__ == "__main__":
    main()
