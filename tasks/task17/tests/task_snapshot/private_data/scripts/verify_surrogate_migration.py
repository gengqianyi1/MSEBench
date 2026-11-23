#!/usr/bin/env python
"""End-to-end proof that replacing the pickled surrogate with the .npz changed no graded number.

scripts/convert_surrogate_to_npz.py verifies the NETWORK output. That is necessary but not
sufficient: the network feeds two of the four responses directly (`strength`, `porosity`) and one
indirectly (`porosity` enters the attrition loss term of `yield_pct`), while `d50` comes from the
population balance and should be untouched. A reviewer needs the difference on each graded quantity
separately, not one aggregate number, so this script reports:

    max |old - new|   for d50, strength, porosity, yield_pct   and for the desirability D

measured over the whole 1-slot and 2-slot design families and a deterministic random sample of the
change-budget neighbourhood. "old" is the legacy path -- the pickled scikit-learn Pipeline, called
exactly as the pre-migration `responses()` called it. "new" is the shipped path.

The legacy pickle is not part of the shipped code path. It is kept only so this comparison can be
re-run by a third party; nothing imports it at evaluation time. If it is absent the script says so and
prints the recorded measurement from data/agglo_strength_porosity.schema.json instead.

Run:
    python scripts/verify_surrogate_migration.py [--sample 5000]

Writes data/SURROGATE_MIGRATION_EVIDENCE.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Any, Dict, List

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
for _p in (SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import agglomeration as A                                        # noqa: E402
from design_space import (default_desirability, iter_budget_designs,  # noqa: E402
                          iter_k_slot_designs)

LEGACY_CANDIDATES = [
    os.path.join(ROOT, "data", "legacy", "agglo_strength_porosity.joblib"),
    os.path.join(ROOT, "data", "agglo_strength_porosity.joblib"),
]
EVIDENCE = os.path.join(ROOT, "data", "SURROGATE_MIGRATION_EVIDENCE.json")
SCHEMA = os.path.join(ROOT, "data", "agglo_strength_porosity.schema.json")
TOL = 1e-9          # anything above this in a graded response invalidates the migration


def find_legacy() -> str:
    for p in LEGACY_CANDIDATES:
        if os.path.exists(p):
            return p
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=5000,
                    help="deterministic random sample size from the change-budget neighbourhood")
    args = ap.parse_args()

    legacy_path = find_legacy()
    if not legacy_path:
        print("The legacy pickle is not present in this tree, so the comparison cannot be re-run "
              "here. The measurement recorded at migration time was:")
        if os.path.exists(EVIDENCE):
            with open(EVIDENCE) as fh:
                print(json.dumps(json.load(fh).get("per_response_max_abs_diff", {}), indent=2))
            return 0
        if os.path.exists(SCHEMA):
            with open(SCHEMA) as fh:
                print(json.dumps(json.load(fh).get("conversion", {}), indent=2))
            return 0
        print("  (no recorded measurement found)")
        return 1

    import joblib                       # ONLY here, and only for the audit
    t0 = time.time()
    legacy = joblib.load(legacy_path)
    print(f"legacy pickle : {os.path.relpath(legacy_path, ROOT)}")
    print(f"shipped bundle: {os.path.relpath(A.DEFAULT_SURROGATE, ROOT)}")

    def legacy_predict(_bundle: Any, X: np.ndarray) -> np.ndarray:
        """Byte-for-byte the expression the pre-migration responses() used."""
        return legacy["pipe"].predict(X) * legacy["sc_y"] + legacy["mu_y"]

    # ------------------------------------------------------------------ the design sets
    designs: List[Dict[str, Any]] = []
    labels: List[str] = []
    for k in (1, 2):
        n0 = len(designs)
        designs += [x for x, _ in iter_k_slot_designs(k)]
        labels.append(f"{k}-slot family ({len(designs) - n0} designs)")
    n_fam = len(designs)

    region = [x for x, _ in iter_budget_designs(A.BUDGET, feasible_only=True)]
    rng = np.random.default_rng(20260729)
    take = min(args.sample, len(region))
    pick = rng.choice(len(region), size=take, replace=False)
    designs += [region[int(i)] for i in sorted(pick)]
    labels.append(f"random sample of the feasible change-budget neighbourhood ({take} of "
                  f"{len(region)}, seed 20260729)")
    designs.append(dict(A.START))
    labels.append("the registered (starting) process")

    print(f"designs compared: {len(designs)}")
    for lab in labels:
        print(f"  - {lab}")

    # ------------------------------------------------------------------ measure both paths
    keys = list(A.RESPONSE_KEYS)
    des = default_desirability()

    def measure(all_designs, predictor):
        saved = A.surrogate_predict
        A.surrogate_predict = predictor
        try:
            out = np.zeros((len(all_designs), len(keys) + 1))
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.filterwarnings("ignore", message=".*encountered in matmul.*",
                                        category=RuntimeWarning)
                for i, x in enumerate(all_designs):
                    r = A.responses(x)
                    for j, k in enumerate(keys):
                        out[i, j] = float(r[k])
                    out[i, len(keys)] = des.D_of(r)
            return out
        finally:
            A.surrogate_predict = saved

    print("\nmeasuring the legacy path ...")
    old = measure(designs, legacy_predict)
    print("measuring the shipped path ...")
    new = measure(designs, A.__dict__["surrogate_predict"])

    diff = np.abs(old - new)
    per: Dict[str, Dict[str, float]] = {}
    print("\n" + "=" * 84)
    print("max |old (pickled sklearn Pipeline)  -  new (.npz + numpy forward pass)|")
    print("=" * 84)
    for j, k in enumerate(keys + ["desirability_D"]):
        m = float(diff[:, j].max())
        rel = float((diff[:, j] / np.maximum(np.abs(old[:, j]), 1e-300)).max())
        per[k] = {"max_abs_diff": m, "max_rel_diff": rel,
                  "n_designs_differing": int((diff[:, j] > 0).sum())}
        unit = {"d50": "um", "strength": "MPa", "porosity": "-", "yield_pct": "%",
                "desirability_D": "-"}[k]
        print(f"  {k:<16s} [{unit:>3s}]   max|delta| = {m:.3e}   max rel = {rel:.3e}   "
              f"designs differing: {int((diff[:, j] > 0).sum())} / {len(designs)}")
    print("=" * 84)

    worst = max(v["max_abs_diff"] for v in per.values())
    verdict = "BIT-IDENTICAL" if worst == 0.0 else (
        "identical to machine precision" if worst <= TOL else "DIFFERENT -- MIGRATION INVALID")
    print(f"verdict: {verdict}   (worst over all four responses and D: {worst:.3e}, "
          f"tolerance {TOL:.0e})")

    evidence = {
        "what": "End-to-end comparison of the four simulated responses and the desirability D "
                "between the legacy pickled scikit-learn surrogate and the shipped non-executable "
                ".npz surrogate.",
        "generated_by": "scripts/verify_surrogate_migration.py",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "legacy_path": os.path.relpath(legacy_path, ROOT),
        "shipped_path": os.path.relpath(A.DEFAULT_SURROGATE, ROOT),
        "n_designs_compared": len(designs),
        "design_sets": labels,
        "n_in_1_and_2_slot_families": n_fam,
        "per_response_max_abs_diff": per,
        "worst_over_all_responses": worst,
        "tolerance": TOL,
        "verdict": verdict,
        "note": "d50 does not read the surrogate at all (population balance only); porosity enters "
                "the attrition term of yield_pct, so a surrogate drift would show up in yield_pct "
                "as well. All five quantities are reported separately for that reason.",
        "seconds": round(time.time() - t0, 1),
    }
    with open(EVIDENCE, "w", encoding="utf-8") as fh:
        json.dump(evidence, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {os.path.relpath(EVIDENCE, ROOT)}  ({time.time() - t0:.1f} s)")
    return 0 if worst <= TOL else 1


if __name__ == "__main__":
    sys.exit(main())
