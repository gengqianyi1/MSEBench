#!/usr/bin/env python
"""Migrate the frozen strength/porosity surrogate from a pickle to a NON-EXECUTABLE format.

Why
---
The surrogate used to ship as `data/agglo_strength_porosity.joblib`, a pickled scikit-learn
Pipeline. Loading it means unpickling, and unpickling is arbitrary code execution: a reviewer cannot
audit the file before running it, the loader depends on the exact scikit-learn minor version, and the
package could not be reproduced inside a locked-down environment. External review flagged this as a
blocking defect.

The replacement is `data/agglo_strength_porosity.npz` (plain float64 arrays, no pickled objects,
`allow_pickle=False`) plus `data/agglo_strength_porosity.schema.json`, which fixes the input/output
contract: feature order, shapes, activations, the output transform and the integrity hashes. The
forward pass is 30 lines of numpy in `src/agglomeration.surrogate_predict`; nothing is deserialised
into code.

What this script guarantees
---------------------------
Numerical identity, measured rather than asserted. The conversion is only accepted if the pure-numpy
forward pass reproduces the pickle's predictions to machine precision on:

  * the 60 000-row training matrix that ships in data/agglo_training_data.npz,
  * every design in the mechanically generated 1-slot and 2-slot families (1 386 designs) pushed
    through the real feature vector,
  * the registered process itself.

`max |old - new|` is printed for each set. Anything above 1e-12 in the network output aborts the
conversion.

Run:
    python scripts/convert_surrogate_to_npz.py            # convert + verify
    python scripts/convert_surrogate_to_npz.py --verify-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
for _p in (SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

DATA = os.path.join(ROOT, "data")
PICKLE = os.path.join(DATA, "agglo_strength_porosity.joblib")
NPZ = os.path.join(DATA, "agglo_strength_porosity.npz")
SCHEMA = os.path.join(DATA, "agglo_strength_porosity.schema.json")
TRAINING = os.path.join(DATA, "agglo_training_data.npz")

TOL = 1e-12


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def numpy_forward(bundle: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    """The reference implementation, written once here and once in src/agglomeration.py.

    Deliberately duplicated: this copy exists so the verification below is not comparing the shipped
    loader against itself.  scripts/check_consistency.py asserts the two agree.
    """
    Z = (np.asarray(X, dtype=np.float64) - bundle["scaler_mean"]) / bundle["scaler_scale"]
    n = int(bundle["n_layers"])
    for i in range(n):
        Z = Z @ bundle[f"W{i}"]
        Z = Z + bundle[f"b{i}"]
        if i != n - 1:
            Z = np.maximum(Z, 0.0)          # relu
    return Z * bundle["sc_y"] + bundle["mu_y"]


def extract(model: Any) -> Dict[str, np.ndarray]:
    pipe = model["pipe"]
    steps = dict(pipe.named_steps)
    sc = steps["sc"]
    mlp = steps["mlp"]
    if mlp.activation != "relu":
        raise SystemExit(f"unsupported hidden activation {mlp.activation!r}")
    if mlp.out_activation_ != "identity":
        raise SystemExit(f"unsupported output activation {mlp.out_activation_!r}")
    if not (sc.with_mean and sc.with_std):
        raise SystemExit("StandardScaler was fitted without mean/std centring; schema assumes both")

    out: Dict[str, np.ndarray] = {
        "scaler_mean": np.asarray(sc.mean_, dtype=np.float64),
        "scaler_scale": np.asarray(sc.scale_, dtype=np.float64),
        "mu_y": np.asarray(model["mu_y"], dtype=np.float64),
        "sc_y": np.asarray(model["sc_y"], dtype=np.float64),
        "n_layers": np.asarray(len(mlp.coefs_), dtype=np.int64),
    }
    for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        out[f"W{i}"] = np.asarray(W, dtype=np.float64)
        out[f"b{i}"] = np.asarray(b, dtype=np.float64)
    return out


def load_npz() -> Dict[str, np.ndarray]:
    with np.load(NPZ, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true",
                    help="do not rewrite the .npz; only re-measure pickle vs numpy agreement")
    args = ap.parse_args()

    if not os.path.exists(PICKLE):
        print(f"the legacy pickle {PICKLE} is absent.")
        if not os.path.exists(NPZ):
            print("and there is no .npz either -- nothing to do")
            return 1
        print("nothing to convert; the package already ships the non-executable format only.")
        print("(Numerical identity against the pickle was measured at conversion time; the record "
              "is in data/agglo_strength_porosity.schema.json under `conversion`.)")
        return 0

    import joblib                                   # only needed for the one-off migration
    t0 = time.time()
    model = joblib.load(PICKLE)
    feats: List[str] = list(model["feats"])
    seed = int(model["seed"])
    bundle = extract(model)

    print("=" * 96)
    print("1  extracted parameters")
    print("=" * 96)
    for k in sorted(bundle):
        print(f"  {k:<14s} {tuple(np.shape(bundle[k]))} {bundle[k].dtype}")
    print(f"  features       {len(feats)}: {feats}")
    print(f"  seed           {seed}")

    if not args.verify_only:
        np.savez(NPZ, **bundle)
        print(f"\n  wrote {NPZ}  ({os.path.getsize(NPZ) / 1024:.0f} kB)")

    bundle_disk = load_npz()

    print("\n" + "=" * 96)
    print("2  numerical identity, measured on three sets")
    print("=" * 96)
    worst = 0.0
    rows = []

    with np.load(TRAINING, allow_pickle=False) as z:
        Xtrain = np.asarray(z["X"], dtype=np.float64)
    sets = [("training matrix (60 000 x 16)", Xtrain)]

    # the real feature vectors of every 1-slot and 2-slot design
    import agglomeration as A                                   # noqa: E402
    from design_space import iter_k_slot_designs                # noqa: E402
    fam = [A.feature_vector(A.intermediates(x))[0]
           for k in (1, 2) for x, _ in iter_k_slot_designs(k)]
    sets.append((f"1/2-slot design families ({len(fam)} designs)",
                 np.asarray(fam, dtype=np.float64)))
    sets.append(("the registered process",
                 A.feature_vector(A.intermediates(A.START))))

    import warnings                                             # noqa: E402
    for label, X in sets:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.filterwarnings("ignore", message=".*encountered in matmul.*",
                                    category=RuntimeWarning)
            ref = model["pipe"].predict(X) * model["sc_y"] + model["mu_y"]
        got = numpy_forward(bundle_disk, X)
        d = float(np.max(np.abs(np.asarray(ref, dtype=np.float64).reshape(got.shape) - got)))
        worst = max(worst, d)
        rows.append((label, len(X), d))
        print(f"  {label:<42s} n={len(X):<6d} max|pickle - npz| = {d:.3e}")

    # and the difference that actually matters: the two responses in physical units
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.filterwarnings("ignore", message=".*encountered in matmul.*",
                                category=RuntimeWarning)
        ref_phys = np.exp(model["pipe"].predict(sets[1][1]) * model["sc_y"] + model["mu_y"])
    got_phys = np.exp(numpy_forward(bundle_disk, sets[1][1]))
    d_sigma = float(np.max(np.abs(ref_phys[:, 0] - got_phys[:, 0])))
    d_eps = float(np.max(np.abs(ref_phys[:, 1] - got_phys[:, 1])))
    print(f"  {'strength, MPa (1/2-slot families)':<42s}          "
          f"max|pickle - npz| = {d_sigma:.3e}")
    print(f"  {'porosity, - (1/2-slot families)':<42s}          "
          f"max|pickle - npz| = {d_eps:.3e}")

    if worst > TOL:
        print(f"\nABORTED: worst network-output disagreement {worst:.3e} exceeds {TOL:.0e}")
        return 1

    if not args.verify_only:
        schema = {
            "name": "agglo_strength_porosity",
            "format": "numpy .npz, float64, allow_pickle=False -- no executable object is "
                      "deserialised at load time",
            "purpose": "SYNTHETIC surrogate for two of the four simulated responses "
                       "(agglomerate crush strength and intra-agglomerate porosity). This is not a "
                       "model of any real product; see data/MODEL_CARD.md.",
            "arrays": {k: {"shape": list(np.shape(bundle_disk[k])),
                           "dtype": str(bundle_disk[k].dtype)} for k in sorted(bundle_disk)},
            "input": {
                "name": "feature_vector",
                "shape": ["n_samples", len(feats)],
                "dtype": "float64",
                "order": feats,
                "source": "src/agglomeration.feature_vector(intermediates(design))",
            },
            "forward_pass": [
                "Z = (X - scaler_mean) / scaler_scale",
                "for i in 0..n_layers-1:  Z = Z @ W{i} + b{i};  relu(Z) except on the last layer",
                "Y = Z * sc_y + mu_y",
                "strength_MPa = exp(Y[:, 0]);  porosity = exp(Y[:, 1])",
            ],
            "output": {
                "shape": ["n_samples", 2],
                "columns": ["log_strength", "log_porosity"],
                "physical": {"strength": "exp(col 0), MPa", "porosity": "exp(col 1), -"},
            },
            "activation": {"hidden": "relu", "output": "identity"},
            "training_seed": seed,
            "conversion": {
                "converted_from": "data/agglo_strength_porosity.joblib "
                                  "(scikit-learn Pipeline: StandardScaler + MLPRegressor(128,96,64))",
                "converted_by": "scripts/convert_surrogate_to_npz.py",
                "converted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "verification": [
                    {"set": label, "n": n, "max_abs_diff_network_output": diff}
                    for label, n, diff in rows
                ],
                "max_abs_diff_strength_MPa": d_sigma,
                "max_abs_diff_porosity": d_eps,
                "tolerance": TOL,
                "verdict": "identical to machine precision",
            },
            "integrity": {
                "npz_sha256": sha256(NPZ),
                "legacy_joblib_sha256": sha256(PICKLE),
            },
        }
        with open(SCHEMA, "w", encoding="utf-8") as fh:
            json.dump(schema, fh, indent=2, ensure_ascii=False)
        print(f"\n  wrote {SCHEMA}")

    print(f"\nOK: pure-numpy forward pass reproduces the pickle to {worst:.3e} "
          f"(tolerance {TOL:.0e}) in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
