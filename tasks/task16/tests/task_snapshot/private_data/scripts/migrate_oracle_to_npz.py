"""One-shot migration: sklearn pickle -> non-executable .npz + JSON schema, with a numerical proof.

Audit finding #2 (2026-07-29): the scoring oracle shipped as `data/oracle.pkl`, an executable
object deserialisation. `pickle.load` can run arbitrary code, so a third party could not safely
load the ruler, and the package could not be reproduced inside a normal review sandbox.

This script converts the fitted `StandardScaler x2 + MLPRegressor(128, 64)` into

  data/oracle.npz          plain float64 arrays: W0/b0/W1/b1/W2/b2, x_mean/x_scale, y_mean/y_scale
  data/oracle_schema.json  input/output contract, layer sizes, activation names, sha256 of the npz

and then proves the migration is exact: it evaluates BOTH implementations over the whole
change-budget neighbourhood and prints max |old - new| per CQA. The migration is only accepted if
that maximum is 0.0 (bit-identical), not merely small.

`.npz` is a zip of raw arrays. `numpy.load(..., allow_pickle=False)` cannot execute anything, and
src/formulation.py passes `allow_pickle=False` explicitly.

Run (needs scikit-learn, i.e. the build interpreter, and a data/oracle.pkl to read):
    $PY scripts/migrate_oracle_to_npz.py
    $PY scripts/migrate_oracle_to_npz.py --verify-only     # no write, just re-prove equality
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PKL = ROOT / "data" / "oracle.pkl"
NPZ = ROOT / "data" / "oracle.npz"
SCHEMA = ROOT / "data" / "oracle_schema.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    if not PKL.exists():
        raise SystemExit(
            "data/oracle.pkl is absent. That is the expected state of a SHIPPED package -- the "
            "pickle was removed after this migration ran. Nothing to migrate; run "
            "scripts/verify_oracle_npz.py instead.")

    with open(PKL, "rb") as fh:                       # nosec - designer-side, one-shot, own file
        blob = pickle.load(fh)
    xs, ys, net = blob["xs"], blob["ys"], blob["net"]

    if net.activation != "relu" or net.out_activation_ != "identity":
        raise SystemExit("unexpected activations: %s / %s" % (net.activation, net.out_activation_))
    if len(net.coefs_) != 3:
        raise SystemExit("unexpected depth: %d weight matrices" % len(net.coefs_))

    arrays = {
        "W0": np.asarray(net.coefs_[0], dtype=np.float64),
        "b0": np.asarray(net.intercepts_[0], dtype=np.float64),
        "W1": np.asarray(net.coefs_[1], dtype=np.float64),
        "b1": np.asarray(net.intercepts_[1], dtype=np.float64),
        "W2": np.asarray(net.coefs_[2], dtype=np.float64),
        "b2": np.asarray(net.intercepts_[2], dtype=np.float64),
        "x_mean": np.asarray(xs.mean_, dtype=np.float64),
        "x_scale": np.asarray(xs.scale_, dtype=np.float64),
        "y_mean": np.asarray(ys.mean_, dtype=np.float64),
        "y_scale": np.asarray(ys.scale_, dtype=np.float64),
    }

    if not args.verify_only:
        np.savez(NPZ, **arrays)
        print("wrote %s (%d bytes)" % (NPZ, NPZ.stat().st_size))

    # ---------------------------------------------------------------- numerical equivalence
    # Import AFTER the npz exists so formulation.Oracle loads the new format.
    import response_model as formulation                            # noqa: E402
    import design_space                                             # noqa: E402

    new = formulation.Oracle(str(NPZ))
    R = design_space.enumerate_rows(4)
    X = design_space.feat_matrix(R)
    print("evaluating both implementations over %d designs x 22 features ..." % len(R))

    Y_new = formulation._quiet_matmul(lambda: new.predict_matrix(X))
    Y_old = formulation._quiet_matmul(
        lambda: ys.inverse_transform(net.predict(xs.transform(X))))

    diff = np.abs(Y_old - Y_new)
    print("max |pickle - npz| per CQA: %s" % np.array2string(diff.max(0), precision=17))
    print("max |pickle - npz| overall: %.17g" % diff.max())
    print("machine epsilon (float64):  %.17g" % np.finfo(np.float64).eps)
    if diff.max() != 0.0:
        raise SystemExit("MIGRATION REJECTED: the two implementations are not bit-identical")
    print("MIGRATION EXACT: max|old-new| == 0.0 over the whole change-budget neighbourhood")

    if args.verify_only:
        return

    schema = {
        "format": "npz-float64-mlp",
        "format_note": (
            "Non-executable. numpy.load(path, allow_pickle=False) reads plain float64 arrays; "
            "no object deserialisation is involved anywhere in the load path."),
        "produced_by": "scripts/migrate_oracle_to_npz.py from the sklearn model fitted with "
                       "seed 20260727 (src/response_model.train_oracle)",
        "data_nature": "SYNTHETIC. Fitted to fixed-seed samples of the closed-form simulator "
                       "src/response_model.cqa_true plus fixed-seed Gaussian noise. Not real product, "
                       "batch or registration data. See data/DATA_CARD.md.",
        "input": {
            "n_features": 22,
            "dtype": "float64",
            "order": "exactly as produced by src/response_model.feature_vector",
            "features": [
                "onehot_T1_mcc_grade_PH101", "onehot_T1_mcc_grade_PH102",
                "onehot_T1_mcc_grade_PH200", "onehot_T1_mcc_grade_PH301",
                "onehot_T3_acid_id_none", "onehot_T3_acid_id_FUM",
                "onehot_T3_acid_id_TAR", "onehot_T3_acid_id_CIT",
                "onehot_T6_dis_id_XPVP", "onehot_T6_dis_id_CCS",
                "onehot_T6_dis_id_LHPC", "onehot_T6_dis_id_SSG",
                "T2_mcc_pct", "T4_acid_pct", "T5_meg_pct", "T7_dis_pct", "T8_mgst_pct",
                "log_T9_blend_min", "T10_sil_pct", "T11_press_MPa_over_100", "T12_man_pct",
                "qs_filler_DCPA_pct",
            ],
        },
        "output": {
            "n_outputs": 5,
            "dtype": "float64",
            "keys": ["microenvironment_pH", "emc_60rh_pct", "tensile_strength_MPa",
                     "ffc", "q30_pct"],
            "units": ["pH of a 20 % w/v slurry", "% w/w at 25 C / 60 % RH", "MPa",
                      "dimensionless", "% dissolved at 30 min"],
        },
        "architecture": {
            "kind": "multilayer perceptron, dense",
            "layer_sizes": [22, 128, 64, 5],
            "hidden_activation": "relu",
            "output_activation": "identity",
            "input_standardisation": "z = (x - x_mean) / x_scale",
            "output_destandardisation": "y = z_out * y_scale + y_mean",
            "forward_pass": "h1 = relu(z @ W0 + b0); h2 = relu(h1 @ W1 + b1); "
                            "z_out = h2 @ W2 + b2",
        },
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
        "migration_check": {
            "reference_implementation": "sklearn 1.7.2 StandardScaler x2 + MLPRegressor(128, 64), "
                                        "loaded from the superseded data/oracle.pkl",
            "n_designs_compared": int(len(R)),
            "max_abs_difference": 0.0,
            "verdict": "bit-identical",
        },
        "sha256_npz": sha256(NPZ),
    }
    SCHEMA.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote %s  (sha256 %s)" % (SCHEMA, schema["sha256_npz"][:16] + "..."))

    # ---------------------------------------------------------------- durable evidence
    # Written so the numerical-equivalence claim survives the deletion of the pickle. The golden
    # vectors let scripts/verify_oracle_npz.py re-prove the ruler on any machine, offline, with no
    # scikit-learn and no pickle present.
    rng = np.random.default_rng(20260729)
    sel = np.sort(rng.choice(len(R), 64, replace=False))
    # Golden vectors are recorded through the SCALAR (one design at a time) path, because that is
    # the path a grader uses and the path verify_oracle_npz.py can re-run. Batched BLAS and
    # row-at-a-time BLAS agree only to ~1e-14, so recording batched values here and re-checking them
    # row-at-a-time would build a 1e-14 tolerance into an otherwise exact claim.
    gold = []
    worst_row = 0.0
    for j in sel:
        d = design_space.to_design(R[j])
        y_old = formulation._quiet_matmul(
            lambda: ys.inverse_transform(net.predict(xs.transform(
                formulation.feature_vector(d)[None, :]))))[0]
        y_new = new.predict_one(d)
        worst_row = max(worst_row, float(np.abs(y_old - y_new).max()))
        gold.append({"row_index": int(j), "design": d, "cqa": y_new.tolist()})
    print("max |pickle - npz| row-at-a-time over %d golden designs: %.17g" % (len(gold), worst_row))
    if worst_row != 0.0:
        raise SystemExit("MIGRATION REJECTED: scalar path is not bit-identical")

    evidence = {
        "check": "pickle -> npz oracle migration, numerical equivalence",
        "date": "2026-07-29",
        "audit_finding": "#2 -- core model used unsafe pickle object deserialisation",
        "old_implementation": "sklearn 1.7.2: StandardScaler.transform -> "
                              "MLPRegressor(128, 64).predict -> StandardScaler.inverse_transform, "
                              "unpickled from data/oracle.pkl",
        "new_implementation": "src/response_model.Oracle.predict_matrix on data/oracle.npz, loaded "
                              "with numpy.load(allow_pickle=False); explicit float64 "
                              "standardise / relu / relu / identity / de-standardise",
        "n_designs_compared": int(len(R)),
        "max_abs_diff_per_cqa": diff.max(0).tolist(),
        "max_abs_diff_overall": float(diff.max()),
        "max_abs_diff_scalar_path": worst_row,
        "float64_eps": float(np.finfo(np.float64).eps),
        "verdict": "bit-identical (max|old - new| == 0.0, not merely within machine precision), "
                   "both batched over the whole neighbourhood and one design at a time",
        "sha256_npz": schema["sha256_npz"],
        "golden_vectors_note": "64 designs drawn with numpy default_rng(20260729) from the "
                               "enumerated <= 4-slot neighbourhood, in ascending row order, "
                               "evaluated ONE AT A TIME (the grading path). "
                               "scripts/verify_oracle_npz.py re-evaluates them and asserts exact "
                               "equality, which re-proves the shipped ruler without the pickle.",
        "golden_vectors": gold,
    }
    (ROOT / "data" / "oracle_migration_check.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote data/oracle_migration_check.json  (max|old-new| = %.17g)" % diff.max())


if __name__ == "__main__":
    main()
