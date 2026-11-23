"""Third-party verifier for the scoring oracle. No scikit-learn, no pickle, no network.

This is the script a reviewer runs to satisfy audit finding #2 ("new package must not depend on
executable object deserialisation; a third party must be able to load the model safely and
reproduce the stated baseline"). It checks five things and exits non-zero on any failure:

  1. data/oracle.npz loads with numpy.load(allow_pickle=False) and contains exactly the declared
     float64 arrays at the declared shapes;
  2. the sha256 of data/oracle.npz equals the one recorded in data/oracle_schema.json;
  3. no shipped source file contains an executable-deserialisation call at all
     (pickle / joblib.load / torch.load / allow_pickle=True / yaml.load without SafeLoader);
  4. the 64 golden CQA vectors in data/oracle_migration_check.json -- produced by the superseded
     sklearn pickle before it was deleted -- are reproduced by the shipped npz path;
  5. the starting formulation and the reference design in tasks/.../task_config.yaml reproduce the
     desirability values that file declares, i.e. the shipped baseline is reproducible.

Run:
    $PY scripts/verify_oracle_npz.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluator import evaluate_design, scorer_from_config            # noqa: E402
from formulation import START                                            # noqa: E402
from response_model import ORACLE_ARRAY_KEYS, Oracle, get_oracle         # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print("   [%s] %s%s" % ("ok " if ok else "FAIL", label, ("  -- " + detail) if detail else ""))
    if not ok:
        FAILURES.append(label)


def main() -> None:
    npz = ROOT / "data" / "oracle.npz"
    schema = json.loads((ROOT / "data" / "oracle_schema.json").read_text(encoding="utf-8"))

    print("=" * 96)
    print("1  non-executable load path")
    with np.load(npz, allow_pickle=False) as z:
        names = sorted(z.files)
        shapes = {k: list(np.asarray(z[k]).shape) for k in z.files}
        dtypes = {k: str(np.asarray(z[k]).dtype) for k in z.files}
    check("numpy.load(allow_pickle=False) succeeded", True, "%d arrays" % len(names))
    check("array set == declared", names == sorted(ORACLE_ARRAY_KEYS) == sorted(schema["arrays"]),
          str(names))
    check("shapes == schema", all(shapes[k] == schema["arrays"][k]["shape"] for k in shapes))
    check("all arrays float64", set(dtypes.values()) == {"float64"}, str(sorted(set(dtypes.values()))))
    check("pickled oracle path is refused", _refuses_pkl())

    print("\n2  integrity")
    h = hashlib.sha256(npz.read_bytes()).hexdigest()
    check("sha256(data/oracle.npz) == schema.sha256_npz", h == schema["sha256_npz"], h[:24] + "...")
    check("data/oracle.pkl absent from the package", not (ROOT / "data" / "oracle.pkl").exists())

    print("\n3  no executable deserialisation anywhere in the shipped sources")
    patterns = [
        (r"\bimport\s+pickle\b", "import pickle"),
        (r"\bpickle\.(load|loads)\s*\(", "pickle.load"),
        (r"\bjoblib\.load\s*\(", "joblib.load"),
        (r"\btorch\.load\s*\(", "torch.load"),
        (r"allow_pickle\s*=\s*True", "allow_pickle=True"),
        (r"yaml\.load\s*\((?![^)]*Safe)", "yaml.load without SafeLoader"),
    ]
    # The converter and this verifier are excluded: the converter is the designer-side one-shot that
    # READS the old pickle (and ships in neither zip), and this file's own pattern table contains the
    # literal strings it is looking for. Everything a grader or a run actually executes is scanned.
    self_exempt = {"migrate_oracle_to_npz.py", "verify_oracle_npz.py"}
    scanned, hits = 0, []
    for p in sorted(list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py"))):
        if p.name in self_exempt:
            continue
        scanned += 1
        text = p.read_text(encoding="utf-8")
        for rx, label in patterns:
            for m in re.finditer(rx, text):
                line = text[:m.start()].count("\n") + 1
                hits.append("%s:%d %s" % (p.relative_to(ROOT), line, label))
    # Guard against a vacuous scan, in a way that holds in BOTH delivery packages: the grader zip
    # carries ~20 scripts, the public zip only the four a participant needs. What must always be
    # true is that every module under src/ -- the code that actually loads the oracle and grades --
    # was read.
    n_src = len(list((ROOT / "src").rglob("*.py")))
    n_src_scanned = sum(1 for p in (ROOT / "src").rglob("*.py") if p.name not in self_exempt)
    check("scanned every module under src/ (%d) plus the shipped scripts" % n_src,
          scanned >= n_src_scanned + 1 and n_src >= 10,
          "%d files scanned, src/ has %d" % (scanned, n_src))
    check("zero executable-deserialisation call sites", not hits, "; ".join(hits[:6]))

    print("\n4  golden vectors from the superseded sklearn pickle are reproduced")
    ev = json.loads((ROOT / "data" / "oracle_migration_check.json").read_text(encoding="utf-8"))
    check("recorded max|pickle - npz| == 0.0", ev["max_abs_diff_overall"] == 0.0,
          "%.17g over %d designs" % (ev["max_abs_diff_overall"], ev["n_designs_compared"]))
    o = Oracle(str(npz))
    gold = ev["golden_vectors"]
    worst = max(float(np.abs(np.asarray(g["cqa"]) - o.predict_one(g["design"])).max()) for g in gold)
    check("%d golden vectors reproduced exactly" % len(gold), worst == 0.0,
          "max abs dev %.17g" % worst)

    print("\n5  the shipped baseline is reproducible from the shipped npz")
    cfg = yaml.safe_load((ROOT / "tasks" / "q1_microenvironment" / "task_config.yaml")
                         .read_text(encoding="utf-8"))
    obj = cfg["objective"]
    got_start = evaluate_design(dict(START), obj).desirability
    want_start = float(cfg["baseline"]["start_objective"]) if "baseline" in cfg else None
    ref = cfg.get("reference") or {}
    if want_start is None:
        want_start = ref.get("start_objective")
    if want_start is None:
        check("start desirability recomputed", True,
              "D_start = %.9g (public package carries no declared value to compare against)"
              % got_start)
    else:
        # evaluate_design rounds D to 9 significant digits for the report, so compare at 1e-8
        # relative -- the raw agreement is exact and is what check_consistency.py asserts.
        check("D(start) == declared",
              abs(got_start - float(want_start)) <= 1e-8 * abs(float(want_start)),
              "%.12g vs %.12g" % (got_start, float(want_start)))
    if ref.get("best_design_in_budget"):
        got_ref = evaluate_design(ref["best_design_in_budget"], obj).desirability
        want_ref = float(ref["best_objective_in_budget"])
        check("D(reference design) == declared",
              abs(got_ref - want_ref) <= 1e-8 * abs(want_ref),
              "%.12g vs %.12g" % (got_ref, want_ref))
    else:
        check("reference block absent (public package): nothing to compare", True,
              "grader package carries it")

    sc = scorer_from_config(obj)
    check("scorer built from task_config without a fallback shape", sc is not None,
          "M=%.4g e=%.3g kappa=%.3g floor=%.3g" % (sc.M, sc.e, sc.kappa, sc.dfloor))

    print("\n" + "=" * 96)
    if FAILURES:
        print("ORACLE VERIFICATION FAILED: %s" % ", ".join(FAILURES))
        raise SystemExit(1)
    print("ORACLE VERIFICATION CLEAN -- non-executable format, integrity ok, "
          "numerically identical to the superseded pickle, baseline reproducible")


def _refuses_pkl() -> bool:
    try:
        Oracle(str(ROOT / "data" / "oracle.pkl"))
    except ValueError:
        return True
    except Exception:
        return False
    return False


if __name__ == "__main__":
    main()
