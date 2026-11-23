#!/usr/bin/env python
"""Build the delivery archive, and refuse to build it if a credential would ship.

The archive has to contain six classes of deliverable -- the runtime environment declaration, the
workflow framework, the evaluation code, the parameter/environment configs, the task description, and
at least three LLM trajectories -- and it must not contain an API key. Those two requirements fight
each other: every run directory embeds a copy of the API configuration it was run with, so the key
has exactly as many chances to leak as there are runs.

Three independent defences, in order:

  1. NAME-BASED EXCLUSION. configs/api_config*.json holds live keys. Only the *.example.json
     templates are archived. This is the same rule as .gitignore, written once here.
  2. CONTENT SCAN. Every file that is about to be archived is read and searched for key-shaped
     strings. A finding aborts the build -- the zip is not written, not written-then-deleted.
     This is the defence that matters, because it does not depend on my having thought of the right
     filename: a key pasted into a note, a stray shell history, a debug dump would all be caught.
  3. POST-CONDITION. After writing, the archive is reopened and every member is scanned again from
     inside the zip. If anything is found the zip is deleted. Checking the inputs and checking the
     output are not the same check: a bug in the include/exclude logic lives exactly in the gap
     between them.

The scan pattern is the one from scripts/check_consistency.py check 8, deliberately wider than the
two vendors used here (DeepSeek sk-<hex>, DashScope sk-ws-<uuid>), because the point is to catch a
shape, not to enumerate vendors.

Run:
    python scripts/package_for_delivery.py
    python scripts/package_for_delivery.py --output /tmp/K.zip --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------------------------------------
# What never ships.
# ------------------------------------------------------------------------------------------------
EXCLUDE_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", ".pytest_cache", ".mypy_cache",
                ".venv", "node_modules",
                # scripts/test_llm_connection.py drops runs/llm_connection_tests/. It is not a
                # trajectory, and deliverable_classes() already refuses to count it as one -- but
                # the exclusion has to live HERE too, or the directory is archived while going
                # uncounted, and a reviewer opening runs/ sees four run dirs for three models.
                "llm_connection_tests"}
EXCLUDE_NAMES = {".DS_Store", "Thumbs.db", ".env"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pyd", ".so", ".log", ".swp", ".tmp", ".orig", ".rej", ".bak"}

# ------------------------------------------------------------------------------------------------
# THE PUBLIC / GRADER SPLIT (external review item 5).
#
# The earlier delivery was a single archive, so the model-visible material carried the reference
# design, the score anchors and the complete synthetic response mechanism. Two archives are built now:
#
#   grader  the whole package. Runs end to end, reproduces every calibrated number, and is what a
#           reviewer should open.
#   public  exactly what a participating agent may see: the task, the input/output contract, the
#           runtime environment, the harness, the diagnostic tool layer and the examples -- and no
#           response mechanism, no surrogate weights, no reference optimum, no score anchors, no gate
#           measurements, no trajectories.
#
# The public archive is deliberately NOT runnable on its own: src/response_model.py is swapped for a
# stub that documents the grader-side evaluation contract and refuses to invent numbers. That is the
# isolation, not a packaging defect, and PUBLIC_README says so.
# ------------------------------------------------------------------------------------------------
GRADER_ONLY_EXACT = {
    # the synthetic response mechanism and everything derived from it
    "src/agglomeration.py",
    # the full task config, which carries the reference: block (optimum, anchors, gate numbers)
    "tasks/q1_agglomeration/task_config.yaml",
    # reviewer-facing documents that quote the reference design and the gate measurements
    "README.md", "审核说明.md", "整改说明.md", "整改说明2.md", "整改进度.md",
    # designer-side derivations and audits
    "scripts/enumerate_reference_best.py", "scripts/expert_baseline.py",
    # the naive-policy probe: it quotes the gate numbers and it encodes the author's reading of the
    # mechanism table as an explicit direction vector, which is a hint if a participant reads it
    "scripts/measure_naive_policies.py",
    "scripts/check_consistency.py", "scripts/leak_check_question.py",
    "scripts/convert_surrogate_to_npz.py", "scripts/verify_surrogate_migration.py",
    "scripts/make_public_task_config.py", "scripts/package_for_delivery.py",
    "scripts/report_calibration.py",
    "scripts/rescore_run.py",
}
GRADER_ONLY_PREFIXES = (
    "data/",       # surrogate weights, training set, model card, migration evidence
    "answers/",    # the reference enumeration and the expert baselines
    "runs/",       # the shipped model trajectories (their reports carry the reference value)
    # The public build inputs. They belong in the grader archive -- a reviewer must be able to see
    # exactly what the public side receives, and to rebuild the public archive -- but they must not be
    # archived at their own paths inside the public archive, where they would be duplicates of the
    # files they were overlaid onto. scripts/check_consistency.py check 11 reads this directory, which
    # is why it has to ship in the grader archive rather than being withheld from both.
    "public_overlay/",
)
# Never shipped in EITHER archive: the superseded pickle. Kept in the working tree only so the
# migration can be re-measured locally; the delivery must contain no pickled model at all.
NEVER_SHIPPED_PREFIXES = ("data/legacy/",)

# Files whose PUBLIC build differs from the working-tree file. Source -> destination.
PUBLIC_OVERLAY = {
    "public_overlay/src/response_model.py": "src/response_model.py",
    "public_overlay/README.md": "README.md",
}
# In the public build the harness must read the public task config.
PUBLIC_TEXT_SUBSTITUTIONS = {
    "configs/default.yaml": [('task_config_file: "task_config.yaml"',
                              'task_config_file: "task_config.public.yaml"')],
    "configs/default_deepseek.yaml": [('task_config_file: "task_config.yaml"',
                                       'task_config_file: "task_config.public.yaml"')],
    "configs/default_glm.yaml": [('task_config_file: "task_config.yaml"',
                                  'task_config_file: "task_config.public.yaml"')],
    "configs/default_qwen.yaml": [('task_config_file: "task_config.yaml"',
                                   'task_config_file: "task_config.public.yaml"')],
}


def grader_only(rel: str) -> bool:
    return rel in GRADER_ONLY_EXACT or rel.startswith(GRADER_ONLY_PREFIXES)


def incomplete_run(rel: str) -> bool:
    """True for a file inside a run directory that has no final_report.json.

    A crashed, killed or still-running replicate leaves a directory with a prompt snapshot and a
    partial log. Archiving it would put a run with no score into the delivery, and `ls runs/` would
    show more directories than there are trajectories -- the exact discrepancy the packaging report is
    supposed to make impossible. Such directories are excluded and listed as skipped, so an aborted
    replicate is visible rather than shipped.
    """
    parts = rel.split("/")
    if len(parts) < 3 or parts[0] != "runs":
        return False
    return not (ROOT / "runs" / parts[1] / "final_report.json").is_file()


def never_shipped(rel: str) -> bool:
    return rel.startswith(NEVER_SHIPPED_PREFIXES)


def holds_live_credentials(path: Path) -> bool:
    """True for configs/api_config*.json, except the *.example.json templates.

    That covers configs/api_config.json and configs/api_config_{deepseek,glm,qwen,kimi}.json by
    construction rather than by list, so a provider added later is excluded by default -- the safe
    direction. configs/api_config{,_glm,_qwen}.example.json survive: configs/environment.yaml points
    each provider at its own template, and they carry the placeholder PUT_YOUR_..._KEY_HERE.
    """
    if path.parent.name != "configs" or not path.name.startswith("api_config"):
        return False
    return not path.name.endswith(".example.json")


def is_temporary(path: Path) -> bool:
    name = path.name
    return (name in EXCLUDE_NAMES
            or path.suffix in EXCLUDE_SUFFIXES
            or name.startswith("~")
            or (name.startswith(".#"))
            or (name.startswith("#") and name.endswith("#"))
            or name.endswith("~"))


def should_include(path: Path, side: str = "grader") -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    rel = str(path.relative_to(ROOT))
    if any(part in EXCLUDE_DIRS for part in path.relative_to(ROOT).parts):
        return False
    if is_temporary(path) or holds_live_credentials(path):
        return False
    if never_shipped(rel):
        return False
    if incomplete_run(rel):
        return False
    if side == "public" and grader_only(rel):
        return False
    return True


# ------------------------------------------------------------------------------------------------
# What the PUBLIC archive must not contain, checked from inside the written zip.
# ------------------------------------------------------------------------------------------------
def public_forbidden_strings() -> dict:
    """Designer-side strings that must not survive into the public archive.

    Derived from answers/reference_enumeration.json rather than hard-coded, so that a re-derived
    calibration is checked against its own new numbers instead of yesterday's.
    """
    enum_path = ROOT / "answers" / "reference_enumeration.json"
    out = {
        "the module holding the response mechanism": "def run_pbe(",
        "the Rumpf strength prefactor": "sig_pre",
        "the breakage prefactor": "kb=4.2e-6",
        "the surrogate feature list": "FEATS = [",
        "the designer-only enumeration file": "reference_enumeration",
    }
    # -----------------------------------------------------------------------------------------
    # Added 2026-08-01 after the delivery archives were unpacked and grepped by hand. The
    # post-conditions above are derived from answers/reference_enumeration.json, so they catch
    # designer-side NUMBERS -- and they missed three designer-side SENTENCES sitting in module
    # docstrings and comments of files that ship to participants:
    #   * src/design_grid.py named the four slots of the optimum inside the "why budget 6" comment;
    #   * src/feedback.py said the optimum is a "four-variable coordinated move";
    #   * configs/environment.yaml carried a stale build status quoting the measured model scores.
    # All three are fixed; these entries make the failure mode non-recurring. The slot-count words
    # are banned as phrases rather than as numbers because that is the form the leak took.
    # -----------------------------------------------------------------------------------------
    out.update({
        "the arity of the reference optimum (prose form)": "four-variable coordinated",
        "the slot set of the cheapest coordinated move": "S3 1 + S5 1 + S9 2 + S10 2",
        "measured model scores in a participant-visible file": "9/100 each",
        "the naive-policy probe output (grader-only)": "naive_policy_probes",
    })
    # NOT banned, deliberately: the NAMES of the grader-only scripts. configs/environment.yaml is the
    # authoritative environment declaration and ships on both sides; it has always listed
    # check_consistency.py / leak_check_question.py / enumerate_reference_best.py as reproduction
    # commands. A script name is not a measurement, and banning one grader-only name while leaving
    # three others visible would be a rule nobody could apply consistently. What is banned is every
    # designer-side FILE OF NUMBERS those scripts write.
    # `best_objective_in_budget` is checked only in DATA files. It is also the name of the key
    # src/scoring.py reads out of a grader-supplied config and one entry of the tool leak guard's
    # forbidden-key list, and a key name in code leaks no value -- banning it there would force the
    # public harness to be a different program from the graded one.
    out_data_only = {"the reference block key (data files only)": "best_objective_in_budget"}
    if enum_path.is_file():
        enum = json.loads(enum_path.read_text(encoding="utf-8"))
        ref = float(enum["reference_best"])
        for nd in (4, 6, 9):
            out[f"reference objective to {nd} dp"] = f"{ref:.{nd}f}"
        out["the enumeration's own change annotation"] = enum["reference_changes"].split(";")[0].strip()
        for k, v in enum["reference_responses"].items():
            out[f"reference response {k}"] = f"{float(v):.4f}"
        g = enum["gates"]
        for name, v in (("improving-design count", g["g3"]["n_strictly_improving"]),
                        ("feasible in-budget count", g["g3"]["n_feasible_in_budget"]),
                        ("G2 feasible count", g["g2"]["n_feasible_and_in_budget"])):
            out[f"gate measurement: {name}"] = str(int(v))
            # Same number, thin-space-grouped, which is how prose writes it. The 2026-08-01 unpack
            # audit found "96 124" surviving in two participant-visible files precisely because only
            # the unspaced form was banned.
            out[f"gate measurement: {name} (grouped)"] = f"{int(v):,}".replace(",", " ")
    return out, out_data_only


# ------------------------------------------------------------------------------------------------
# The credential scan.
# ------------------------------------------------------------------------------------------------
KEY_SHAPE = re.compile(r"sk-[A-Za-z0-9_.\-]{16,}")
# Strings that match the shape but cannot be secrets. There is deliberately NO allow-list of literal
# key-shaped strings: the placeholders in configs/api_config*.example.json are
# PUT_YOUR_<VENDOR>_API_KEY_HERE, which does not start with "sk-" and so never matches in the first
# place. That choice is load-bearing for a reviewer, because it means a plain
# `grep -rE 'sk-[A-Za-z0-9_.\-]{16,}'` over the extracted archive must return NOTHING -- an
# allow-list here would put key-shaped strings into the delivery and make that check noisy, so the
# one clean check a reviewer will actually run would stop being clean.
PLACEHOLDER_MARKERS = ("PUT_YOUR", "YOUR_", "<redacted>", "REDACTED", "xxxxxxxx", "EXAMPLE")

BINARY_SUFFIXES = {".joblib", ".npz", ".png", ".pdf", ".zip", ".jpg", ".gz"}


def scan(paths_and_text) -> list:
    """[(label, matched_sample)] for everything that looks like a live key."""
    findings = []
    for label, text in paths_and_text:
        for match in KEY_SHAPE.findall(text):
            if any(marker in match for marker in PLACEHOLDER_MARKERS):
                continue
            findings.append((label, match[:10] + "..." + f"(len {len(match)})"))
    return findings


def read_text(path: Path) -> str:
    if path.suffix in BINARY_SUFFIXES:
        # Still scanned: a key can be embedded in a pickle. Read as latin-1 so every byte maps.
        return path.read_bytes().decode("latin-1", errors="ignore")
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


# ------------------------------------------------------------------------------------------------
# The six deliverable classes. Asserted, not assumed: a zip that is missing one is not a delivery.
# ------------------------------------------------------------------------------------------------
def deliverable_classes(rels: set, side: str = "grader") -> dict:
    """The six deliverable classes the brief requires.

    They are asserted on the GRADER archive, which is the delivery. The public archive is a subset of
    it by construction (it omits the response mechanism and the trajectories on purpose), so asserting
    the same six classes there would be asserting the opposite of the isolation requirement.
    """
    def any_of(*names):
        return sorted(r for r in rels if r in names)

    def under(prefix, suffix=""):
        return sorted(r for r in rels if r.startswith(prefix) and r.endswith(suffix))

    runs = {r.split("/")[1] for r in rels
            if r.startswith("runs/") and "/" in r[5:] and r.split("/")[1] != "llm_connection_tests"}
    trajectories = sorted(d for d in runs if f"runs/{d}/trajectory.jsonl" in rels)
    if side == "public":
        return {
            "1 runtime environment": any_of("requirements.txt", "configs/environment.yaml"),
            "2 workflow framework": any_of("src/main.py", "src/llm_client.py", "src/parser.py",
                                           "src/feedback.py", "src/markdown_report.py",
                                           "src/utils.py", "src/tools.py"),
            "3 public task definition (NO response mechanism)":
                any_of("src/design_grid.py", "src/design_space.py", "src/evaluator.py",
                       "src/scoring.py", "src/response_model.py"),
            "4 parameter / environment config": under("configs/", ".yaml") + under("configs/", ".json"),
            "5 task description markdown":
                any_of("tasks/q1_agglomeration/question.md",
                       "tasks/q1_agglomeration/task_config.public.yaml", "README.md"),
        }
    return {
        "1 runtime environment": any_of("requirements.txt", "configs/environment.yaml"),
        "2 workflow framework": any_of("src/main.py", "src/llm_client.py", "src/parser.py",
                                       "src/feedback.py", "src/markdown_report.py", "src/utils.py",
                                       "src/tools.py"),
        "3 evaluation code": any_of("src/evaluator.py", "src/scoring.py", "src/agglomeration.py",
                                    "src/design_space.py", "src/design_grid.py",
                                    "src/response_model.py"),
        "4 parameter / environment config": under("configs/", ".yaml") + under("configs/", ".json"),
        "5 task description markdown": any_of("tasks/q1_agglomeration/question.md",
                                              "tasks/q1_agglomeration/task_config.yaml",
                                              "README.md", "审核说明.md"),
        "6 LLM trajectories (>= 3 models)": trajectories,
    }


def trajectory_summary(rels: set) -> list:
    out = []
    for run in sorted(p for p in (ROOT / "runs").iterdir() if p.is_dir()):
        report = run / "final_report.json"
        if not report.is_file() or f"runs/{run.name}/final_report.json" not in rels:
            continue
        data = json.loads(report.read_text(encoding="utf-8"))
        api_used = run / "api_config_used.json"
        model = json.loads(api_used.read_text(encoding="utf-8")).get("model") \
            if api_used.is_file() else None
        score = data.get("score") or {}
        out.append({
            "run_dir": run.name,
            "model": model,
            "total_score": score.get("total_score"),
            "achievement": (score.get("optimality_score_details") or {}).get("achievement_fraction"),
            "rounds": len(data.get("candidates") or []),
            "readable_txt": f"runs/{run.name}/trajectory.readable.txt" in rels,
        })
    return out


def build(side: str, out: Path, dry_run: bool) -> int:
    """Build one archive. `side` is "grader" or "public"."""
    everything = sorted(p for p in ROOT.rglob("*") if p.is_file())
    files = [p for p in everything if should_include(p, side)]
    rels = {str(p.relative_to(ROOT)) for p in files}

    # PUBLIC build: swap in the stub response model and the participant-facing README, and repoint the
    # harness configs at the public task config. Every substitution is asserted to have applied.
    contents: dict = {}
    if side == "public":
        for src_rel, dst_rel in PUBLIC_OVERLAY.items():
            src = ROOT / src_rel
            if not src.is_file():
                print(f"ABORTED -- the public overlay file {src_rel} is missing")
                return 1
            contents[dst_rel] = src.read_bytes()
            rels.add(dst_rel)
        for rel, subs in PUBLIC_TEXT_SUBSTITUTIONS.items():
            if rel not in rels:
                continue
            body = (ROOT / rel).read_text(encoding="utf-8")
            for old, new in subs:
                if old not in body:
                    print(f"ABORTED -- public substitution did not apply in {rel}: {old!r}")
                    return 1
                body = body.replace(old, new)
            contents[rel] = body.encode("utf-8")
        # the public task config must exist and must be current
        if "tasks/q1_agglomeration/task_config.public.yaml" not in rels:
            print("ABORTED -- tasks/q1_agglomeration/task_config.public.yaml is missing; run "
                  "scripts/make_public_task_config.py")
            return 1

    print("=" * 96)
    print(f"1  [{side}] selection: {len(files)} of {len(everything)} files"
          + (f"  (+{len(contents)} overlaid)" if contents else ""))
    print("=" * 96)
    n_grader_only = 0
    for p in everything:
        if p in files:
            continue
        rel = str(p.relative_to(ROOT))
        if never_shipped(rel):
            why = "never shipped (superseded pickle / build input)"
        elif incomplete_run(rel):
            why = "INCOMPLETE RUN (no final_report.json -- aborted or still running)"
        elif holds_live_credentials(p):
            why = "holds live credentials"
        elif is_temporary(p):
            why = "temporary/build artefact"
        elif side == "public" and grader_only(rel):
            why = "GRADER ONLY (response mechanism / reference answer / trajectory)"
            n_grader_only += 1
        else:
            why = "excluded directory"
        print(f"  skip  {rel}   ({why})")
    if side == "public":
        print(f"       {n_grader_only} files withheld from the public archive as grader-only")

    def body_of(rel: str) -> str:
        if rel in contents:
            return contents[rel].decode("latin-1", errors="ignore")
        return read_text(ROOT / rel)

    print("\n" + "=" * 96)
    print(f"2  [{side}] credential scan of every file that would be archived")
    print("=" * 96)
    findings = scan((rel, body_of(rel)) for rel in sorted(rels))
    if findings:
        print("ABORTED -- key-shaped strings found; the archive was NOT written:")
        for label, sample in findings:
            print(f"  {label}: {sample}")
        return 1
    print(f"  clean: no match for {KEY_SHAPE.pattern} in any of the {len(rels)} files")

    print("\n" + "=" * 96)
    print(f"3  [{side}] deliverable classes")
    print("=" * 96)
    classes = deliverable_classes(rels, side)
    missing = []
    for name, members in classes.items():
        need = 3 if name.startswith("6") else 1
        state = "OK  " if len(members) >= need else "MISS"
        if len(members) < need:
            missing.append(name)
        shown = ", ".join(members[:4]) + (f", +{len(members) - 4} more" if len(members) > 4 else "")
        print(f"  {state} {name}: {len(members)} -> {shown}")

    if side == "grader":
        trajectories = trajectory_summary(rels)
        print()
        for tr in trajectories:
            print(f"       {tr['model'] or '?':16s} score={tr['total_score']:<4} "
                  f"achievement={tr['achievement']:<10} rounds={tr['rounds']} "
                  f"readable_txt={tr['readable_txt']}  ({tr['run_dir']})")
        models = {tr["model"] for tr in trajectories if tr["model"]}
        if len(models) < 3:
            missing.append(f"only {len(models)} distinct model(s), the brief requires >= 3")

    if missing:
        print("\nABORTED -- incomplete delivery; the archive was NOT written:")
        for m in missing:
            print(f"  {m}")
        return 1

    if dry_run:
        print("\ndry run: nothing written")
        return 0

    print("\n" + "=" * 96)
    print(f"4  [{side}] writing {out}")
    print("=" * 96)
    root_name = ROOT.name if side == "grader" else f"{ROOT.name}_public"
    tmp = out.with_suffix(".zip.partial")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in sorted(rels):
            arc = str(Path(root_name) / rel)
            if rel in contents:
                zf.writestr(arc, contents[rel])
            else:
                zf.write(ROOT / rel, arc)

    # Defence 3: scan the archive from the inside. Checking the inputs is not checking the output.
    with zipfile.ZipFile(tmp) as zf:
        members = zf.namelist()
        member_text = {n: zf.read(n).decode("latin-1", errors="ignore") for n in members}
    again = scan(member_text.items())
    if again:
        tmp.unlink()
        print("ABORTED -- key-shaped strings inside the written archive; it has been deleted:")
        for label, sample in again:
            print(f"  {label}: {sample}")
        return 1

    # Defence 4 (public only): the isolation post-condition, checked from inside the written zip.
    if side == "public":
        print("\n" + "=" * 96)
        print("5  [public] isolation post-condition, checked from INSIDE the written archive")
        print("=" * 96)
        leaks = []
        everywhere, data_only = public_forbidden_strings()
        for what, needle in everywhere.items():
            hits = [n for n, body in member_text.items() if needle in body]
            if hits:
                leaks.append(f"{what} ({needle!r}) in {hits[:3]}")
            else:
                print(f"  OK   absent: {what}")
        for what, needle in data_only.items():
            hits = [n for n, body in member_text.items()
                    if needle in body and not n.endswith(".py")]
            if hits:
                leaks.append(f"{what} ({needle!r}) in {hits[:3]}")
            else:
                print(f"  OK   absent: {what}")
        # A grader-only path may legitimately reappear if the public overlay supplies its own build of
        # it (README.md is the case): the file present is then the public build, not the grader one.
        overlaid = set(PUBLIC_OVERLAY.values())
        for rel in sorted(GRADER_ONLY_EXACT) + list(GRADER_ONLY_PREFIXES):
            if rel in overlaid:
                continue
            hits = [n for n in members if n.startswith(f"{root_name}/{rel}")]
            if hits:
                leaks.append(f"grader-only path {rel} present as {hits[:3]}")
        # ... and the overlaid ones must actually BE the overlay build
        for src_rel, dst_rel in PUBLIC_OVERLAY.items():
            want = (ROOT / src_rel).read_bytes().decode("latin-1", errors="ignore")
            if member_text.get(f"{root_name}/{dst_rel}") != want:
                leaks.append(f"{dst_rel} in the archive is not the public overlay build")
        print(f"  OK   the {len(PUBLIC_OVERLAY)} overlaid files are the public builds")
        if any(n.endswith(".joblib") or n.endswith(".pkl") for n in members):
            leaks.append("a pickled model file is present")
        if leaks:
            tmp.unlink()
            print("\nABORTED -- the public archive is not isolated; it has been deleted:")
            for m in leaks:
                print(f"  {m}")
            return 1
        print(f"  OK   none of the {len(GRADER_ONLY_EXACT) + len(GRADER_ONLY_PREFIXES)} grader-only "
              f"paths is present")
        print("  OK   no pickled model file is present")
        # and the stub really is the stub
        stub = member_text.get(f"{root_name}/src/response_model.py", "")
        if 'BUILD = "public"' not in stub or "ResponseModelUnavailable" not in stub:
            tmp.unlink()
            print("ABORTED -- src/response_model.py in the public archive is not the stub build")
            return 1
        print("  OK   src/response_model.py is the stub build (no mechanism)")

    tmp.replace(out)
    size_kb = out.stat().st_size / 1024
    print(f"\n  archive re-scanned from the inside: clean ({len(members)} members)")
    print(f"  {out}   {size_kb:.1f} KiB   {len(rels)} files")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["grader", "public", "both"], default="both",
                    help="which archive(s) to build (default: both)")
    ap.add_argument("--output", default=None,
                    help="explicit output path; only valid with a single --side")
    ap.add_argument("--dry-run", action="store_true", help="scan and report, write nothing")
    args = ap.parse_args()

    sides = ["grader", "public"] if args.side == "both" else [args.side]
    if args.output and len(sides) > 1:
        print("--output can only be combined with a single --side")
        return 1

    for side in sides:
        default = (ROOT.parent / f"{ROOT.name}_grader.zip" if side == "grader"
                   else ROOT.parent / f"{ROOT.name}_public.zip")
        out = Path(args.output) if args.output else default
        rc = build(side, out, args.dry_run)
        if rc:
            print(f"\nFAILED while building the {side} archive")
            return rc
        print(f"\n--- {side} archive done ---\n")

    print("PACKAGED: grader archive (complete delivery) and public archive (model-visible only).")
    print("Reviewer entry point: unzip the GRADER archive, then "
          "$PY scripts/check_consistency.py (must end CONSISTENT).")
    print("The public archive is intentionally NOT runnable on its own: src/response_model.py is a "
          "stub and the response mechanism is grader-side.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
