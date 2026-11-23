"""Package this benchmark into TWO delivery zips, with a hard stop on credential leakage.

Common review point A of the 2026-07-29 audit: the reference answer must not sit on the side a graded
model (or an agent with file access) can read. So this script writes two archives from one tree:

  <name>_public.zip   what a participant needs in order to RUN a model: the task text, the harness,
                      the evaluator, the tool layer, the oracle, the configs, the data card. It
                      carries NO reference optimum -- the `reference:` block of task_config.yaml is
                      replaced by an explanatory stub -- and no answers/, no designer enumeration
                      scripts, no run directories, no reviewer notes. A run produced here records
                      every submission and leaves `score` unset.
  <name>_grader.zip   the complete package: everything above plus answers/, the enumeration and gate
                      scripts, the reference block, the reviewer notes, the changelog and the >= 3
                      model trajectories. scripts/grade_run.py scores a public run here, and the
                      score is a pure function of the recorded submissions, so the two sides agree by
                      construction.

The public archive is then ATTACKED rather than trusted: every member is grepped for the reference
design in several renderings, the reference desirability at several precisions, every gate anchor and
every distinctive string from answers/. Anything found aborts the write.

Excluded from BOTH archives, by rule and by name:
  * every configs/api_config*.json that is NOT a *.example.json template -- i.e. the live
    api_config.json and api_config_{deepseek,glm,qwen,kimi}.json. Only the placeholder templates ship;
  * __pycache__ / .pyc, editor and OS scratch files (*~, *.swp, *.tmp, *.bak, .DS_Store),
    .git, .pytest_cache, .ipynb_checkpoints.

Two secret scans per archive, not one:
  1. before writing, every file that is about to be archived is read and grepped for key shapes;
  2. after writing, the archive is REOPENED and every member is grepped again from the zip's own
     bytes, then the zip is deleted if anything is found. A pre-write scan only proves that the set
     of paths the scanner walked was clean; it does not prove that the set the writer walked was the
     same set.

WHICH DIRECTORIES COUNT AS "the >= 3 LLM trajectories"
  Exactly the run directories under runs/, and they ship in the GRADER archive. That definition lives
  once, in is_trajectory_run_dir(), and both the archive traversal and the trajectory count call it;
  verify_run_accounting() aborts if the two ever disagree. Sister problem K put the rule in the
  counting function only: the summary kept printing 3 while the zip had grown a fourth run directory
  nobody had counted.

Run:
    python scripts/package_for_delivery.py                 # writes both archives
    python scripts/package_for_delivery.py --side public
    python scripts/package_for_delivery.py --side grader --output /path/to/out.zip
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Files that hold live credentials and must never be archived.
EXCLUDE_NAMES = {".DS_Store", "Thumbs.db"}
EXCLUDE_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", ".pytest_cache", ".mypy_cache",
                ".venv", "node_modules"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".swp", ".swo", ".tmp", ".temp", ".bak", ".orig", ".rej"}

# Belt as well as braces: the rule below already covers these, but the delivery brief names them, so
# a rule that silently stopped matching would be caught by an explicit list too.
NAMED_SECRET_FILES = {"api_config.json", "api_config_deepseek.json", "api_config_glm.json",
                      "api_config_qwen.json", "api_config_kimi.json"}
# The one credentials file that MUST ship, so that a reviewer can see the expected shape.
REQUIRED_TEMPLATE = "configs/api_config.example.json"

# ------------------------------------------------------------------------------------------------
# Run directories: what is a trajectory, and what is merely archived evidence.
# ------------------------------------------------------------------------------------------------
TRAJECTORY_ROOT = "runs"
# Archived so a cross-reviewer can check 审核说明.md's recalibration section against the raw records,
# but explicitly NOT trajectories of this problem: they were graded by a ruler the code no longer
# implements. Each such directory must carry a README saying so, or packaging aborts.
# Ruler 2.0.0 deleted runs_old_hardzero/: exactly one scoring version's results ship (audit finding
# #4). The mechanism is kept so that a future evidence tree cannot be archived without a decision.
EVIDENCE_ROOTS: set[str] = set()

# ------------------------------------------------------------------------------------------------
# THE PUBLIC / GRADER SPLIT (common review point A).
#
# One list, used by the traversal AND by the post-write attack on the public archive, so the two
# cannot drift. Anything here is grader-only because it contains, or can be used to reconstruct, the
# reference answer or the band anchors.
# ------------------------------------------------------------------------------------------------
GRADER_ONLY_TOP = {
    "answers",          # reference enumeration, gate probes, naive-answer ceilings, boundary cases
    "runs",             # the trajectories; their final_report.json carries the reference objective
    "审核说明.md",       # reviewer notes: quotes the reference design and its desirability
    "整改说明.md",       # audit-remediation report, same
    "整改进度.md",       # working checkpoint
}
# Individual files, not whole trees. These are grader-only because they either ARE the response
# surface, or they carry difficulty-calibration numbers that question.md does not disclose to the
# model (gate ceilings by family, shortcut achievements, the full score histogram, band populations).
GRADER_ONLY_FILES = {
    "data/oracle.npz",                   # the surrogate WEIGHTS -- the response surface itself
    "data/oracle_schema.json",           # its exact input/output contract + sha256
    "data/oracle_migration_check.json",  # 64 recorded (design -> CQA) pairs
    "README.md",                         # full calibration table, histogram, band populations
    "CHANGELOG.md",                      # per-version gate numbers and shortcut achievements
    "configs/environment.yaml",          # verification commands with their expected gate outputs
}
# src/response_model.py is NOT listed above: it is REPLACED by a stub in the public archive, the way
# task_config.yaml's reference block is. See public_response_model().
BOUNDARY_MODULE = "src/response_model.py"

GRADER_ONLY_SCRIPTS = {
    "enumerate_reference_best.py",   # derives the reference optimum
    "measure_naive_answers.py",      # shortcut ceilings
    "measure_trap_families.py",
    "measure_pregates.py",
    "audit_recalibration.py",
    "check_consistency.py",          # embeds every anchor
    "leak_check_question.py",        # reads answers/
    "test_boundary_cases.py",        # prints the reference design
    "rescore_run.py",                # needs the reference block
    "grade_run.py",                  # the grading entry point
    "test_tool_protocol.py",         # drives run_benchmark, so it needs a live api_config file
    "verify_oracle_npz.py",          # loads the weights
    "trajectory_to_txt.py",          # re-evaluates a run, so it needs the response model
    "trajectory_to_md.py",           # same
    "migrate_oracle_to_npz.py",      # one-shot designer converter; reads the retired pickle
    "package_for_delivery.py",       # this file
}
PUBLIC_NOTE_NAME = "PUBLIC_PACKAGE_NOTE.md"


def is_grader_only(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if rel.parts[0] in GRADER_ONLY_TOP:
        return True
    if str(rel) in GRADER_ONLY_FILES:
        return True
    if rel.parts[0] == "scripts" and rel.name in GRADER_ONLY_SCRIPTS:
        return True
    return False


def public_response_model() -> str:
    """The PUBLIC stub of src/response_model.py.

    Same names, same import surface, every callable raising. Generated here rather than kept as a
    second file on disk so the two halves cannot drift: if a name is added to the real module and not
    to this stub, the public archive's import check fails at packaging time.
    """
    return '''"""RESPONSE MODEL -- PUBLIC STUB. The real module ships only in the grader package.

This package deliberately does NOT contain the response surface. In the grader package this module
holds the invented per-substance property table, the closed-form synthetic CQA simulator, and the
fitted surrogate that is the scoring ruler, loaded from data/oracle.npz. None of those is here, and
data/oracle.npz is not here either.

Why: the reference optimum is not the only thing worth hiding. An agent that can EVALUATE candidates
offline does not need to know the reference value to win -- it can optimise the response surface
directly and skip the five rounds of reasoning the benchmark is measuring. So the surface sits behind
the same boundary as the reference answer.

Consequence, and it is intended: this package can render the task, validate a formulation against the
declared design space and the change budget, run the tool layer, and record a trajectory -- but it
CANNOT produce a CQA value, a desirability, or a score. Evaluation is run by the party holding the
grader package:

    python scripts/grade_run.py runs/<run_dir>        # inside the grader package

Every name below exists so that importing the harness still works. Calling any of them raises.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

SEED = 20260727
DEFAULT_ORACLE_PATH = "<not shipped in the public package>"
ORACLE_SCHEMA_PATH = "<not shipped in the public package>"
ORACLE_ARRAY_KEYS = ("W0", "b0", "W1", "b1", "W2", "b2",
                     "x_mean", "x_scale", "y_mean", "y_scale")

_MSG = ("the response model is not shipped in the public package. The per-substance property table, "
        "the CQA simulator and the fitted surrogate (data/oracle.npz) live only in the grader "
        "package, so this package cannot measure a formulation, compute a desirability, or score a "
        "run. Use the grader package: python scripts/grade_run.py runs/<run_dir>")


class ResponseModelUnavailable(RuntimeError):
    """Raised by every callable in this stub."""


# The property tables are absent, not empty-by-accident: anything that indexes them fails loudly.
SUB: Dict[str, Dict[str, float]] = {}
MCC_GRADE: Dict[str, Dict[str, float]] = {}


def _props(d: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    raise ResponseModelUnavailable(_MSG)


def cqa_true(d: Dict[str, Any]) -> np.ndarray:
    raise ResponseModelUnavailable(_MSG)


def feature_vector(d: Dict[str, Any]) -> np.ndarray:
    raise ResponseModelUnavailable(_MSG)


def _quiet_matmul(fn):
    raise ResponseModelUnavailable(_MSG)


class Oracle:
    """Not available in the public package. Constructing one raises."""

    def __init__(self, path: str | None = None) -> None:
        raise ResponseModelUnavailable(_MSG)

    def predict_matrix(self, X: np.ndarray) -> np.ndarray:
        raise ResponseModelUnavailable(_MSG)

    def predict(self, designs: Sequence[Dict[str, Any]]) -> np.ndarray:
        raise ResponseModelUnavailable(_MSG)

    def predict_one(self, design: Dict[str, Any]) -> np.ndarray:
        raise ResponseModelUnavailable(_MSG)


def get_oracle(path: str | None = None) -> Oracle:
    raise ResponseModelUnavailable(_MSG)


def random_designs(n: int, rng: "np.random.Generator") -> List[Dict[str, Any]]:
    raise ResponseModelUnavailable(_MSG)


def train_oracle(n: int = 60000, out_path: str | None = None):
    raise ResponseModelUnavailable(_MSG)
'''


def public_task_config(text: str) -> str:
    """Return task_config.yaml with the `reference:` block replaced by an explanatory stub.

    Text surgery rather than a YAML round-trip on purpose: the file's comments ARE its
    documentation, and a round-trip would delete all of them. The block runs from the line
    `reference:` to the next line that starts a top-level key.
    """
    lines = text.splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.startswith("reference:"))
    end = next(i for i in range(start + 1, len(lines))
               if lines[i][:1] not in (" ", "#", "\n", "") and ":" in lines[i])
    stub = (
        "# -------------------------------------------------------------------------------------------------\n"
        "# REFERENCE BLOCK REMOVED FROM THE PUBLIC PACKAGE (audit remediation, common review point A).\n"
        "#\n"
        "# The reference optimum, its design, the achievement anchors, the difficulty-gate values and the\n"
        "# score histogram are the answer to this task, so they live only in the grader package. This\n"
        "# package can run a model and record every submission; it cannot compute a score, and\n"
        "# src/main.py says so in final_report.json -> score.note.\n"
        "#\n"
        "# To score a run produced here, copy the run directory into the grader package and run:\n"
        "#     python scripts/grade_run.py runs/<run_dir>\n"
        "# The score is a pure function of the submitted formulations, so it is exactly reproducible and\n"
        "# needs no API call.\n"
        "# -------------------------------------------------------------------------------------------------\n"
        "reference:\n"
        "  available_in_this_package: false\n"
        "  held_by: \"grader package -- tasks/q1_microenvironment/task_config.yaml\"\n"
        "  grade_with: \"python scripts/grade_run.py runs/<run_dir>\"\n"
        "\n")
    return "".join(lines[:start]) + stub + "".join(lines[end:])


PUBLIC_NOTE = """# Public package -- what is here, what is not, and why it cannot score

This is the **participant-facing / model-visible** half of the delivery. It carries the task, the
contracts and the harness. It does **not** carry the response surface, and it **cannot compute a
CQA, a desirability, or a score**. That is deliberate; see "Why this package cannot measure" below.

**The data is synthetic.** Read `data/DATA_CARD.md` before quoting any result: there is no real
product, no measurement and no regulatory meaning behind any number in this benchmark.

## What is here

- `tasks/q1_microenvironment/question.md` -- the task text, including all five specification
  windows, the complete level grid for the twelve conditions, every hard constraint, the release
  gate and the scoring bands
- `tasks/q1_microenvironment/task_config.yaml` -- the machine-readable twin. Its `reference:` block
  is an explanatory stub
- `src/` -- the harness (`main.py`, `llm_client.py`, `parser.py`, `utils.py`,
  `markdown_report.py`), the design space and budget arithmetic (`formulation.py`,
  `design_space.py`), the evaluator and scorer logic (`evaluator.py`, `scoring.py`,
  `feedback.py`), and the diagnostic tool layer (`tools.py`)
- `src/response_model.py` -- **a stub**. In the grader package this is the boundary module that
  holds the excipient property table, the CQA simulator and the fitted surrogate. Here every
  callable raises `ResponseModelUnavailable`
- `configs/default*.yaml`, `configs/api_config*.example.json`, `configs/README.md`,
  `requirements.txt`
- `data/DATA_CARD.md`
- `scripts/test_llm_connection.py`

## What is deliberately absent

| Absent | Because |
|---|---|
| `data/oracle.npz` and its schema / migration record | these ARE the response surface |
| the excipient property table and the CQA simulator (`src/response_model.py` internals) | same |
| `answers/` -- exhaustive enumeration, reference optimum, every gate probe | the answer |
| the `reference:` block of `task_config.yaml` | reference design, reference desirability, achievement anchors, gate values, score histogram |
| the designer scripts that derive or assert those numbers | same |
| difficulty-calibration notes: shortcut-family ceilings, band populations, the score histogram | naming and valuing the strongest offline shortcut is a strategy hint |
| `README.md`, `CHANGELOG.md`, `审核说明.md`, and the shipped model trajectories | reviewer-side documents that quote the above |

## Why this package cannot measure

Hiding the reference value is not enough. An agent that can **evaluate** candidate formulations
offline does not need to know the reference in order to win -- it can optimise the response surface
directly and skip the multi-round reasoning the benchmark is measuring. So the surface sits behind
the same boundary as the reference answer: the weights, the simulator and the property table live
only in the grader package, reached through the single boundary module `src/response_model.py`.

What this package can still do: render the exact prompt a graded model sees, validate a formulation
against the declared design space and the change budget, run the diagnostic tool layer, and record a
complete trajectory. What it cannot do: return a CQA. `src/main.py` says so out loud, and
`evaluator.evaluate_reference_start` raises `ResponseModelUnavailable` on the first call.

Evaluation is run by the party holding the grader package.

## Environment

Any CPython 3.10-3.12. Only `numpy` and `PyYAML` are needed for the offline parts; `openai` is
needed to drive a model. `scikit-learn` is **not** required. Nothing is downloaded at run time.

```bash
PY=python
$PY -m pip install -r requirements.txt
cp configs/api_config.example.json configs/api_config_deepseek.json   # then paste your key
```

## Grading

Hand the run directory to the grader side, which runs:

```bash
$PY scripts/grade_run.py runs/<run_dir>
```

The score is a pure function of the formulations recorded in `trajectory.jsonl`, so grading needs no
API call and is exactly reproducible.
"""



def run_root_of(rel: Path) -> str | None:
    """First path component, if it looks like a run tree ('runs', 'runs_old_hardzero', ...)."""
    first = rel.parts[0] if rel.parts else ""
    return first if first == TRAJECTORY_ROOT or first.startswith(TRAJECTORY_ROOT + "_") else None


def is_trajectory_run_dir(path: Path) -> bool:
    """THE definition of 'one of the >= 3 LLM trajectories', used by both callers.

    A directory directly under runs/ that holds a final_report.json. Anything under an evidence root
    is deliberately excluded -- see the module docstring.
    """
    if not path.is_dir():
        return False
    rel = path.relative_to(ROOT)
    if len(rel.parts) != 2 or rel.parts[0] != TRAJECTORY_ROOT:
        return False
    return (path / "final_report.json").is_file()


def is_secret_config(path: Path) -> bool:
    """True for api_config*.json except the tracked example templates.

    Any configs/api_config*.example.json is a template (there is one per provider) and carries a
    PUT_YOUR... placeholder; everything else matching api_config*.json may hold a live key and is
    excluded from the archive.
    """
    if path.name in NAMED_SECRET_FILES:
        return True
    if path.parent.name != "configs":
        return False
    if not path.name.startswith("api_config"):
        return False
    return not path.name.endswith(".example.json")


def is_scratch(path: Path) -> bool:
    return (path.name in EXCLUDE_NAMES or path.suffix in EXCLUDE_SUFFIXES
            or path.name.endswith("~") or path.name.startswith(".#"))


def should_include(path: Path) -> bool:
    if not path.is_file():
        return False
    if any(part in EXCLUDE_DIRS for part in path.parts):
        return False
    if is_scratch(path):
        return False
    if is_secret_config(path):
        return False
    return True


# Key shapes. The first two are the exact shapes of the providers used here -- DeepSeek (sk-<32 hex>)
# and DashScope (sk-ws-...). The last is the deliberately over-broad shape the delivery brief asks
# for; it is wider than either real key, which is the point: an abort on a false positive costs a
# minute, a shipped key cannot be recalled.
KEY_PATTERNS = [
    re.compile(r"sk-[0-9a-f]{32}"),
    re.compile(r"sk-ws-[A-Za-z0-9._\-]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_.\-]{16,}"),
]
PLACEHOLDER_MARKERS = ("PUT_YOUR", "<redacted>", "your_api_key", "API-key", "YOUR_KEY")


def find_keys(text: str) -> list[str]:
    hits = []
    for pattern in KEY_PATTERNS:
        for match in pattern.findall(text):
            if any(marker in match for marker in PLACEHOLDER_MARKERS):
                continue
            hits.append(match)
    return hits


def scan_for_secrets(files: list[Path]) -> list[tuple[Path, str]]:
    findings = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for match in find_keys(text):
            findings.append((path, match[:12] + "..."))
    return findings


def scan_archive(zip_path: Path) -> list[tuple[str, str]]:
    """Grep the written archive from its own bytes. See the module docstring for why."""
    findings = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            try:
                text = zf.read(name).decode("utf-8", errors="ignore")
            except Exception:
                continue
            for match in find_keys(text):
                findings.append((name, match[:12] + "..."))
    return findings


def collect_trajectories() -> list[dict]:
    summaries = []
    runs_dir = ROOT / TRAJECTORY_ROOT
    if not runs_dir.is_dir():
        return summaries
    for run in sorted(p for p in runs_dir.iterdir() if is_trajectory_run_dir(p)):
        report = run / "final_report.json"
        api_used = run / "api_config_used.json"
        data = json.loads(report.read_text(encoding="utf-8"))
        model = None
        if api_used.is_file():
            model = json.loads(api_used.read_text(encoding="utf-8")).get("model")
        score = data.get("score", {})
        summaries.append({
            "run_dir": run.name,
            "model": model,
            "total_score": score.get("total_score"),
            "constraint_points": score.get("constraint_points"),
            "optimality_score": score.get("optimality_score"),
            "achievement": (score.get("optimality_score_details") or {}).get("achievement_fraction"),
            "ruler_version": str(data.get("ruler_version") or score.get("ruler_version")
                                 or "unversioned"),
            "has_conversation_md": (run / "conversation.md").is_file(),
            "has_trajectory_jsonl": (run / "trajectory.jsonl").is_file(),
        })
    return summaries


def verify_run_accounting(files: list[Path], trajectories: list[dict]) -> list[str]:
    """Make the archive traversal and the trajectory count answer to the same rule.

    Returns a list of complaints (empty == consistent). Three ways they can disagree, all fatal:
      * a run tree is archived under a root that is neither runs/ nor a declared evidence root, so
        nobody decided whether it is a trajectory;
      * a directory the archive puts under runs/ is not in the counted set, or vice versa;
      * an evidence root ships without a README explaining that it is not a trajectory.
    """
    problems: list[str] = []
    archived_by_root: dict[str, set[str]] = {}
    for p in files:
        rel = p.relative_to(ROOT)
        root = run_root_of(rel)
        if root is None:
            continue
        if root != TRAJECTORY_ROOT and root not in EVIDENCE_ROOTS:
            problems.append(f"{root}/ is archived but is neither runs/ nor a declared evidence root; "
                            f"add it to EVIDENCE_ROOTS or stop archiving it")
            continue
        if len(rel.parts) >= 3:
            archived_by_root.setdefault(root, set()).add(rel.parts[1])

    archived_runs = archived_by_root.get(TRAJECTORY_ROOT, set())
    counted = {t["run_dir"] for t in trajectories}
    for name in sorted(archived_runs - counted):
        problems.append(f"runs/{name} is in the archive but is not counted as a trajectory")
    for name in sorted(counted - archived_runs):
        problems.append(f"runs/{name} is counted as a trajectory but is not in the archive")

    included = {str(p.relative_to(ROOT)) for p in files}
    for root in sorted(archived_by_root):
        if root in EVIDENCE_ROOTS and f"{root}/README.md" not in included:
            problems.append(f"{root}/ ships {len(archived_by_root[root])} run directories with no "
                            f"README.md saying they are not this problem's trajectories")
    return problems


def leak_probes() -> list[tuple[str, str, bool]]:
    """(label, needle) pairs that must NOT appear anywhere in the public archive.

    Built from answers/reference_enumeration.json and the grader task_config, so the attack cannot go
    stale when the calibration changes: whatever the reference is today is what gets grepped for.
    """
    import yaml
    cfg = yaml.safe_load((ROOT / "tasks" / "q1_microenvironment" / "task_config.yaml")
                         .read_text(encoding="utf-8"))
    ref = cfg["reference"]
    # (label, needle, data_files_only). A designer KEY NAME is only a leak when it appears in a data
    # file (a yaml/json/md that could be carrying the value); src/scoring.py and src/main.py have to
    # name `best_objective_in_budget` in order to read it, and naming it reveals nothing. The
    # reference VALUES and DESIGNS are leaks wherever they appear, so they are probed everywhere.
    probes: list[tuple[str, str, bool]] = []
    dref = float(ref["best_objective_in_budget"])
    for nd in (4, 5, 6, 7, 9):
        probes.append((f"reference desirability at {nd} dp", f"{dref:.{nd}f}", False))
    probes.append(("reference desirability at full precision", repr(dref), False))
    probes.append(("reference change list", str(ref["best_design_changes"]), False))
    probes.append(("reference overlap", f"{float(ref['reference_overlap']):.6f}", False))
    probes.append(("reference overlap at full precision", str(ref["reference_overlap"]), False))
    probes.append(("reference achievement anchor in D",
                   f"{float(ref['start_objective']) + 0.45 * (dref - float(ref['start_objective'])):.6f}",
                   False))
    for key in ("gate_g1_one_slot_points", "gate_g1_two_slot_points", "gate_g1_three_slot_points",
                "gate_g2_safe_overlap_points", "gate_naive_answer_points",
                "gate_hand_policy_points", "n_designs_at_or_above_21_points",
                "n_designs_at_or_above_30_points", "n_designs_at_or_above_44_points",
                "score_histogram", "best_design_in_budget", "best_objective_in_budget",
                "best_design_changes"):
        # NOTE `n_qualifying` is deliberately NOT probed: question.md discloses the 13,018 count as
        # difficulty information, and the public task_config restates it under
        # release_gate.measured_effect_of_the_gate. What must not leak is the reference DESIGN, the
        # reference VALUE and the histogram, all of which are probed above.
        # the point VALUES are disclosed in question.md on purpose; a designer KEY carried by a DATA
        # file means the value came with it
        probes.append((f"designer key {key} in a data file", key, True))
    # A bare 4-digit integer is too weak a probe -- 1482 occurs by chance inside a float in
    # data/oracle_migration_check.json. The key name above is the reliable signal.
    enum = ROOT / "answers" / "reference_enumeration.json"
    if enum.is_file():
        e = json.loads(enum.read_text(encoding="utf-8"))
        probes.append(("enumeration reference change list", str(e["reference"]["changes"]), False))
        probes.append(("enumeration G2 best change list",
                       str(e["gates"]["G2_best"]["changes"]), False))
    return probes


def attack_public_archive(zip_path: Path) -> list[str]:
    """Grep the written public archive for every reference/anchor probe. Empty == clean."""
    problems: list[str] = []
    probes = leak_probes()
    with zipfile.ZipFile(zip_path) as zf:
        members = {n: zf.read(n).decode("utf-8", errors="ignore")
                   for n in zf.namelist() if not n.endswith("/")}
    DATA_SUFFIXES = (".yaml", ".yml", ".json", ".md", ".txt", ".jsonl", ".csv")
    for label, needle, data_only in probes:
        if not needle:
            continue
        for name, body in members.items():
            if data_only and not name.endswith(DATA_SUFFIXES):
                continue
            if needle in body:
                problems.append(f"{label} ({needle!r}) found in {name}")
    for name in members:
        rel = name.split("/", 1)[1] if "/" in name else name
        first = rel.split("/", 1)[0]
        if first in GRADER_ONLY_TOP and first != "runs":
            problems.append(f"grader-only path {rel} is in the public archive")
        if first == "runs" and rel != "runs/.gitkeep":
            problems.append(f"grader-only path {rel} is in the public archive")
        if first == "scripts" and Path(rel).name in GRADER_ONLY_SCRIPTS:
            problems.append(f"grader-only script {rel} is in the public archive")
        if rel in GRADER_ONLY_FILES:
            problems.append(f"grader-only file {rel} is in the public archive")
        # (a) NO MODEL WEIGHTS OF ANY KIND. Extension-based, so a renamed or re-serialised copy of
        #     the surface is caught too. The second-round audit found data/oracle.npz shipped
        #     publicly: hiding the reference value is not enough if the surface itself is shippable.
        if Path(rel).suffix.lower() in WEIGHT_SUFFIXES:
            problems.append(f"weight/model file {rel} must not be in the public archive")
    # (b0) The stub must be INERT BY INSPECTION, not merely by behaviour: every function and method
    #      body is a single `raise`, and the property tables are empty. A reviewer should be able to
    #      read the file and be done, without having to trust a call trace.
    problems.extend(_assert_stub_is_inert())
    # (b) THE PUBLIC TREE MUST NOT BE ABLE TO CONSTRUCT AN ORACLE. Checked by extracting the archive
    #     into a temp dir and importing it in a SUBPROCESS with that tree first on sys.path -- so the
    #     check is on the shipped bytes, not on the developer tree that is sitting in this process.
    problems.extend(_assert_public_tree_cannot_evaluate(zip_path))
    return problems


WEIGHT_SUFFIXES = {".npz", ".npy", ".pkl", ".pickle", ".joblib", ".pt", ".pth", ".onnx",
                   ".safetensors", ".h5", ".pb", ".ckpt", ".bin"}

# (c) Calibration numbers and shortcut-family names that question.md does NOT disclose to the model.
#     Anything here is a strategy hint (it names or values an offline shortcut) or a designer-side
#     population count. What question.md DOES disclose on purpose -- 195,892 / 16,028 / 13,018 /
#     34,706, the gate point values 17/30/44/14/44, and the >= 30 and >= 44 counts -- is deliberately
#     NOT in this list; removing it would silently change the model's input.
PUBLIC_FORBIDDEN_STRINGS = [
    ("shortcut achievement (dissolution subspace)", "0.55855"),
    ("shortcut achievement, rounded", "0.5586"),
    ("shortcut achievement (G2 raised floor)", "0.51438"),
    ("shortcut achievement, rounded", "0.5144"),
    ("named the strongest shortcut family", "dissolution subspace"),
    ("band population >= 21 points", "n_designs_at_or_above_21_points"),
    ("the full score histogram", "score_histogram"),
    ("gate anchor key", "gate_g1_"),
    ("gate anchor key", "gate_g2_"),
    ("gate anchor key", "gate_g3_"),
    ("naive-answer ceiling key", "gate_naive_answer_points"),
    ("hand-policy ceiling key", "gate_hand_policy_points"),
    ("hand-policy family size", "n_hand_policies"),
    ("gate verdict line with its values", "G1="),
    ("superadditivity ratio", "6.802"),
    ("0.x population statistic", "3,849"),
    ("0.x population statistic", "1,917"),
    ("1.x scoring population", "47,724"),
    ("1.x scoring population", "47724"),
    ("0.x zero-score count", "193,977"),
]


def _assert_stub_is_inert() -> list[str]:
    """Parse the generated public stub and assert that nothing in it can compute anything."""
    import ast
    problems: list[str] = []
    src = public_response_model()
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [f"the generated public stub does not parse: {exc}"]

    def body_is_single_raise(node) -> bool:
        body = [n for n in node.body if not isinstance(n, ast.Expr)
                or not isinstance(getattr(n, "value", None), ast.Constant)]  # drop docstrings
        return len(body) == 1 and isinstance(body[0], ast.Raise)

    funcs, methods = [], []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(node)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append((node.name, sub))
    bad = [f.name for f in funcs if not body_is_single_raise(f)]
    bad += [f"{cls}.{m.name}" for cls, m in methods
            if cls != "ResponseModelUnavailable" and not body_is_single_raise(m)]
    if bad:
        problems.append(f"public stub: these bodies are not a single raise: {bad}")
    if len(funcs) + len(methods) < 8:
        problems.append(f"public stub: only {len(funcs) + len(methods)} callables -- the stub must "
                        f"cover every name the real module exports")
    # the property tables must be empty literals, not populated
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") in ("SUB", "MCC_GRADE"):
            if not (isinstance(node.value, ast.Dict) and not node.value.keys):
                problems.append(f"public stub: {node.target.id} is not an empty dict")
    # every name the REAL module exports must exist in the stub, or the public harness breaks on
    # import in a way the subprocess test would report as a mystery
    real = (ROOT / "src" / "response_model.py").read_text(encoding="utf-8")
    real_tree = ast.parse(real)
    exported = {n.name for n in real_tree.body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))} | {
        t.id for n in real_tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))
        for t in ([n.target] if isinstance(n, ast.AnnAssign) else n.targets)
        if isinstance(t, ast.Name) and not t.id.startswith("_")}
    stub_names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))} | {
        t.id for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))
        for t in ([n.target] if isinstance(n, ast.AnnAssign) else n.targets)
        if isinstance(t, ast.Name)}
    missing = sorted(exported - stub_names - {"PACKAGE_ROOT"})
    if missing:
        problems.append(f"public stub is missing names the real module exports: {missing}")
    return problems


def _assert_public_tree_cannot_evaluate(zip_path: Path) -> list[str]:
    """Extract the public archive and prove, in a subprocess, that it cannot measure anything."""
    import subprocess
    import tempfile
    probe = (
        "import sys, json\n"
        "sys.path.insert(0, sys.argv[1] + '/src')\n"
        "out = {}\n"
        "import formulation as F\n"
        "out['formulation_imports'] = True\n"
        "out['formulation_has_mechanism'] = any(hasattr(F, n) for n in "
        "('SUB', 'MCC_GRADE', 'cqa_true', 'Oracle', 'get_oracle', 'feature_vector'))\n"
        "import response_model as RM\n"
        "try:\n"
        "    RM.Oracle(); out['oracle_constructed'] = True\n"
        "except Exception as e:\n"
        "    out['oracle_constructed'] = False; out['oracle_error'] = type(e).__name__\n"
        "for fn in ('cqa_true', 'feature_vector', 'get_oracle'):\n"
        "    try:\n"
        "        getattr(RM, fn)(dict(F.START)); out[fn] = 'RETURNED'\n"
        "    except Exception as e:\n"
        "        out[fn] = type(e).__name__\n"
        "import main, evaluator, scoring, tools, feedback, design_space, parser\n"
        "out['harness_imports'] = True\n"
        "try:\n"
        "    evaluator.evaluate_reference_start(json.loads(sys.argv[2])); out['evaluated'] = True\n"
        "except Exception as e:\n"
        "    out['evaluated'] = False; out['evaluate_error'] = type(e).__name__\n"
        "print(json.dumps(out))\n")
    import yaml
    obj = yaml.safe_load((ROOT / "tasks" / "q1_microenvironment" / "task_config.yaml")
                         .read_text(encoding="utf-8"))["objective"]
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(td)
        tree = Path(td) / ROOT.name
        # (c) forbidden strings, on the EXTRACTED bytes
        for label, needle in PUBLIC_FORBIDDEN_STRINGS:
            for f in tree.rglob("*"):
                if not f.is_file():
                    continue
                try:
                    body = f.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if needle in body:
                    problems.append(f"{label} ({needle!r}) found in "
                                    f"{f.relative_to(tree)}")
        res = subprocess.run([sys.executable, "-c", probe, str(tree), json.dumps(obj)],
                             capture_output=True, text=True, cwd=str(tree))
        if res.returncode != 0:
            problems.append(f"the public tree does not even import: {res.stderr.strip()[-300:]}")
            return problems
        got = json.loads(res.stdout.strip().splitlines()[-1])
        if got.get("formulation_has_mechanism"):
            problems.append("src/formulation.py in the public archive still exposes the mechanism "
                            "or the oracle")
        if got.get("oracle_constructed"):
            problems.append("an Oracle CAN be constructed from the public archive -- the response "
                            "surface is reachable and the task can be optimised offline")
        for fn in ("cqa_true", "feature_vector", "get_oracle"):
            if got.get(fn) != "ResponseModelUnavailable":
                problems.append(f"response_model.{fn} in the public archive did not raise "
                                f"ResponseModelUnavailable (got {got.get(fn)!r})")
        if not got.get("harness_imports"):
            problems.append("the public harness does not import")
        if got.get("evaluated"):
            problems.append("evaluator.evaluate_reference_start SUCCEEDED in the public archive -- "
                            "it can measure a formulation")
        if got.get("evaluate_error") != "ResponseModelUnavailable":
            problems.append(f"evaluate_reference_start failed for the wrong reason in the public "
                            f"archive ({got.get('evaluate_error')!r})")
    return problems


def build(side: str, out: Path, files: list[Path], trajectories: list[dict]) -> None:
    """Write one archive. `side` is 'public' or 'grader'."""
    print("\n" + "=" * 96)
    print(f"building the {side.upper()} archive -> {out}")
    selected = [p for p in files if side == "grader" or not is_grader_only(p)]
    print(f"  files: {len(selected)} of {len(files)}"
          + ("" if side == "grader" else f"  ({len(files) - len(selected)} held back as grader-only)"))

    included = {str(p.relative_to(ROOT)) for p in selected}
    if REQUIRED_TEMPLATE not in included:
        print(f"  ABORTED: {REQUIRED_TEMPLATE} must ship so a reviewer can see the expected shape "
              f"of a credentials file.")
        sys.exit(1)

    findings = scan_for_secrets(selected)
    if findings:
        print("  ABORTED: credential-shaped strings found in files that would be archived:")
        for path, sample in findings:
            print(f"    {path.relative_to(ROOT)}: {sample}")
        sys.exit(1)
    print(f"  pre-write secret scan: clean ({len(selected)} files read)")

    if side == "grader":
        complaints = verify_run_accounting(selected, trajectories)
        if complaints:
            print("  ABORTED: the archive traversal and the trajectory count disagree:")
            for c in complaints:
                print(f"    {c}")
            sys.exit(1)
        print("  run-directory accounting: archive traversal and trajectory count agree "
              f"({len(trajectories)} trajectories)")

    stem = Path(ROOT.name)
    n_written = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in selected:
            rel = path.relative_to(ROOT)
            if side == "public" and str(rel) == "tasks/q1_microenvironment/task_config.yaml":
                zf.writestr(str(stem / rel),
                            public_task_config(path.read_text(encoding="utf-8")))
            elif side == "public" and str(rel) == BOUNDARY_MODULE:
                zf.writestr(str(stem / rel), public_response_model())
            else:
                zf.write(path, stem / rel)
            n_written += 1
        if side == "public":
            zf.writestr(str(stem / PUBLIC_NOTE_NAME), PUBLIC_NOTE)
            zf.writestr(str(stem / "runs" / ".gitkeep"), "")
            n_written += 2

    archived = scan_archive(out)
    if archived:
        print("  ABORTED: credential-shaped strings found INSIDE the written archive:")
        for name, sample in archived:
            print(f"    {name}: {sample}")
        out.unlink()
        sys.exit(1)
    with zipfile.ZipFile(out) as zf:
        n_members = sum(1 for n in zf.namelist() if not n.endswith("/"))
    print(f"  post-write secret scan of the archive bytes: clean ({n_members} members read)")
    if n_members != n_written:
        print(f"  ABORTED: wrote {n_written} entries but the archive holds {n_members} members.")
        sys.exit(1)

    if side == "public":
        probes = leak_probes()
        problems = attack_public_archive(out)
        if problems:
            print(f"  ABORTED: the public archive leaks the reference answer "
                  f"({len(problems)} finding(s)):")
            for c in problems[:20]:
                print(f"    {c}")
            out.unlink()
            print(f"    {out} deleted.")
            sys.exit(1)
        print(f"  reference-leak attack on the archive bytes: clean "
              f"({len(probes)} reference probes + {len(PUBLIC_FORBIDDEN_STRINGS)} "
              f"calibration-string probes x {n_members} members, "
              f"{len(WEIGHT_SUFFIXES)} weight extensions, and a subprocess import test proving the "
              f"extracted tree cannot construct an Oracle or measure a formulation)")
        with zipfile.ZipFile(out) as zf:
            cfg_member = next(n for n in zf.namelist() if n.endswith("task_config.yaml"))
            body = zf.read(cfg_member).decode("utf-8")
        assert "available_in_this_package: false" in body, "the public task_config was not stubbed"
        print("  public task_config.yaml carries the reference stub, not the reference")

    print(f"  WRITTEN {out} ({out.stat().st_size / 1024:.1f} KB, {n_members} files)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=("public", "grader", "both"), default="both")
    ap.add_argument("--output", default=None,
                    help="only with --side public|grader; default is "
                         "<parent>/<name>_<side>.zip")
    args = ap.parse_args()
    if args.output and args.side == "both":
        ap.error("--output needs a single --side")

    everything = [p for p in ROOT.rglob("*") if p.is_file()]
    files = sorted(p for p in everything if should_include(p))

    print(f"Files in the tree that are archivable at all: {len(files)}")
    for p in sorted(p for p in everything if is_secret_config(p)):
        print(f"  excluded (may hold credentials): {p.relative_to(ROOT)}")
    n_scratch = sum(1 for p in everything
                    if is_scratch(p) or any(d in p.parts for d in EXCLUDE_DIRS))
    print(f"  excluded (caches and scratch files): {n_scratch}")

    trajectories = collect_trajectories()
    print(f"\nTrajectories (runs/, grader archive only): {len(trajectories)}")
    for tr in trajectories:
        print(f"  {tr['model'] or '?':18s} score={str(tr['total_score']):<5} "
              f"achievement={tr['achievement']}  ruler={tr['ruler_version']}  ({tr['run_dir']})")
    distinct_models = {tr["model"] for tr in trajectories if tr["model"]}
    if len(distinct_models) < 3:
        print(f"\nWARNING: only {len(distinct_models)} distinct model(s); the brief requires >= 3.")
    versions = {tr["ruler_version"] for tr in trajectories}
    if len(versions) > 1:
        print(f"\nABORTED: the shipped runs span more than one ruler version {sorted(versions)}; "
              f"exactly one scoring version's results may ship (audit finding #4).")
        sys.exit(1)

    sides = ("public", "grader") if args.side == "both" else (args.side,)
    for side in sides:
        out = Path(args.output) if args.output else ROOT.parent / f"{ROOT.name}_{side}.zip"
        build(side, out, files, trajectories)

    print("\n" + "=" * 96)
    print("done. Remember to EXTRACT each archive and re-scan it before sending "
          "(the post-write scan reads the zip's own bytes, but an extraction is the reviewer's view).")


if __name__ == "__main__":
    main()
