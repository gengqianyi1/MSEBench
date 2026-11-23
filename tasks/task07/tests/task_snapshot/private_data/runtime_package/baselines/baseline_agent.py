"""Deterministic offline RDKit-edit baseline for the QED benchmark."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
from typing import Any

import yaml
from rdkit import Chem, rdBase


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Required for ``python baselines/baseline_agent.py`` as well as ``-m``.
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.rdkit_tools import evaluate_candidate  # noqa: E402


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "task.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline"
OFFLINE_BASELINE_MARKER = "OFFLINE_BASELINE"


@dataclass(frozen=True)
class _Edit:
    smiles: str
    rationale: str
    priority: tuple[int, int, float, str]


@dataclass
class _MeasuredCandidate:
    name: str
    smiles: str
    rationale: str
    evaluation: dict[str, Any]


def _load_config(config: Any = None) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    path = DEFAULT_CONFIG_PATH if config is None else Path(config)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("Task configuration root must be a mapping.")
    return dict(loaded)


def _canonical_after_edit(editable: Chem.RWMol) -> str | None:
    try:
        mol = editable.GetMol()
        with rdBase.BlockLogs():
            Chem.SanitizeMol(mol)
        if mol.GetNumAtoms() == 0 or len(Chem.GetMolFrags(mol)) != 1:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def _terminal_chain_length(mol: Chem.Mol, atom_index: int) -> int:
    """Length of the unbranched aliphatic-carbon tail containing ``atom_index``."""

    length = 0
    previous: int | None = None
    current = atom_index
    while True:
        atom = mol.GetAtomWithIdx(current)
        if atom.GetAtomicNum() != 6 or atom.GetIsAromatic():
            break
        length += 1
        next_atoms = [neighbor for neighbor in atom.GetNeighbors() if neighbor.GetIdx() != previous]
        if len(next_atoms) != 1:
            break
        next_atom = next_atoms[0]
        if next_atom.GetAtomicNum() != 6 or next_atom.GetIsAromatic():
            break
        previous, current = current, next_atom.GetIdx()
    return length


def _generate_local_edits(
    smiles: str,
    allowed_atomic_numbers: list[int],
    rng: random.Random,
    *,
    include_aromatic_halogen: bool = True,
    maximum_halogen_count: int | None = None,
) -> list[_Edit]:
    """Generate one-bond-local graph edits; no target SMILES are embedded."""

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    edits: dict[str, _Edit] = {}
    halogen_atomic_numbers = {9, 17, 35, 53, 85}
    current_halogen_count = sum(
        atom.GetAtomicNum() in halogen_atomic_numbers for atom in mol.GetAtoms()
    )

    def permits_added_atom(atomic_number: int) -> bool:
        return (
            atomic_number not in halogen_atomic_numbers
            or maximum_halogen_count is None
            or current_halogen_count < maximum_halogen_count
        )

    terminal_atoms = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetDegree() == 1 and not atom.GetIsAromatic()
    ]

    # Deletions are tried first because the starting molecule is near the upper
    # MW/LogP/rotatable-bond bounds.  Long unbranched carbon tails are preferred
    # by a structural heuristic, not by a memorised product.
    for atom in terminal_atoms:
        index = atom.GetIdx()
        chain_length = _terminal_chain_length(mol, index)
        editable = Chem.RWMol(mol)
        symbol = atom.GetSymbol()
        editable.RemoveAtom(index)
        canonical = _canonical_after_edit(editable)
        if canonical and canonical != smiles:
            edits.setdefault(
                canonical,
                _Edit(
                    smiles=canonical,
                    rationale=f"Deleted a terminal {symbol} atom from a local substituent.",
                    priority=(0, -chain_length, rng.random(), canonical),
                ),
            )

    # Atom replacements explore polarity and lipophilicity changes while
    # preserving graph topology.
    for atom in terminal_atoms:
        index = atom.GetIdx()
        chain_length = _terminal_chain_length(mol, index)
        for atomic_number in allowed_atomic_numbers:
            if atomic_number == atom.GetAtomicNum() or atomic_number <= 0:
                continue
            if (
                atom.GetAtomicNum() not in halogen_atomic_numbers
                and not permits_added_atom(atomic_number)
            ):
                continue
            editable = Chem.RWMol(mol)
            replacement = editable.GetAtomWithIdx(index)
            old_symbol = replacement.GetSymbol()
            replacement.SetAtomicNum(atomic_number)
            replacement.SetFormalCharge(0)
            canonical = _canonical_after_edit(editable)
            if not canonical or canonical == smiles:
                continue
            new_symbol = Chem.GetPeriodicTable().GetElementSymbol(atomic_number)
            edits.setdefault(
                canonical,
                _Edit(
                    smiles=canonical,
                    rationale=f"Replaced terminal {old_symbol} with {new_symbol}.",
                    priority=(1, -chain_length, rng.random(), canonical),
                ),
            )

    # Single-atom extensions make the edit space bidirectional.  They are
    # lower priority but can repair candidates that fall below a property bound.
    for atom in terminal_atoms:
        if atom.GetTotalNumHs() <= 0:
            continue
        index = atom.GetIdx()
        chain_length = _terminal_chain_length(mol, index)
        for atomic_number in allowed_atomic_numbers:
            if atomic_number not in {6, 7, 8, 9, 17}:
                continue
            if not permits_added_atom(atomic_number):
                continue
            editable = Chem.RWMol(mol)
            new_index = editable.AddAtom(Chem.Atom(atomic_number))
            editable.AddBond(index, new_index, Chem.BondType.SINGLE)
            canonical = _canonical_after_edit(editable)
            if not canonical or canonical == smiles:
                continue
            symbol = Chem.GetPeriodicTable().GetElementSymbol(atomic_number)
            edits.setdefault(
                canonical,
                _Edit(
                    smiles=canonical,
                    rationale=f"Extended a terminal substituent by one {symbol} atom.",
                    priority=(2, -chain_length, rng.random(), canonical),
                ),
            )

    # Generic aromatic substitution is a common local medicinal-chemistry move
    # and keeps the baseline useful across scaffold families.  The graph rule
    # contains no case-specific product SMILES.  Public halogen caps suppress
    # F/Cl additions without suppressing the chemically distinct methyl edit.
    aromatic_substituents = [6]
    if include_aromatic_halogen:
        aromatic_substituents.extend([9, 17])
    for atom in mol.GetAtoms():
        if not atom.GetIsAromatic() or atom.GetTotalNumHs() <= 0:
            continue
        for atomic_number in aromatic_substituents:
            if (
                atomic_number not in allowed_atomic_numbers
                or not permits_added_atom(atomic_number)
            ):
                continue
            editable = Chem.RWMol(mol)
            new_index = editable.AddAtom(Chem.Atom(atomic_number))
            editable.AddBond(atom.GetIdx(), new_index, Chem.BondType.SINGLE)
            canonical = _canonical_after_edit(editable)
            if not canonical or canonical == smiles:
                continue
            substituent = "methyl" if atomic_number == 6 else Chem.GetPeriodicTable().GetElementSymbol(atomic_number)
            edits.setdefault(
                canonical,
                _Edit(
                    smiles=canonical,
                    rationale=f"Added {substituent} at an aromatic position bearing hydrogen.",
                    priority=(-1, 0, rng.random(), canonical),
                ),
            )

    return sorted(edits.values(), key=lambda edit: edit.priority)


def _beam_order(candidate: _MeasuredCandidate) -> tuple[int, float, float, str]:
    evaluation = candidate.evaluation
    feasible_rank = 0 if evaluation.get("feasible") else 1
    qed = evaluation.get("qed")
    similarity = evaluation.get("similarity_to_start")
    return (
        feasible_rank,
        -float(qed if isinstance(qed, (int, float)) else -1.0),
        -float(similarity if isinstance(similarity, (int, float)) else -1.0),
        candidate.smiles,
    )


def run_baseline(
    config: Any = None,
    *,
    beam_width: int = 3,
    max_depth: int = 3,
    process_events: list[dict[str, Any]] | None = None,
    include_aromatic_halogen: bool = True,
) -> dict[str, Any]:
    """Run a budget-aware local-edit beam search and return final-answer JSON."""

    if process_events is not None and type(process_events) is not list:
        raise ValueError("process_events must be a list when supplied.")
    if process_events:
        raise ValueError("process_events must be empty when supplied.")

    task_config = _load_config(config)
    task = task_config.get("task", {})
    constraints = task_config.get("constraints", {})
    budget = task_config.get("budget", {})
    runtime = task_config.get("runtime", {})
    if not isinstance(task, Mapping) or not isinstance(constraints, Mapping):
        raise ValueError("Task and constraints sections must be mappings.")
    if not isinstance(budget, Mapping) or not isinstance(runtime, Mapping):
        raise ValueError("Budget and runtime sections must be mappings.")

    start_smiles = task.get("start_smiles")
    if not isinstance(start_smiles, str) or not start_smiles.strip():
        raise ValueError("Task configuration must define task.start_smiles.")
    task_id = task.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("Task configuration must define task.id.")

    try:
        max_tool_calls = int(budget.get("max_tool_calls", 1))
        max_unique = int(budget.get("max_unique_evaluated_candidates", max_tool_calls))
        random_seed = int(runtime.get("random_seed", 42))
        beam_width = max(1, int(beam_width))
        max_depth = max(1, int(max_depth))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Baseline search parameters must be integers.") from exc
    evaluation_limit = min(max_tool_calls, max_unique)
    if evaluation_limit < 1:
        raise ValueError("At least one tool call and unique candidate must be allowed.")

    raw_allowed = constraints.get("allowed_atomic_numbers", [6, 7, 8, 9, 16, 17])
    try:
        allowed = sorted({int(number) for number in raw_allowed})
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("allowed_atomic_numbers must be an integer list.") from exc

    halogen_specification = constraints.get("halogen_count")
    maximum_halogen_count: int | None = None
    if isinstance(halogen_specification, Mapping) and "max" in halogen_specification:
        try:
            maximum_halogen_count = int(halogen_specification["max"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("constraints.halogen_count.max must be an integer.") from exc

    rng = random.Random(random_seed)
    measured: list[_MeasuredCandidate] = []
    seen: set[str] = set()
    evaluation_attempts = 0

    def record_event(event: str, **payload: Any) -> None:
        if process_events is None:
            return
        process_events.append(
            {
                "sequence": len(process_events) + 1,
                "event": event,
                "provenance": OFFLINE_BASELINE_MARKER,
                **payload,
            }
        )

    record_event(
        "run_start",
        execution_mode=OFFLINE_BASELINE_MARKER,
        provider=OFFLINE_BASELINE_MARKER,
        model=OFFLINE_BASELINE_MARKER,
        is_llm=False,
    )

    def measure(smiles: str, name: str, rationale: str) -> _MeasuredCandidate | None:
        nonlocal evaluation_attempts
        if evaluation_attempts >= evaluation_limit:
            return None
        evaluation_attempts += 1
        call_id = f"offline-baseline-evaluate-{evaluation_attempts:03d}"
        try:
            evaluation = evaluate_candidate(smiles, task_config)
        except Exception as exc:
            record_event(
                "tool_call",
                call_id=call_id,
                tool="evaluate_candidate",
                arguments={"smiles": smiles},
                status="FAILED",
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            raise
        record_event(
            "tool_call",
            call_id=call_id,
            tool="evaluate_candidate",
            arguments={"smiles": smiles},
            status="SUCCESS",
            result=evaluation,
        )
        canonical = evaluation.get("canonical_smiles")
        if (
            evaluation.get("ok") is not True
            or evaluation.get("valid") is not True
            or not isinstance(canonical, str)
            or canonical in seen
        ):
            return None
        seen.add(canonical)
        record = _MeasuredCandidate(name, canonical, rationale, evaluation)
        measured.append(record)
        return record

    start = measure(
        start_smiles.strip(),
        "baseline_start",
        "Measured the configured starting molecule before applying local edits.",
    )
    if start is None or not start.evaluation.get("valid"):
        raise ValueError("Configured starting molecule is not a valid benchmark molecule.")
    beam = [start]

    for depth in range(1, max_depth + 1):
        if len(measured) >= evaluation_limit or not beam:
            break
        pooled: list[tuple[int, _Edit]] = []
        for parent_rank, parent in enumerate(sorted(beam, key=_beam_order)):
            for edit in _generate_local_edits(
                parent.smiles,
                allowed,
                rng,
                include_aromatic_halogen=include_aromatic_halogen,
                maximum_halogen_count=maximum_halogen_count,
            ):
                if edit.smiles not in seen:
                    pooled.append((parent_rank, edit))
        pooled.sort(
            key=lambda item: (
                item[1].priority[0],
                item[0],
                item[1].priority[1],
                item[1].priority[2],
                item[1].priority[3],
            )
        )

        # Reserve one measurement for each remaining depth so sequential edits
        # remain possible under the eight-candidate canonical budget.
        remaining = evaluation_limit - len(measured)
        reserve = max(0, max_depth - depth)
        round_limit = max(1, min(beam_width, remaining - reserve))
        next_beam: list[_MeasuredCandidate] = []
        pooled_seen: set[str] = set()
        for _, edit in pooled:
            if len(next_beam) >= round_limit or len(measured) >= evaluation_limit:
                break
            if edit.smiles in pooled_seen or edit.smiles in seen:
                continue
            pooled_seen.add(edit.smiles)
            record = measure(
                edit.smiles,
                f"beam_edit_{len(measured):02d}",
                f"Depth {depth}: {edit.rationale}",
            )
            if record is not None:
                next_beam.append(record)
        beam = sorted(next_beam, key=_beam_order)[:beam_width]

    feasible = [record for record in measured if record.evaluation.get("feasible")]
    if not feasible:
        raise RuntimeError("Bounded baseline search found no feasible measured molecule.")
    best = sorted(feasible, key=_beam_order)[0]
    start_qed = float(start.evaluation["qed"])
    best_qed = float(best.evaluation["qed"])
    delta = best_qed - start_qed
    final_rationale = (
        "Deterministic bounded RDKit beam search selected this feasible local edit; "
        f"measured QED {best_qed:.6f} versus start {start_qed:.6f} "
        f"(delta {delta:+.6f})."
    )

    unique_evaluated = [record.smiles for record in measured]
    output_config = task_config.get("output", {})
    output_schema_version = (
        output_config.get("schema_version", "1.0")
        if isinstance(output_config, Mapping)
        else "1.0"
    )
    final_answer = {
        "schema_version": output_schema_version,
        "task_id": task_id,
        "final_candidate": {
            "name": best.name,
            "smiles": best.smiles,
            "rationale": final_rationale,
        },
        "candidate_history": [
            {
                "name": record.name,
                "smiles": record.smiles,
                "rationale": record.rationale,
            }
            for record in measured
        ],
        "run_metadata": {
            "tool_call_count": evaluation_attempts,
            "budget_exceeded": False,
            "unique_evaluated_candidates": unique_evaluated,
            "final_candidate_evaluated": best.smiles in seen,
        },
    }
    record_event(
        "run_complete",
        execution_mode=OFFLINE_BASELINE_MARKER,
        is_llm=False,
        final_candidate={"canonical_smiles": best.smiles},
    )
    return final_answer


def write_final_answer(result: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Write the runner-compatible ``final_answer.json`` atomically enough for CLI use."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "final_answer.json"
    destination.write_text(
        json.dumps(dict(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def write_trajectory(
    process_events: list[dict[str, Any]], output_dir: str | Path
) -> Path:
    """Write the actual offline RDKit calls as runner-compatible JSONL."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "trajectory.jsonl"
    destination.write_text(
        "".join(
            json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
            for event in process_events
        ),
        encoding="utf-8",
        newline="\n",
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that will receive final_answer.json.",
    )
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        process_events: list[dict[str, Any]] = []
        result = run_baseline(
            args.config,
            beam_width=args.beam_width,
            max_depth=args.max_depth,
            process_events=process_events,
        )
        destination = write_final_answer(result, args.output_dir)
        trajectory_destination = write_trajectory(
            process_events, args.output_dir
        )
    except Exception as exc:
        print(f"Baseline failed: {exc}", file=sys.stderr)
        return 2
    print(destination)
    print(trajectory_destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
