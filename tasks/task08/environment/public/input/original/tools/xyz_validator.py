"""Strict multi-frame XYZ validation used by restart and result tools."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Sequence

from .file_tools import ToolInputError, WorkspaceSandbox, sha256_bytes, sha256_file


_ELEMENTS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La "
    "Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po "
    "At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg "
    "Cn Nh Fl Mc Lv Ts Og".split()
)
MAX_ATOMS = 2_000
MIN_INTERATOMIC_DISTANCE_ANGSTROM = 0.50
MAX_ABSOLUTE_COORDINATE_ANGSTROM = 100_000.0
# Pyykko-style single-bond radii (angstrom), rounded deliberately: this check
# is for *obvious* fragmentation, not bond-order assignment.  Unknown heavy
# elements use a conservative 1.50 A fallback to avoid false fragmentation.
_COVALENT_RADII = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84,
    "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07,
    "S": 1.05, "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76,
    "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39,
    "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20,
    "I": 1.39, "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Pb": 1.46,
}
FRAGMENT_BOND_SCALE = 1.35


@dataclass(frozen=True)
class XYZAtom:
    symbol: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class XYZFrame:
    index: int
    start_line: int
    end_line: int
    atom_count: int
    comment: str
    atoms: tuple[XYZAtom, ...]
    text: str

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(atom.symbol for atom in self.atoms)

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.text.encode("utf-8"))


@dataclass(frozen=True)
class XYZDocument:
    frames: tuple[XYZFrame, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    truncated: bool
    line_count: int


def _canonical_symbol(value: str) -> str:
    if not value or not value.isalpha() or len(value) > 3:
        return value
    return value[0].upper() + value[1:].lower()


def parse_xyz_text(text: str) -> XYZDocument:
    if not isinstance(text, str):
        raise TypeError("XYZ text must be a string")
    lines = text.splitlines()
    frames: list[XYZFrame] = []
    errors: list[str] = []
    warnings: list[str] = []
    truncated = False
    cursor = 0
    while cursor < len(lines):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            break
        start = cursor
        header = lines[cursor].strip()
        if not re.fullmatch(r"[0-9]+", header):
            errors.append(f"line {cursor + 1}: XYZ frame atom count is not an integer")
            break
        try:
            atom_count = int(header, 10)
        except ValueError:
            errors.append(f"line {cursor + 1}: XYZ frame atom count is not an integer")
            break
        if atom_count < 1 or atom_count > MAX_ATOMS:
            errors.append(f"line {cursor + 1}: XYZ atom count is outside 1..{MAX_ATOMS}")
            break
        if cursor + 1 >= len(lines):
            errors.append(f"frame {len(frames)}: missing comment line")
            truncated = True
            break
        comment = lines[cursor + 1]
        atom_start = cursor + 2
        atom_end = atom_start + atom_count
        if atom_end > len(lines):
            errors.append(
                f"frame {len(frames)}: truncated atom block; expected {atom_count}, "
                f"found {max(0, len(lines) - atom_start)}"
            )
            truncated = True
            break
        atoms: list[XYZAtom] = []
        frame_failed = False
        for offset, line in enumerate(lines[atom_start:atom_end], atom_start):
            fields = line.split()
            if len(fields) < 4:
                errors.append(f"line {offset + 1}: atom row has fewer than four fields")
                frame_failed = True
                continue
            symbol = _canonical_symbol(fields[0])
            if symbol not in _ELEMENTS:
                errors.append(f"line {offset + 1}: unsupported element symbol")
                frame_failed = True
                continue
            try:
                coordinates = tuple(float(value) for value in fields[1:4])
            except ValueError:
                errors.append(f"line {offset + 1}: coordinate is not numeric")
                frame_failed = True
                continue
            if not all(math.isfinite(value) for value in coordinates):
                errors.append(f"line {offset + 1}: coordinate is not finite")
                frame_failed = True
                continue
            if any(abs(value) > MAX_ABSOLUTE_COORDINATE_ANGSTROM for value in coordinates):
                errors.append(f"line {offset + 1}: coordinate magnitude is implausible")
                frame_failed = True
                continue
            if len(fields) > 4:
                warnings.append(f"line {offset + 1}: extra XYZ columns were ignored")
            atoms.append(XYZAtom(symbol, *coordinates))
        frame_lines = lines[start:atom_end]
        frame_text = "\n".join(frame_lines) + "\n"
        if not frame_failed and len(atoms) == atom_count:
            frames.append(
                XYZFrame(
                    index=len(frames),
                    start_line=start + 1,
                    end_line=atom_end,
                    atom_count=atom_count,
                    comment=comment,
                    atoms=tuple(atoms),
                    text=frame_text,
                )
            )
        else:
            truncated = True
            break
        cursor = atom_end
    return XYZDocument(tuple(frames), tuple(errors), tuple(warnings), truncated, len(lines))


def _geometry_errors(frame: XYZFrame) -> list[str]:
    errors: list[str] = []
    for left_index, left in enumerate(frame.atoms):
        for right_index in range(left_index + 1, len(frame.atoms)):
            right = frame.atoms[right_index]
            distance = math.sqrt(
                (left.x - right.x) ** 2
                + (left.y - right.y) ** 2
                + (left.z - right.z) ** 2
            )
            if distance < MIN_INTERATOMIC_DISTANCE_ANGSTROM:
                errors.append(
                    f"frame {frame.index}: atoms {left_index + 1} and {right_index + 1} "
                    f"are implausibly close ({distance:.6f} angstrom)"
                )
    return errors


def _fragmentation(frame: XYZFrame) -> dict[str, Any]:
    """Return a deterministic distance-graph fragmentation observation."""

    count = len(frame.atoms)
    adjacency: list[set[int]] = [set() for _ in range(count)]
    for left_index, left in enumerate(frame.atoms):
        left_radius = _COVALENT_RADII.get(left.symbol, 1.50)
        for right_index in range(left_index + 1, count):
            right = frame.atoms[right_index]
            right_radius = _COVALENT_RADII.get(right.symbol, 1.50)
            distance = math.sqrt(
                (left.x - right.x) ** 2
                + (left.y - right.y) ** 2
                + (left.z - right.z) ** 2
            )
            threshold = FRAGMENT_BOND_SCALE * (left_radius + right_radius)
            if MIN_INTERATOMIC_DISTANCE_ANGSTROM <= distance <= threshold:
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)
    components: list[list[int]] = []
    unseen = set(range(count))
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        component = [seed]
        pending = [seed]
        while pending:
            current = pending.pop()
            for neighbor in sorted(adjacency[current]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.append(neighbor)
                    pending.append(neighbor)
        components.append(sorted(component))
    return {
        "fragmentation_detected": len(components) > 1,
        "connected_component_count": len(components),
        "component_sizes": [len(component) for component in components],
        "components": [[index + 1 for index in component] for component in components],
        "method": "covalent-radius distance graph",
        "bond_scale": FRAGMENT_BOND_SCALE,
    }


def validate_xyz_text(
    text: str,
    *,
    expected_symbols: Sequence[str] | None = None,
    frame_index: int | None = None,
) -> dict[str, Any]:
    document = parse_xyz_text(text)
    syntax_errors = list(document.errors)
    errors = list(document.errors)
    warnings = list(document.warnings)
    expected = tuple(_canonical_symbol(value) for value in expected_symbols) if expected_symbols else None
    first_symbols = document.frames[0].symbols if document.frames else None
    frame_results: list[dict[str, Any]] = []
    for frame in document.frames:
        frame_errors = _geometry_errors(frame)
        fragmentation = _fragmentation(frame)
        if fragmentation["fragmentation_detected"]:
            frame_errors.append(
                f"frame {frame.index}: obvious fragmentation detected "
                f"({fragmentation['connected_component_count']} disconnected components)"
            )
        if expected is not None and frame.symbols != expected:
            frame_errors.append(f"frame {frame.index}: atom identity/order differs from expected input")
        if first_symbols is not None and frame.symbols != first_symbols:
            frame_errors.append(f"frame {frame.index}: atom identity/order differs across XYZ frames")
        errors.extend(frame_errors)
        frame_results.append(
            {
                "frame_index": frame.index,
                "line_start": frame.start_line,
                "line_end": frame.end_line,
                "atom_count": frame.atom_count,
                "symbols": list(frame.symbols),
                "frame_sha256": frame.sha256,
                "geometry_plausible": not frame_errors,
                "fragmentation": fragmentation,
                "fragmentation_detected": fragmentation["fragmentation_detected"],
                "connected_component_count": fragmentation["connected_component_count"],
                "errors": frame_errors,
            }
        )
    selected: XYZFrame | None = None
    if frame_index is not None:
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            errors.append("frame_index must be a non-negative integer or null")
        elif frame_index >= len(document.frames):
            errors.append("frame_index does not identify a complete frame")
        else:
            selected = document.frames[frame_index]
    elif document.frames:
        selected = document.frames[-1]
    valid = bool(document.frames and not errors and not document.truncated)
    selected_result = (
        frame_results[selected.index]
        if selected is not None and 0 <= selected.index < len(frame_results)
        else None
    )
    selected_scientifically_plausible = bool(
        selected_result is not None and not selected_result["errors"]
    )
    syntax_valid = bool(document.frames and not syntax_errors and not document.truncated)
    scientifically_plausible = bool(
        document.frames and all(not result["errors"] for result in frame_results)
    )
    tail_truncation_only = bool(
        document.truncated
        and syntax_errors
        and all(
            "truncated atom block" in error or "missing comment line" in error
            for error in syntax_errors
        )
    )
    safely_recoverable = bool(
        selected is not None
        and selected_scientifically_plausible
        and (not syntax_errors or tail_truncation_only)
    )
    # A partial final frame does not invalidate an earlier byte-complete frame.
    # This field is advisory; the strict whole-document ``valid`` remains false.
    recoverability_reason = (
        "selected complete frame is syntax-valid and scientifically plausible"
        if safely_recoverable
        else "no complete scientifically plausible selected frame is available"
    )
    last_complete = document.frames[-1] if document.frames else None
    return {
        "valid": valid,
        "errors": errors,
        "syntax_errors": syntax_errors,
        "warnings": warnings,
        "line_count": document.line_count,
        "frame_count": len(document.frames),
        "complete_frame_count": len(document.frames),
        "truncated": document.truncated,
        "tail_truncated": document.truncated and bool(document.frames),
        "recoverable_tail_truncation": tail_truncation_only and bool(document.frames),
        "atom_order_consistent": not any("atom identity/order" in item for item in errors),
        "frames": frame_results,
        "selected_frame_index": selected.index if selected else None,
        "selected_frame_sha256": selected.sha256 if selected else None,
        "selected_frame_is_complete": selected is not None,
        "last_complete_frame_index": last_complete.index if last_complete else None,
        "last_complete_frame_sha256": last_complete.sha256 if last_complete else None,
        "syntax_valid": syntax_valid,
        "syntax_status": "SYNTAX_VALID" if syntax_valid else "SYNTAX_INVALID",
        "scientifically_plausible": scientifically_plausible,
        "scientifically_suspicious": bool(document.frames and not scientifically_plausible),
        "scientific_status": (
            "SCIENTIFICALLY_PLAUSIBLE"
            if scientifically_plausible
            else "SCIENTIFICALLY_SUSPICIOUS" if document.frames else "UNKNOWN"
        ),
        "fragmentation_detected": bool(
            selected_result and selected_result["fragmentation_detected"]
        ),
        "fragmentation": (
            selected_result["fragmentation"] if selected_result else {
                "fragmentation_detected": False,
                "connected_component_count": 0,
                "component_sizes": [],
                "components": [],
                "method": "covalent-radius distance graph",
                "bond_scale": FRAGMENT_BOND_SCALE,
            }
        ),
        "safely_recoverable": safely_recoverable,
        "unsafe_to_repair": not safely_recoverable,
        "recoverability": "SAFELY_RECOVERABLE" if safely_recoverable else "UNSAFE_TO_REPAIR",
        "recoverability_reason": recoverability_reason,
        "safe_recovery_frame_index": selected.index if safely_recoverable and selected else None,
    }


def extract_xyz_frame(text: str, frame_index: int | None = None) -> XYZFrame:
    document = parse_xyz_text(text)
    if not document.frames:
        raise ToolInputError("INVALID_XYZ", "XYZ does not contain a complete valid frame")
    # A syntactically truncated tail may follow one or more complete frames.
    # Those complete frames remain recoverable; other parse errors fail closed.
    tail_truncation_only = bool(
        document.truncated
        and document.errors
        and all(
            "truncated atom block" in error or "missing comment line" in error
            for error in document.errors
        )
    )
    if document.errors and not tail_truncation_only:
        raise ToolInputError("INVALID_XYZ", "XYZ contains a non-recoverable syntax error")
    index = len(document.frames) - 1 if frame_index is None else frame_index
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(document.frames):
        raise ToolInputError("INVALID_FRAME", "frame_index does not identify a complete frame")
    frame = document.frames[index]
    geometry_errors = _geometry_errors(frame)
    if geometry_errors:
        raise ToolInputError("IMPLAUSIBLE_GEOMETRY", geometry_errors[0])
    fragmentation = _fragmentation(frame)
    if fragmentation["fragmentation_detected"]:
        raise ToolInputError("FRAGMENTED_GEOMETRY", "selected XYZ frame is obviously fragmented")
    return frame


def validate_xyz(
    sandbox: WorkspaceSandbox,
    path: str,
    *,
    expected_symbols: Sequence[str] | None = None,
    frame_index: int | None = None,
) -> dict[str, Any]:
    resolved = sandbox.resolve(path)
    size = sandbox._assert_readable_text(resolved)
    raw = resolved.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ToolInputError("INVALID_UTF8", "XYZ file is not valid UTF-8") from exc
    result = validate_xyz_text(text, expected_symbols=expected_symbols, frame_index=frame_index)
    result.update(
        {
            "path": sandbox.relative(resolved),
            "file_size_bytes": size,
            "file_sha256": sha256_file(resolved),
        }
    )
    return result


__all__ = [
    "XYZAtom",
    "XYZDocument",
    "XYZFrame",
    "extract_xyz_frame",
    "parse_xyz_text",
    "validate_xyz",
    "validate_xyz_text",
]
