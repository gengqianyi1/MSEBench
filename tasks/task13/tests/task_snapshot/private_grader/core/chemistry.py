"""Private-grader shim over the public, versioned reaction contract.

Keeping one executable chemistry source prevents drift between Agent feedback
and private scoring while revealing no private reference route.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PUBLIC = Path(__file__).resolve().parents[2] / "input" / "tools" / "reaction_validator.py"
_SPEC = importlib.util.spec_from_file_location("task13_public_reaction_contract_v202", _PUBLIC)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load public reaction contract: {_PUBLIC}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

REACTION_LIBRARY = _MODULE.REACTION_LIBRARY
normalize_reaction_name = _MODULE.normalize_reaction_name
canonical = _MODULE.canonical
mol_weight = _MODULE.mol_weight
run_named_reaction = _MODULE.run_named_reaction
step_produces = _MODULE.step_produces
check_atom_conservation = _MODULE.check_atom_conservation
atom_economy = _MODULE.atom_economy
is_purchasable = _MODULE.is_purchasable
build_inventory_set = _MODULE.build_inventory_set
