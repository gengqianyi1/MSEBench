"""
Route evaluator for the retrosynthesis-route benchmark.

Given a target molecule, a purchasable inventory, and a proposed route (an
ordered list of forward reaction steps), this module deterministically checks
every constraint. There is no LLM judgement here.

A route is a list of steps. Each step is:
    {
      "reaction": "<name in REACTION_LIBRARY>",
      "reactants": ["<smiles>", ...],
      "product": "<smiles>"
    }

The route is read in synthesis (forward) order: step 1 starts from purchasable
building blocks; later steps may consume purchasable blocks and/or the products
of earlier steps; the final step's product must equal the target.

All numeric limits live in the task constraints (see task_config.yaml). This
module only reads thresholds; it does not hard-code them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set

from chemistry import (
    REACTION_LIBRARY,
    atom_economy,
    build_inventory_set,
    canonical,
    check_atom_conservation,
    normalize_reaction_name,
    step_produces,
)


@dataclass
class StepEvaluation:
    index: int
    reaction: Optional[str]
    reactants: List[str]
    declared_product: Optional[str]
    reaction_in_library: bool = False
    reactants_available: bool = False           # every reactant is purchasable or an earlier-step product
    fired: bool = False
    product_matches: bool = False               # engine reproduced declared product
    atom_economy: Optional[float] = None
    atom_economy_ok: bool = False
    atom_conservation_ok: bool = False           # no heavy atom created from nothing
    phantom_atoms: Dict[str, int] = field(default_factory=dict)
    unavailable_reactants: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RouteEvaluation:
    parsed: bool
    n_steps: int
    step_evals: List[StepEvaluation]
    final_product: Optional[str] = None
    target_canonical: Optional[str] = None
    # constraint flags (all "must be True" to be a fully valid route)
    all_reactions_in_library: bool = False
    all_steps_fire_and_match: bool = False
    all_reactants_available: bool = False
    all_atom_economy_ok: bool = False
    all_atom_conservation_ok: bool = False
    final_equals_target: bool = False
    n_steps_within_limit: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["step_evals"] = [s.to_dict() for s in self.step_evals]
        return d


def _num(constraints: Dict[str, Any], path: List[str], default: Any) -> Any:
    node: Any = constraints
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def evaluate_route(
    route: Any,
    target_smiles: str,
    inventory_smiles: List[str],
    constraints: Dict[str, Any],
) -> RouteEvaluation:
    """Deterministically evaluate a proposed synthesis route."""
    target_c = canonical(target_smiles)
    inventory = build_inventory_set(inventory_smiles)

    max_steps = int(_num(constraints, ["max_steps", "max"], 6))
    min_steps = int(_num(constraints, ["max_steps", "min"], 1))
    ae_min = float(_num(constraints, ["atom_economy", "min"], 0.0))

    if not isinstance(route, list) or len(route) == 0:
        return RouteEvaluation(
            parsed=False,
            n_steps=0,
            step_evals=[],
            target_canonical=target_c,
            error="route must be a non-empty list of steps",
        )

    # products available so far: purchasable inventory + everything produced upstream
    available: Set[str] = set(inventory)
    step_evals: List[StepEvaluation] = []
    last_product_c: Optional[str] = None

    for i, step in enumerate(route, start=1):
        reaction = step.get("reaction") if isinstance(step, dict) else None
        reactants = step.get("reactants") if isinstance(step, dict) else None
        product = step.get("product") if isinstance(step, dict) else None
        reactants = reactants if isinstance(reactants, list) else []

        se = StepEvaluation(
            index=i,
            reaction=reaction,
            reactants=reactants,
            declared_product=product,
        )

        se.reaction_in_library = normalize_reaction_name(reaction) in REACTION_LIBRARY

        # availability: every reactant must be purchasable or produced by an earlier step
        unavailable = []
        for r in reactants:
            rc = canonical(r)
            if rc is None or rc not in available:
                unavailable.append(r)
        se.unavailable_reactants = unavailable
        se.reactants_available = len(unavailable) == 0 and len(reactants) > 0

        if se.reaction_in_library:
            res = step_produces(reaction, reactants, product) if product else {
                "fired": False, "matched": False, "error": "missing_product"
            }
            se.fired = bool(res.get("fired"))
            se.product_matches = bool(res.get("matched"))
            se.error = res.get("error")
        else:
            se.error = f"reaction_not_in_library:{reaction}"

        # atom economy for this step
        if product is not None and reactants:
            ae = atom_economy(reactants, product)
            se.atom_economy = ae
            se.atom_economy_ok = ae is not None and ae >= ae_min
        else:
            se.atom_economy_ok = False

        # heavy-atom conservation: product may only be built from reactant atoms.
        # This is a hard scientific floor (law of conservation of matter) that is
        # checked independently of whether the SMARTS reaction fired.
        if product is not None and reactants:
            cons = check_atom_conservation(reactants, product)
            se.atom_conservation_ok = bool(cons.get("conserved"))
            se.phantom_atoms = cons.get("phantom_atoms", {})
        else:
            se.atom_conservation_ok = False
            se.phantom_atoms = {}

        # a matched product becomes available for downstream steps
        pc = canonical(product) if product else None
        if se.product_matches and pc is not None:
            available.add(pc)
            last_product_c = pc

        step_evals.append(se)

    n_steps = len(route)
    all_in_lib = all(s.reaction_in_library for s in step_evals)
    all_match = all(s.fired and s.product_matches for s in step_evals)
    all_avail = all(s.reactants_available for s in step_evals)
    all_ae = all(s.atom_economy_ok for s in step_evals)
    all_cons = all(s.atom_conservation_ok for s in step_evals)
    final_eq = last_product_c is not None and target_c is not None and last_product_c == target_c
    steps_ok = min_steps <= n_steps <= max_steps

    return RouteEvaluation(
        parsed=True,
        n_steps=n_steps,
        step_evals=step_evals,
        final_product=last_product_c,
        target_canonical=target_c,
        all_reactions_in_library=all_in_lib,
        all_steps_fire_and_match=all_match,
        all_reactants_available=all_avail,
        all_atom_economy_ok=all_ae,
        all_atom_conservation_ok=all_cons,
        final_equals_target=final_eq,
        n_steps_within_limit=steps_ok,
    )


def route_constraint_flags(route_eval: RouteEvaluation) -> Dict[str, bool]:
    """Process-validity flags. Exact target match is scored separately."""
    return {
        "parsed": route_eval.parsed,
        "all_reactions_in_library": route_eval.all_reactions_in_library,
        "all_reactants_available": route_eval.all_reactants_available,
        "all_steps_fire_and_match": route_eval.all_steps_fire_and_match,
        "all_atom_conservation_ok": route_eval.all_atom_conservation_ok,
        "all_atom_economy_ok": route_eval.all_atom_economy_ok,
        "n_steps_within_limit": route_eval.n_steps_within_limit,
    }


def route_is_fully_valid(route_eval: RouteEvaluation) -> bool:
    return all(route_constraint_flags(route_eval).values()) and route_eval.final_equals_target
