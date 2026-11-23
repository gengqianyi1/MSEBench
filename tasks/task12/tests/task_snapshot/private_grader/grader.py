from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

FIELDS = ["cc_c_rate", "cc_until_voltage_V", "cv_voltage_V", "cv_until_current_C", "negative_electrode_thickness_um", "positive_electrode_thickness_um", "electrode_porosity", "particle_radius_um"]

CONSTRAINTS = {
    "protocol": {
        "cc_c_rate": {"min": 1.0, "max": 4.0}, "cc_until_voltage_V": {"min": 3.95, "max": 4.20},
        "cv_voltage_V": {"min": 3.95, "max": 4.20}, "cv_until_current_C": {"min": 0.05, "max": 0.30},
        "max_cc_cv_voltage_mismatch_V": 0.03,
    },
    "electrode": {
        "negative_electrode_thickness_um": {"min": 50.0, "max": 110.0}, "positive_electrode_thickness_um": {"min": 50.0, "max": 110.0},
        "electrode_porosity": {"min": 0.25, "max": 0.45}, "particle_radius_um": {"min": 3.0, "max": 10.0},
    },
    "performance": {
        "charge_time_min": {"max": 75.0}, "discharge_capacity_Ah": {"min": 0.32},
        "max_voltage_V": {"max": 4.20}, "min_voltage_V": {"min": 2.45},
        "min_negative_particle_surface_concentration_mol_m3": {"min": 1400.0},
    },
}
SCORING = {
    "best_design_breakdown": {
        "valid_submission_and_protocol": 0,
        "electrode_manufacturability": 0,
        "simulation_completed": 0,
        "fast_charge": 30,
        "discharge_capacity": 25,
        "voltage_safety": 20,
        "plating_proxy": 25,
    }
}
TOOL = {"model": "SPMe", "parameter_set": "Chen2020", "initial_soc": 0.10, "discharge_rate_c": 1.0, "discharge_cutoff_voltage_v": 2.50, "rest_time_min": 5, "solver": "CasadiSolver", "solver_mode": "safe", "solver_rtol": 1e-7, "solver_atol": 1e-7, "required_experiment_steps": 4}

def _core():
    path=Path(__file__).parent/'core'/'evaluator.py'
    name='pybamm_private_core_v2'
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module); return module

def _extract_candidate(payload):
    if not isinstance(payload,dict): raise ValueError
    candidate=payload.get('candidate',payload)
    if not isinstance(candidate,dict) or any(field not in candidate for field in FIELDS): raise ValueError
    if any(isinstance(candidate[field],bool) or not math.isfinite(float(candidate[field])) for field in FIELDS): raise ValueError
    return {**candidate,**{field:float(candidate[field]) for field in FIELDS}}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _continuous_scores(evaluation, flags):
    hard_gate = all(flags.get(key, False) for key in (
        "valid_submission_and_protocol", "electrode_manufacturability", "simulation_completed"
    ))
    if not hard_gate:
        return {key: 0.0 for key in SCORING["best_design_breakdown"]}
    charge = evaluation.charge_time_min
    capacity = evaluation.discharge_capacity_Ah
    vmax = evaluation.max_voltage_V
    vmin = evaluation.min_voltage_V
    plating = evaluation.min_negative_particle_surface_concentration_mol_m3
    fast = 30.0 * _clamp((90.0 - charge) / 15.0) if charge is not None else 0.0
    cap = 25.0 * _clamp((capacity - 0.30) / 0.02) if capacity is not None else 0.0
    if vmax is None or vmin is None:
        safe = 0.0
    else:
        safe = 20.0 * min(_clamp((4.30 - vmax) / 0.10), _clamp((vmin - 2.35) / 0.10))
    plate = 25.0 * _clamp((plating - 1300.0) / 100.0) if plating is not None else 0.0
    return {
        "valid_submission_and_protocol": 0.0,
        "electrode_manufacturability": 0.0,
        "simulation_completed": 0.0,
        "fast_charge": round(fast, 6),
        "discharge_capacity": round(cap, 6),
        "voltage_safety": round(safe, 6),
        "plating_proxy": round(plate, 6),
    }

def grade(context) -> dict:
    try:
        # 2026-09-27 fix: prefer canonical final_answer.json over candidate.json.
        # Old behavior (kept for audit): candidate.json was read first, so a
        # stale/draft candidate.json silently overrode the canonical
        # final_answer.json whenever the two files disagreed, contradicting
        # instruction.md ("the Harness saves canonical final_answer.json";
        # candidate.json is an optional audit artifact). New behavior:
        # final_answer.json is authoritative; candidate.json is consulted only
        # as a fallback when final_answer.json is absent.
        path=context.submission_dir/'final_answer.json'
        if not path.is_file(): path=context.submission_dir/'candidate.json'
        payload=json.loads(path.read_text(encoding='utf-8'))
        candidate=_extract_candidate(payload)
    except Exception:
        return {'score':0.0,'subscores':{},'failure_codes':['INVALID_CANDIDATE'],'evidence':{}}
    core=_core(); evaluation=core.evaluate_candidate(candidate,CONSTRAINTS,SCORING,TOOL)
    flags=evaluation.constraint_flags
    subscores = _continuous_scores(evaluation, flags)
    failures=[key for key,value in flags.items() if not value]
    if evaluation.simulation_error and 'simulation_completed' in failures: failures.append('PYBAMM_SIMULATION_ERROR')
    return {'score':sum(subscores.values()),'subscores':subscores,'failure_codes':failures,'evidence':{'candidate':candidate,'metrics':{'charge_time_min':evaluation.charge_time_min,'discharge_capacity_Ah':evaluation.discharge_capacity_Ah,'max_voltage_V':evaluation.max_voltage_V,'min_voltage_V':evaluation.min_voltage_V,'min_negative_particle_surface_concentration_mol_m3':evaluation.min_negative_particle_surface_concentration_mol_m3},'constraint_flags':flags,'simulation_error':evaluation.simulation_error,'runtime_contract':TOOL,'proxy_interpretation':'task_internal_concentration_floor_not_validated_plating_criterion'}}
