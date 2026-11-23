from __future__ import annotations

import os
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class BatteryEvaluation:
    candidate: Dict[str, Any]
    valid_submission: bool
    protocol_valid: bool = False
    electrode_valid: bool = False
    simulation_completed: bool = False
    charge_time_min: Optional[float] = None
    discharge_capacity_Ah: Optional[float] = None
    max_voltage_V: Optional[float] = None
    min_voltage_V: Optional[float] = None
    min_negative_particle_surface_concentration_mol_m3: Optional[float] = None
    score_for_candidate: float = 0.0
    constraint_flags: Dict[str, bool] = field(default_factory=dict)
    simulation_error: Optional[str] = None
    design_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _in_range(value: float, cfg: Dict[str, float]) -> bool:
    return cfg["min"] <= value <= cfg["max"]


def _round_or_none(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _candidate_key(candidate: Dict[str, Any]) -> str:
    keys = [
        "cc_c_rate",
        "cc_until_voltage_V",
        "cv_voltage_V",
        "cv_until_current_C",
        "negative_electrode_thickness_um",
        "positive_electrode_thickness_um",
        "electrode_porosity",
        "particle_radius_um",
    ]
    return "|".join(f"{key}={round(float(candidate[key]), 4)}" for key in keys)


def _check_protocol(candidate: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
    protocol = constraints["protocol"]
    checks = [
        _in_range(candidate["cc_c_rate"], protocol["cc_c_rate"]),
        _in_range(candidate["cc_until_voltage_V"], protocol["cc_until_voltage_V"]),
        _in_range(candidate["cv_voltage_V"], protocol["cv_voltage_V"]),
        _in_range(candidate["cv_until_current_C"], protocol["cv_until_current_C"]),
        abs(candidate["cc_until_voltage_V"] - candidate["cv_voltage_V"])
        <= protocol["max_cc_cv_voltage_mismatch_V"],
    ]
    return all(checks)


def _check_electrode(candidate: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
    electrode = constraints["electrode"]
    return all(
        [
            _in_range(candidate["negative_electrode_thickness_um"], electrode["negative_electrode_thickness_um"]),
            _in_range(candidate["positive_electrode_thickness_um"], electrode["positive_electrode_thickness_um"]),
            _in_range(candidate["electrode_porosity"], electrode["electrode_porosity"]),
            _in_range(candidate["particle_radius_um"], electrode["particle_radius_um"]),
        ]
    )


def _array_min(solution: Any, variable_names: list[str]) -> Optional[float]:
    for name in variable_names:
        try:
            values = solution[name].entries
            return float(values.min())
        except Exception:
            continue
    return None


def _array_max(solution: Any, variable_names: list[str]) -> Optional[float]:
    for name in variable_names:
        try:
            values = solution[name].entries
            return float(values.max())
        except Exception:
            continue
    return None


def _run_pybamm_simulation(candidate: Dict[str, Any], tool_cfg: Dict[str, Any]) -> Dict[str, Optional[float]]:
    os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
    runtime_home = Path.cwd() / ".runtime_home"
    os.environ.setdefault("LOCALAPPDATA", str(runtime_home / "LocalAppData"))
    os.environ.setdefault("APPDATA", str(runtime_home / "AppData"))
    Path(os.environ["LOCALAPPDATA"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["APPDATA"]).mkdir(parents=True, exist_ok=True)

    try:
        import pybamm
    except Exception as exc:
        raise RuntimeError(
            "PyBaMM is not installed or could not be imported. Install requirements.txt before scoring."
        ) from exc

    model_name = tool_cfg.get("model", "SPMe")
    if model_name != "SPMe":
        raise RuntimeError(f"Unsupported configured model: {model_name}")

    model = pybamm.lithium_ion.SPMe()
    parameter_values = pybamm.ParameterValues(tool_cfg.get("parameter_set", "Chen2020"))

    porosity = float(candidate["electrode_porosity"])
    radius_m = float(candidate["particle_radius_um"]) * 1e-6
    parameter_values.update(
        {
            "Negative electrode thickness [m]": float(candidate["negative_electrode_thickness_um"]) * 1e-6,
            "Positive electrode thickness [m]": float(candidate["positive_electrode_thickness_um"]) * 1e-6,
            "Negative electrode porosity": porosity,
            "Positive electrode porosity": porosity,
            "Negative particle radius [m]": radius_m,
            "Positive particle radius [m]": radius_m,
        },
        check_already_exists=True,
    )

    solver_rtol = float(tool_cfg.get("solver_rtol", 1e-7))
    solver_atol = float(tool_cfg.get("solver_atol", 1e-7))
    if solver_rtol != 1e-7 or solver_atol != 1e-7:
        raise RuntimeError(
            f"SOLVER_TOLERANCE_MISMATCH: expected rtol=atol=1e-7, got rtol={solver_rtol}, atol={solver_atol}"
        )
    solver = pybamm.CasadiSolver(
        mode=tool_cfg.get("solver_mode", "safe"),
        rtol=solver_rtol,
        atol=solver_atol,
    )
    experiment = pybamm.Experiment(
        [
            (
                f"Charge at {candidate['cc_c_rate']} C until {candidate['cc_until_voltage_V']} V",
                f"Hold at {candidate['cv_voltage_V']} V until {candidate['cv_until_current_C']} C",
                f"Rest for {tool_cfg.get('rest_time_min', 5)} minutes",
                f"Discharge at {tool_cfg.get('discharge_rate_c', 1.0)} C until {tool_cfg.get('discharge_cutoff_voltage_v', 2.5)} V",
            )
        ],
        period="60 seconds",
    )
    simulation = pybamm.Simulation(
        model,
        parameter_values=parameter_values,
        experiment=experiment,
        solver=solver,
    )
    solution = simulation.solve(initial_soc=float(tool_cfg.get("initial_soc", 0.10)))

    cycle = solution.cycles[0] if getattr(solution, "cycles", None) else solution
    steps = getattr(cycle, "steps", None) or []
    required_steps = int(tool_cfg.get("required_experiment_steps", 4))
    if len(steps) < required_steps:
        raise RuntimeError(
            f"INCOMPLETE_EXPERIMENT: expected {required_steps} completed steps, got {len(steps)}; partial solutions are not scored"
        )
    charge_steps = steps[:2]
    discharge_step = steps[3]

    charge_start = min((_array_min(step, ["Time [min]"]) for step in charge_steps), default=0.0)
    charge_end = max((_array_max(step, ["Time [min]"]) for step in charge_steps), default=None)
    charge_start = charge_start or 0.0
    charge_time_min = None if charge_end is None else charge_end - charge_start

    discharge_capacity = _array_max(discharge_step, ["Discharge capacity [A.h]"])
    if discharge_capacity is None or abs(discharge_capacity) <= 1e-12:
        throughput_min = _array_min(discharge_step, ["Throughput capacity [A.h]"])
        throughput_max = _array_max(discharge_step, ["Throughput capacity [A.h]"])
        if throughput_min is not None and throughput_max is not None:
            discharge_capacity = abs(throughput_max - throughput_min)
    if discharge_capacity is not None:
        discharge_capacity = abs(discharge_capacity)

    return {
        "charge_time_min": charge_time_min,
        "discharge_capacity_Ah": discharge_capacity,
        "max_voltage_V": _array_max(solution, ["Terminal voltage [V]"]),
        "min_voltage_V": _array_min(solution, ["Terminal voltage [V]"]),
        "min_negative_particle_surface_concentration_mol_m3": _array_min(
            solution,
            [
                "X-averaged negative particle surface concentration [mol.m-3]",
                "Negative particle surface concentration [mol.m-3]",
            ],
        ),
    }


def check_constraints(evaluation: BatteryEvaluation, constraints: Dict[str, Any]) -> Dict[str, bool]:
    performance = constraints["performance"]
    flags = {
        "valid_submission_and_protocol": bool(evaluation.valid_submission and evaluation.protocol_valid),
        "electrode_manufacturability": bool(evaluation.electrode_valid),
        "simulation_completed": bool(evaluation.simulation_completed),
        "fast_charge": (
            evaluation.charge_time_min is not None
            and evaluation.charge_time_min <= performance["charge_time_min"]["max"]
        ),
        "discharge_capacity": (
            evaluation.discharge_capacity_Ah is not None
            and evaluation.discharge_capacity_Ah >= performance["discharge_capacity_Ah"]["min"]
        ),
        "voltage_safety": (
            evaluation.max_voltage_V is not None
            and evaluation.min_voltage_V is not None
            and evaluation.max_voltage_V <= performance["max_voltage_V"]["max"]
            and evaluation.min_voltage_V >= performance["min_voltage_V"]["min"]
        ),
        "plating_proxy": (
            evaluation.min_negative_particle_surface_concentration_mol_m3 is not None
            and evaluation.min_negative_particle_surface_concentration_mol_m3
            >= performance["min_negative_particle_surface_concentration_mol_m3"]["min"]
        ),
    }
    return flags


def candidate_score(flags: Dict[str, bool], scoring_cfg: Dict[str, Any]) -> float:
    points = 0.0
    for key, max_points in scoring_cfg["best_design_breakdown"].items():
        if flags.get(key, False):
            points += float(max_points)
    return points


def evaluate_candidate(
    candidate: Dict[str, Any],
    constraints: Dict[str, Any],
    scoring_cfg: Dict[str, Any],
    tool_cfg: Dict[str, Any],
) -> BatteryEvaluation:
    valid_submission = bool(candidate)
    protocol_valid = valid_submission and _check_protocol(candidate, constraints)
    electrode_valid = valid_submission and _check_electrode(candidate, constraints)

    evaluation = BatteryEvaluation(
        candidate=candidate,
        valid_submission=valid_submission,
        protocol_valid=protocol_valid,
        electrode_valid=electrode_valid,
        design_key=_candidate_key(candidate) if valid_submission else None,
    )

    if protocol_valid and electrode_valid:
        try:
            metrics = _run_pybamm_simulation(candidate, tool_cfg)
            evaluation.simulation_completed = all(value is not None for value in metrics.values())
            evaluation.charge_time_min = _round_or_none(metrics["charge_time_min"], 4)
            evaluation.discharge_capacity_Ah = _round_or_none(metrics["discharge_capacity_Ah"], 4)
            evaluation.max_voltage_V = _round_or_none(metrics["max_voltage_V"], 4)
            evaluation.min_voltage_V = _round_or_none(metrics["min_voltage_V"], 4)
            evaluation.min_negative_particle_surface_concentration_mol_m3 = _round_or_none(
                metrics["min_negative_particle_surface_concentration_mol_m3"],
                4,
            )
        except Exception as exc:
            evaluation.simulation_error = str(exc)

    flags = check_constraints(evaluation, constraints)
    evaluation.constraint_flags = flags
    evaluation.score_for_candidate = candidate_score(flags, scoring_cfg)
    return evaluation


def satisfies_all_constraints(evaluation: BatteryEvaluation, constraints: Dict[str, Any]) -> bool:
    return all(check_constraints(evaluation, constraints).values())

