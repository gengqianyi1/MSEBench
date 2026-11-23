from __future__ import annotations

from typing import Any, Dict, Optional

from evaluator import MoleculeEvaluation


def generate_feedback(
    evaluation: MoleculeEvaluation,
    start_eval: MoleculeEvaluation,
    constraints: Dict[str, Any],
    best_so_far: Optional[MoleculeEvaluation] = None,
) -> Dict[str, Any]:
    """Reduced-bandwidth evaluator feedback.

    Design decision: the evaluator reports the COMPUTED VALUES for the submitted molecule and
    nothing else. It deliberately does NOT return:
      - per-constraint pass/fail flags,
      - a restatement of the target windows,
      - whether the submission counted as a successful optimized analog.

    All windows are stated in the question file, so the model has complete information and can
    determine feasibility itself. Withholding the verdict prevents the benchmark from degenerating
    into "search against a free perfect verifier", which is what made an earlier version trivial:
    the model could repair one constraint dimension at a time by reading the flags back.

    The model must maintain its own constraint bookkeeping and decide how to trade QED against
    the narrow molecular-weight and LogP windows.
    """
    if not evaluation.valid:
        return {
            "status": "invalid_smiles",
            "detail": evaluation.error or "RDKit could not parse the submitted SMILES.",
            "computed_properties": None,
            "reference_start_qed": start_eval.qed,
        }

    computed = {
        "molecular_weight": evaluation.molecular_weight,
        "logp": evaluation.logp,
        "tpsa": evaluation.tpsa,
        "qed": evaluation.qed,
        "sa_score": evaluation.sa_score,
        "n_structural_alerts": evaluation.n_alerts,
        "structural_alerts": evaluation.alerts,
        "similarity_to_start": evaluation.similarity_to_start,
    }

    feedback: Dict[str, Any] = {
        "status": "evaluated",
        "computed_properties": computed,
        "reference_start_qed": start_eval.qed,
    }

    # The single aggregate signal kept: the best QED reached so far across the trajectory.
    # This is a scalar progress marker, not a per-constraint verdict.
    if best_so_far is not None and best_so_far.valid:
        feedback["best_qed_so_far"] = best_so_far.qed

    return feedback
