from src.scoring import COMPLIANCE_PENALTY_WEIGHTS, score_workflow


def test_compliance_penalty_is_bounded_and_additive():
    clean = {key: True for key in COMPLIANCE_PENALTY_WEIGHTS}
    assert score_workflow(clean)["compliance_penalty"] == 0.0

    failed = dict(clean)
    failed["data_audit_accuracy"] = False
    failed["scientific_interpretation"] = False
    report = score_workflow(failed)
    assert report["compliance_penalty"] == 2.0
    assert report["max_compliance_penalty"] == 6.0
