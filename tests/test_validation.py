import pytest

from apps.validation import validate_result
from tests.worker_fixtures import synthetic_result


def test_zero_score_is_evaluated_not_missing():
    result = synthetic_result()
    result.scores.violence = 0
    assert validate_result(result).scores.violence == 0


@pytest.mark.parametrize("field,value", [
    ("evaluated_labels", []), ("evaluated_labels", ["violence", "violence"]),
    ("processing_seconds", float("inf")), ("mode", "mock"),
])
def test_invalid_contract(field, value):
    result = synthetic_result().model_dump()
    result[field] = value
    with pytest.raises(ValueError):
        validate_result(result)


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), 99])
def test_invalid_evidence_time(timestamp):
    result = synthetic_result().model_dump()
    result["evidence"][0]["end_seconds"] = timestamp
    with pytest.raises(ValueError):
        validate_result(result, duration=2)


def test_mode_mismatch():
    with pytest.raises(ValueError):
        validate_result(synthetic_result(), backend="mock")


def test_empty_scope_cannot_claim_no_risk():
    result = synthetic_result().model_dump()
    result.update(scores={}, evaluated_labels=[], evidence=[], decision="no_risk_detected")
    with pytest.raises(ValueError):
        validate_result(result)


def test_mutated_model_instance_is_revalidated():
    result = synthetic_result()
    result.scores.violence = 2
    with pytest.raises(ValueError):
        validate_result(result)


def test_app_warnings_do_not_mutate_pipeline_cached_result():
    result = synthetic_result()
    validated = validate_result(result)
    validated.warnings.append("app warning")
    assert "app warning" not in result.warnings
