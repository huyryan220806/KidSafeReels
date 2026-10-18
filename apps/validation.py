"""App-side checks for the existing v1 contract; shared ML schema stays unchanged."""

import math

from kidsafe.schemas import AnalysisResult


def validate_result(payload, *, backend=None, duration=None):
    # Pydantic normally trusts existing instances, including fields mutated after creation.
    # Re-parse a copy so invalid mutations cannot bypass the boundary or leak warnings.
    if isinstance(payload, AnalysisResult):
        payload = payload.model_dump()
    result = AnalysisResult.model_validate(payload)
    scores = result.scores.model_dump()
    evaluated = {label for label, score in scores.items() if score is not None}
    if set(result.evaluated_labels) != evaluated or len(result.evaluated_labels) != len(evaluated):
        raise ValueError("evaluated_labels must match non-null scores")
    if backend is not None and result.mode != backend:
        raise ValueError("Result mode does not match requested backend")
    if not math.isfinite(result.processing_seconds):
        raise ValueError("Invalid processing time")
    if result.mode == "mock" and (evaluated or result.evidence or result.decision != "review"):
        raise ValueError("Mock cannot claim to evaluate content")
    if not evaluated and result.decision != "review":
        raise ValueError("Unevaluated content must require review")
    for item in result.evidence:
        if item.label not in evaluated:
            raise ValueError("Evidence for an unevaluated label")
        if not all(math.isfinite(value) for value in (item.start_seconds, item.end_seconds)):
            raise ValueError("Invalid evidence timestamp")
        if duration is not None and item.end_seconds > duration + 0.001:
            raise ValueError("Evidence is outside the video")
    return result
