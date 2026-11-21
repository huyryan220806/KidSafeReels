import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from ml.package_deep import passes_metrics


def test_accuracy_alone_cannot_promote_model_that_misses_more_toxic():
    baseline = dict(macro_f1=.7, f1=.5, recall=.6, precision=.4)
    candidate = dict(macro_f1=.8, f1=.6, recall=.5, precision=.8)
    assert not passes_metrics("toxic", baseline, candidate)
    assert not passes_metrics("toxic", baseline, dict(candidate, recall=.7), eligible=False)
    assert passes_metrics("toxic", baseline, dict(candidate, recall=.7))


def test_violence_requires_external_check_even_when_rlvs_improves():
    baseline = dict(macro_f1=.8, f1=.8, recall=.8)
    candidate = dict(macro_f1=.9, f1=.9, recall=.9)
    assert not passes_metrics("violence", baseline, candidate)
    external = dict(baseline=baseline, selected=dict(candidate, recall=.7))
    assert not passes_metrics("violence", baseline, candidate, external)
    assert passes_metrics("violence", baseline, candidate, dict(baseline=baseline, selected=candidate))
