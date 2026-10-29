import numpy as np
import pytest

from ml.hybrid_evaluation import align_cache, decision_metrics


def test_decision_metrics_count_real_errors_without_fake_probabilities():
    m = decision_metrics([0, 0, 1, 1], [0, 1, 0, 1])
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)
    assert m["accuracy"] == m["f1"] == m["false_positive_rate"] == .5
    assert "log_loss" not in m


def test_cache_reorders_by_identity_and_rejects_label_or_threshold_changes():
    values = dict(ids=["a", "b"], groups=["a", "b"], y=np.array([0, 1]))
    rows = [dict(sample_id="b", group_id="b", split="test", label="1", score=".9", prediction="1"),
            dict(sample_id="a", group_id="a", split="test", label="0", score=".7", prediction="0")]
    assert align_cache(rows, "test", values, .78).tolist() == [.7, .9]
    with pytest.raises(ValueError, match="threshold mismatch"):
        align_cache(rows, "test", values, .5)
    rows[0]["label"] = "0"
    with pytest.raises(ValueError, match="label"):
        align_cache(rows, "test", values, .78)


def test_cache_rejects_duplicate_ids():
    row = dict(sample_id="a", group_id="a", split="test", label="0", score=".1", prediction="0")
    with pytest.raises(ValueError, match="duplicated"):
        align_cache([row, row], "test", dict(ids=["a"], groups=["a"], y=np.array([0])), .78)
