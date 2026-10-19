import math

import numpy as np
import pytest

from ml.asr_experiments import errors, normalized
from ml.experiment_metrics import binary_metrics, choose_operating_point


def test_metrics_match_hand_counted_confusion_and_cross_entropy():
    result = binary_metrics([0, 0, 1, 1], [.1, .8, .3, .9], .5)
    assert [result[k] for k in ("tn", "fp", "fn", "tp")] == [1, 1, 1, 1]
    assert result["accuracy"] == result["precision"] == result["recall"] == result["f1"] == .5
    assert result["log_loss"] == pytest.approx(-sum(map(math.log, [.9, .2, .3, .9])) / 4)


def test_missing_class_metrics_stay_unknown_and_scores_are_validated():
    result = binary_metrics([0, 0], [.1, .2], .5)
    assert result["roc_auc"] is result["average_precision"] is result["recall"] is None
    assert "roc_auc" in result["undefined_metrics"]
    for bad in ([.5, float("nan")], [-.1, .3], [.1, 1.1]):
        with pytest.raises(ValueError):
            binary_metrics([0, 1], bad, .5)


def test_toxic_selection_cannot_trade_away_recall_for_fewer_false_alarms():
    base = dict(recall=.5, precision=.5, macro_f1=.7, fp=10, threshold=.5)
    candidates = [base, dict(base, recall=.4, precision=.9, macro_f1=.8, fp=1),
                  dict(base, recall=.5, precision=.6, macro_f1=.73, fp=8)]
    assert choose_operating_point("toxic", candidates, base) == candidates[2]


def test_violence_selection_rejects_high_recall_with_excess_false_alarms():
    base = dict(recall=.9, precision=.9, macro_f1=.9, fp=10, threshold=.5)
    candidates = [base, dict(base, recall=1., macro_f1=.95, fp=11),
                  dict(base, recall=.95, macro_f1=.9, fp=10)]
    assert choose_operating_point("violence", candidates, base) == candidates[2]


def test_asr_edit_metrics_count_missing_and_inserted_words():
    assert normalized("TÔI, ăn cơm!") == "tôi ăn cơm"
    assert errors("tôi ăn cơm", "tôi uống cơm")["word_errors"] == 1
    assert errors("tôi ăn cơm", "")["word_errors"] == 3
    assert errors("tôi", "tôi ăn cơm") == dict(word_errors=2, words=1, char_errors=5, chars=3)
    assert errors("café", "cafe\u0301")["char_errors"] == 0


def test_temporal_aggregation_sees_variation_but_static_frames_have_zero_delta():
    from ml.visual import aggregate_features
    static = np.array([[1., 2.], [1., 2.]])
    assert aggregate_features(static, "mean-max-std-diff").tolist() == [1., 2., 1., 2., 0., 0., 0., 0.]
    moving = aggregate_features(np.array([[1., 2.], [3., 4.]]), "mean-max-std-diff")
    assert moving[-2:].tolist() == [2., 2.]
