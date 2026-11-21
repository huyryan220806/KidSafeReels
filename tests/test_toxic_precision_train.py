import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from ml.experiment_metrics import binary_metrics
from ml.toxic_precision_train import hybrid_metrics, operating_point, weighted_clip_loss


def test_negative_weight_increases_only_negative_gradient():
    for target, ratio in ((0., 2.), (1., 1.)):
        logit = torch.tensor(.7, requires_grad=True)
        weighted_clip_loss(logit, torch.tensor(target), 1.).backward()
        original = logit.grad.clone(); logit.grad.zero_()
        weighted_clip_loss(logit, torch.tensor(target), 2.).backward()
        assert logit.grad.item() == pytest.approx(original.item()*ratio)


def test_selection_accounts_for_rules_and_cannot_hide_irreducible_rule_fp():
    labels = [0, 0, 1, 1]
    baseline = binary_metrics(labels, [.1, .2, .7, .8], .5)
    base_hybrid = hybrid_metrics(labels, [.1, .2, .7, .8], [False]*4, .5)
    point, _ = operating_point(labels, [.1, .2, .7, .8], [True, False, False, False], (), baseline, base_hybrid)
    assert not point['eligible'] and point['hybrid_fp'] >= 1


def test_selection_requires_recall_floor_in_addition_to_fewer_false_positives():
    labels = [0, 0, 1, 1]
    baseline = binary_metrics(labels, [.1, .8, .7, .8], .5)
    base_hybrid = hybrid_metrics(labels, [.1, .8, .7, .8], [False]*4, .5)
    point, _ = operating_point(labels, [.1, .65, .6, .9], [False]*4, (), baseline, base_hybrid)
    assert point['eligible'] and point['hybrid_recall'] == 1 and point['hybrid_fp'] == 1


def test_missing_asr_is_review_not_a_probability():
    values = hybrid_metrics([0, 1], [.1, .9], [False, False], .5, missing_labels=[1])
    assert values['n'] == 3 and values['fn'] == 1 and values['recall'] == .5
    assert 'log_loss' not in values
