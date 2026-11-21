import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from ml.speech_mil import clip_logit, operating_point
from ml.experiment_metrics import binary_metrics


def test_max_pool_matches_max_segment_probability_without_relabeling_segments():
    logits=torch.tensor([[2.,0.],[0.,3.],[1.,0.]],requires_grad=True)
    pooled=clip_logit(logits)
    assert torch.sigmoid(pooled).item()==pytest.approx(torch.softmax(logits,dim=-1)[:,1].max().item())
    torch.nn.functional.binary_cross_entropy_with_logits(pooled,torch.tensor(1.)).backward()
    assert logits.grad[0].abs().sum()==0 and logits.grad[2].abs().sum()==0
    assert logits.grad[1,1]<0


def test_negative_clip_penalizes_highest_scoring_segment():
    logits=torch.tensor([[2.,0.],[0.,3.]],requires_grad=True)
    torch.nn.functional.binary_cross_entropy_with_logits(clip_logit(logits),torch.tensor(0.)).backward()
    assert logits.grad[1,1]>0 and logits.grad[0].abs().sum()==0


def test_missing_segments_do_not_become_zero_probability():
    with pytest.raises(ValueError):clip_logit(torch.empty((0,2)))


def test_operating_point_uses_precision_and_recall_constraints():
    y=[0,0,1,1]
    baseline=binary_metrics(y,[.1,.4,.6,.9],.5)
    point=operating_point(y,[.2,.4,.7,.8],baseline)
    assert point['eligible'] and point['fp']==point['fn']==0
