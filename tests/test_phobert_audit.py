"""Audit math, source/speaker boundaries, ID alignment and label granularity."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ml.audit_validation import aligned, drift, metrics, record
from ml.experiment_metrics import binary_metrics
from ml.local_adaptation import speech_groups
from ml.phobert_audit import duplicate_audit, overlaps
from ml.speech_mil import clip_logit


def test_audit_metrics_distinguish_binary_macro_weighted_and_average_precision():
    y=[0,0,0,1,1];p=[.1,.2,.8,.3,.9];m=metrics(y,p)
    assert [m[k] for k in ('tn','fp','fn','tp')]==[2,1,1,1]
    assert m['accuracy']==pytest.approx(.6)
    assert m['toxic_precision']==m['toxic_recall']==m['toxic_f1']==.5
    assert m['macro_f1']==pytest.approx((2/3+.5)/2)
    assert m['weighted_f1']==pytest.approx(.6)
    assert m['average_precision']==pytest.approx(5/6)
    actual=binary_metrics(y,p,.5)
    for old,new in [('accuracy','accuracy'),('f1','toxic_f1'),('macro_f1','macro_f1'),
                    ('log_loss','log_loss'),('average_precision','average_precision')]:
        assert actual[old]==pytest.approx(m[new])


def test_threshold_equality_is_positive_and_bad_metrics_input_fails():
    assert metrics([1],[.5])['tp']==1
    for y,p in [([],[]),([0,1],[.2]),([0,2],[.1,.2]),([0,1],[np.nan,.2]),([0],[1.1])]:
        with pytest.raises(ValueError):metrics(y,p)


def test_drift_aligns_ids_and_rejects_label_changes():
    a=[dict(video_id='a',label=0,probability=.8),dict(video_id='b',label=1,probability=.1)]
    b=[dict(video_id='b',label=1,probability=.4),dict(video_id='a',label=0,probability=.6)]
    d=drift(a,b)
    assert d['prediction_flips']==0 and d['mean_absolute_probability_drift']==pytest.approx(.25)
    assert metrics([0,1],[.6,.4])['log_loss']<metrics([0,1],[.8,.1])['log_loss']
    b[0]['label']=0
    with pytest.raises(ValueError):drift(a,b)
    with pytest.raises(ValueError):aligned(a,['a','a'])
    with pytest.raises(ValueError):aligned(a[:1],['a','b'])


def test_source_disjoint_does_not_imply_speaker_disjoint():
    rows=[dict(video_id='a',source_group_id='one',speaker_id='speaker',split='train'),
          dict(video_id='b',source_group_id='two',speaker_id='speaker',split='valid')]
    assert overlaps(rows,'source_group_id')[0]['cross_split_values']==0
    assert overlaps(rows,'speaker_id')[0]['cross_split_values']==1
    assert overlaps(rows,'unknown')[0]['missing']==2


def test_transitive_duplicate_groups_cannot_be_treated_as_independent_sources():
    rows=[dict(video_id='a',source_group_id='one',sha256='sha1'),
          dict(video_id='b',source_group_id='one',sha256='sha2'),
          dict(video_id='c',source_group_id='two',sha256='sha3'),
          dict(video_id='d',source_group_id='three',sha256='sha3')]
    grouped=speech_groups(rows,[dict(left='b',right='c')])
    assert len(set(grouped.values()))==1


def test_normalized_duplicate_audit_excludes_empty_and_reports_clip_label_disagreement():
    result,details=duplicate_audit([dict(video_id='a',split='train',label=0,text='ĐÉO  '),
        dict(video_id='b',split='valid',label=1,text='đe\u0301o'),
        dict(video_id='c',split='test',label=1,text=' ')])
    assert result['duplicate_groups']==result['cross_split_label_disagreement_groups']==1
    assert result['nonempty_records']==2 and 'text' not in details[0]['items'][0]


def test_clip_bce_uses_logit_difference_and_only_max_segment_gets_gradient():
    logits=torch.tensor([[2.,1.],[0.,3.]],requires_grad=True)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(clip_logit(logits),torch.tensor(1.))
    expected=torch.nn.functional.cross_entropy(logits[1:2],torch.tensor([1]))
    assert loss.item()==pytest.approx(expected.item())
    loss.backward()
    assert torch.count_nonzero(logits.grad[0])==0
    assert torch.count_nonzero(logits.grad[1])==2


def test_recorder_preserves_segment_label_unknown_and_forces_eval_no_grad(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.drop=torch.nn.Dropout(.9)
            self.config=SimpleNamespace(num_labels=2,id2label={0:'LABEL_0',1:'LABEL_1'})
        def forward(self,input_ids):return SimpleNamespace(logits=self.drop(input_ids.float()))
    model=Model().train()
    clips=[dict(row=dict(video_id='v',split='valid',training_label=1),
        tokens={'input_ids':torch.tensor([[3,0],[0,2]])},
        segments=[dict(start=0,end=1,text='a'),dict(start=1,end=2,text='b')])]
    result=record(model,clips,tmp_path/'snapshot',device='cpu',origin='unit fixture',checkpoint='fixture',epoch=1)
    assert result['gradient_disabled'] and not result['model_training']
    assert result['repeated_forward_max_logit_delta']==0
    assert result['metrics']['tp']==1
    values=np.load(tmp_path/'snapshot/validation.npz',allow_pickle=False)
    assert values['segment_labels'].tolist()==[-1,-1]
    assert values['inherited_video_labels'].tolist()==[1,1]
    assert values['segment_logits'].shape==(2,2)
    clips[0]['row']['split']='test'
    with pytest.raises(ValueError):record(model,clips,tmp_path/'bad',device='cpu',origin='fixture',checkpoint='fixture',epoch=1)
