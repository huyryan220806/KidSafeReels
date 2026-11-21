"""Prevent label invention, train/test leakage and written-recall regressions."""
from copy import deepcopy

import numpy as np
import pytest

from ml.toxic_reviewed import attach_annotations
from ml.toxic_reviewed_train import eligible, rehearsal_indices, rank


def fixture():
    source = dict(speech=[dict(video_id='a', training_group='a', sha256='one', split='train', training_label=0),
                         dict(video_id='b', training_group='b', sha256='two', split='test', training_label=1)])
    annotations = dict(rows=[dict(video_id='a', toxic_language=1), dict(video_id='b', toxic_language=0)])
    return source, annotations


def test_import_retains_split_and_original_labels_without_mutation():
    source, annotations = fixture(); original = deepcopy(source)
    result = attach_annotations(source, annotations)
    assert source == original
    assert [(r['split'],r['training_label'],r['previous_weak_label']) for r in result['speech']] == [('train',1,0),('test',0,1)]
    result['speech'][0]['annotation']['toxic_language'] = 0
    assert annotations['rows'][0]['toxic_language'] == 1


@pytest.mark.parametrize('mode', ['missing','duplicate','uncertain','bool','group','media'])
def test_import_rejects_unsafe_label_or_split_conditions(mode):
    source, annotations = fixture()
    if mode == 'missing': annotations['rows'].pop()
    if mode == 'duplicate': annotations['rows'][1] = deepcopy(annotations['rows'][0])
    if mode == 'uncertain': annotations['rows'][0]['toxic_language'] = 'uncertain'
    if mode == 'bool': annotations['rows'][0]['toxic_language'] = True
    if mode == 'group': source['speech'][1]['training_group'] = 'a'
    if mode == 'media': source['speech'][1]['sha256'] = 'one'
    with pytest.raises(ValueError): attach_annotations(source, annotations)


def test_rehearsal_is_balanced_reproducible_and_from_supplied_train_pool():
    rows = [dict(labels=0) for _ in range(20)] + [dict(labels=1) for _ in range(3)]
    chosen = rehearsal_indices(rows, 8, 42)
    assert len(chosen) == 8 and sum(rows[i]['labels'] for i in chosen) == 4
    assert np.array_equal(chosen, rehearsal_indices(rows, 8, 42))
    assert all(0 <= i < len(rows) for i in chosen)
    with pytest.raises(ValueError): rehearsal_indices(rows, 7, 42)


def test_validation_gate_prevents_hiding_recall_drop_behind_accuracy():
    baseline = dict(model=dict(f1=.5), hybrid=dict(fp=2,fn=2), written=dict(recall=.57,f1=.58))
    assert eligible(dict(f1=.6),dict(fp=1,fn=2),dict(recall=.6,f1=.6),baseline)
    assert not eligible(dict(f1=.6),dict(fp=0,fn=2),dict(recall=.38,f1=.5),baseline)
    assert not eligible(dict(f1=.6),dict(fp=0,fn=3),dict(recall=.6,f1=.6),baseline)
    assert not eligible(dict(f1=.4),dict(fp=0,fn=2),dict(recall=.6,f1=.6),baseline)


def test_eligible_candidate_wins_even_against_better_ungated_speech_score():
    good=dict(eligible=True,speech_hybrid_f1=.8,speech_hybrid_fp=1,written_f1=.6,speech_log_loss=.5,epoch=1,threshold=.7)
    bad=dict(good,eligible=False,speech_hybrid_f1=.95,speech_hybrid_fp=0)
    assert rank(good)>rank(bad)
