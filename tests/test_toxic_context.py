"""Context must preserve the scored segment and the frozen data boundaries."""
import pytest

from ml.toxic_context import center_window, development_rows, passes_gate


def test_context_prioritizes_center_and_nearest_neighbors():
    assert center_window([1,2,3],[4,5],[6,7,8],6)==[2,3,4,5,6,7]
    assert center_window([],[4,5],[6,7,8],5)==[4,5,6,7,8]
    assert center_window([1,2,3],[4,5],[],5)==[1,2,3,4,5]
    assert center_window([1],[2,3,4,5],[6],3)==[2,3,4]
    with pytest.raises(ValueError): center_window([],[],[],5)


def test_context_never_exceeds_budget_or_loses_untruncated_center():
    for left in range(8):
        for right in range(8):
            for budget in range(3,12):
                result=center_window([-1]*left,[11,12,13],[-2]*right,budget)
                assert len(result)<=budget
                start=result.index(11)
                assert result[start:start+3]==[11,12,13]


def test_context_experiment_excludes_test_and_rejects_cross_split_group():
    rows=[dict(video_id='a',split='train',training_group='one',sha256='a'),
          dict(video_id='b',split='valid',training_group='two',sha256='b'),
          dict(video_id='c',split='test',training_group='three',sha256='c')]
    assert [r['video_id'] for r in development_rows({'speech':rows})]==['a','b']
    rows[2]['training_group']='one'
    with pytest.raises(ValueError): development_rows({'speech':rows})


def test_fewer_false_alarms_cannot_compensate_for_more_misses():
    baseline=dict(fp=2,fn=2,f1=.86)
    assert passes_gate(dict(fp=1,fn=2,f1=.89),baseline)
    assert not passes_gate(dict(fp=0,fn=3,f1=.9),baseline)
    assert not passes_gate(dict(fp=2,fn=1,f1=.9),baseline)
