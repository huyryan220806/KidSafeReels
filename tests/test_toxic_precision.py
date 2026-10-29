import pytest

from ml.toxic_precision import choose_review, development_rows


def test_review_never_selects_test_or_duplicate_source_groups():
    rows, predictions = [], []
    for label in (0, 1):
        for i in range(7):
            sid = f'{label}-{i}'
            split = 'test' if i == 0 else 'train'
            group = f'{label}-{max(i, 2)}'  # i=1 and i=2 are duplicate source groups.
            rows.append(dict(video_id=sid, split=split, training_group=group))
            predictions.append(dict(video_id=sid, split=split, variant='baseline', label=str(label),
                hybrid_flag=str(not bool(label)), score='0.99' if label == 0 else '0.01'))
    selected = choose_review({'speech':rows}, predictions)
    assert len(selected) == len({r['training_group'] for r in selected}) == 10
    assert {r['split'] for r in selected} == {'train'}
    assert len(development_rows({'speech':rows})) == 12


def test_review_refuses_to_fill_shortfall_from_test():
    with pytest.raises(ValueError, match='Not enough'):
        choose_review({'speech':[dict(video_id='test', split='test', training_group='a')]},
            [dict(video_id='test', split='test', variant='baseline', label='0', hybrid_flag='True', score='1')])
