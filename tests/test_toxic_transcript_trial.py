import pytest
from ml.toxic_transcript_trial import evaluate_rows, fit_and_score


def test_missing_asr_keeps_null_probability_and_counts_unflagged():
    samples = [dict(id='a', text='', label=1, rule=False), dict(id='b', text='x', label=0, rule=False),
               dict(id='c', text='y', label=1, rule=True)]
    scores = {'b':.1, 'c':.2}
    decisions, model_metrics, flags = evaluate_rows(samples, scores, 'valid', 'text')
    assert model_metrics['n'] == 2
    assert flags == [False, False, True]
    assert decisions[1]['fn'] == 1 and decisions[1]['missing_asr'] == 1
    with pytest.raises(ValueError):
        evaluate_rows(samples, dict(scores, a=0.), 'valid', 'text')


def test_vectorizer_never_fits_heldout_text_or_missing_speech():
    train = [dict(id=str(i), text=t, label=i%2) for i,t in enumerate(
        ('hôm nay', 'lời khác', 'hôm nay trời', 'lời khác nữa'))]
    held = [dict(id='x', text='zzzzzzzz'), dict(id='y', text='')]
    model, scores = fit_and_score(train, held, 'text')
    assert set(scores)=={'x'}
    assert 'zzz' not in model.steps[0][1].vocabulary_
