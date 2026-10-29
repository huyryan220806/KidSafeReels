from types import SimpleNamespace

import pytest

from ml.asr_recovery import repeated_asr, recover_repetition


def segments(texts):
    return [dict(start=i,end=i+1,text=t,avg_logprob=-.2,compression_ratio=1.) for i,t in enumerate(texts)]


def test_repetition_trigger_requires_long_exact_dominant_phrases():
    assert repeated_asr(segments(['nó chứa trên đầu']*4))
    assert not repeated_asr(segments(['nó chứa trên đầu']*3))
    assert not repeated_asr(segments(['ừ đúng']*10))
    assert not repeated_asr(segments(['nó chứa trên đầu']*4+[f'lời khác số {i}' for i in range(5)]))


@pytest.mark.parametrize('language,probability', [('en',.99),('vi',.49)])
def test_retry_never_forces_unconfirmed_vietnamese(language,probability):
    original=segments(['repeated long phrase']*4)
    class ASR:
        def transcribe(self,*args,**kwargs):raise AssertionError('Must not retry')
    result,warning=recover_repetition(ASR(),[],original,language=language,language_probability=probability)
    assert result is original and warning is None


def test_non_repeated_transcript_is_kept_without_extra_asr():
    original=segments(['câu bình thường khác nhau'])
    result,warning=recover_repetition(None,[],original,language='vi',language_probability=.9)
    assert result is original and warning is None


@pytest.mark.parametrize('mode', ['empty','error','repeated'])
def test_failed_retry_preserves_original_and_warns(mode):
    np=pytest.importorskip('numpy')
    original=segments(['nó chứa trên đầu']*4)
    class ASR:
        def transcribe(self,*args,**kwargs):
            if mode=='error':raise RuntimeError('decoder unavailable')
            rows=[] if mode=='empty' else [SimpleNamespace(start=i,end=i+1,text='nó chứa trên đầu',
                avg_logprob=-.2,compression_ratio=1.,no_speech_prob=0.) for i in range(4)]
            return iter(rows),None
    result,warning=recover_repetition(ASR(),np.zeros(5*16000),original,language='vi',language_probability=.9)
    assert result is original and warning


def test_non_toxic_retry_is_accepted_without_searching_for_profane_words():
    np=pytest.importorskip('numpy')
    class ASR:
        def transcribe(self,*args,**kwargs):
            assert kwargs['language']=='vi' and kwargs.get('initial_prompt') is None
            return iter([SimpleNamespace(start=0,end=1,text='xin chào các bạn',avg_logprob=-.2,
                compression_ratio=1.,no_speech_prob=0.)]),None
    result,warning=recover_repetition(ASR(),np.zeros(16000),segments(['nó chứa trên đầu']*4),language='vi',language_probability=.9)
    assert result[0]['text']=='xin chào các bạn' and warning
