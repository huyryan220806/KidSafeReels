from copy import deepcopy

import pytest

from ml.speech_review_followup import audio_windows, decode_variant, resolve_annotations
from ml.speech_intake import validate


def test_explicit_negative_correction_keeps_original_evidence_without_positive_conflict():
    original = dict(manifest_sha256='digest',annotator='huy',rows=[dict(video_id='a',annotator='huy',
        watched_full=True,language='vi',toxic_language=1,evidence=[dict(start_seconds=0,end_seconds=1,reason='quote')],
        corrected_transcript='',transcript_verified=False,notes='context ambiguous')])
    before=deepcopy(original)
    result=resolve_annotations(original,dict(source='human chat',confirmed_by_user=True,
        rows=[dict(video_id='a',toxic_language=0,reason='explicitly confirmed negative')]))
    assert original==before
    assert result['rows'][0]['evidence']==[]
    assert result['rows'][0]['prior_evidence_for_adjudication']==before['rows'][0]['evidence']
    assert validate(result,dict(assignments={'huy':['a']},rows=[dict(video_id='a',duration_seconds=2)]),'huy','digest')['negative']==1


def test_unconfirmed_or_unknown_corrections_are_not_accepted():
    original=dict(rows=[dict(video_id='a',toxic_language=None,evidence=[])])
    with pytest.raises(ValueError,match='explicit'):
        resolve_annotations(original,dict(source='model',confirmed_by_user=False,rows=[]))
    with pytest.raises(ValueError,match='Invalid'):
        resolve_annotations(original,dict(source='chat',confirmed_by_user=True,
            rows=[dict(video_id='wrong',toxic_language=1,reason='x')]))


def test_chunk_windows_cover_tail_without_empty_extra_window():
    assert audio_windows(10*16000)==[(0,160000)]
    assert audio_windows(19*16000)==[(0,160000),(144000,304000)]
    assert audio_windows(20*16000)==[(0,160000),(144000,304000),(288000,320000)]
    with pytest.raises(ValueError):audio_windows(0)


def test_chunk_timestamps_are_absolute_and_overlap_keeps_one_core_copy():
    np=pytest.importorskip('numpy')
    from types import SimpleNamespace
    class ASR:
        calls=0
        def transcribe(self,audio,**kwargs):
            self.calls+=1
            # Same utterance at 9.0-10.0 in both overlapping windows.
            start,end=(9.,10.) if self.calls==1 else (0.,1.)
            segment=SimpleNamespace(start=start,end=end,text='speech',avg_logprob=-.2,compression_ratio=1.,no_speech_prob=0.)
            return iter([segment]),None
    result=decode_variant(ASR(),np.zeros(19*16000,dtype=np.float32),'vi_chunk10')
    assert len(result)==1
    assert result[0]['start']==9 and result[0]['end']==10
