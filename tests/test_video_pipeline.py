import pytest

from kidsafe.errors import InvalidMediaError, ModelNotReadyError
from ml.inference import VideoService, combine_result


class FakeText:
    def predict(self, text):
        return {"score": 0.8, "flagged": True}


def test_missing_modalities_stay_unknown_and_below_threshold_stays_review():
    result = combine_result(0.1, 0.5, [], FakeText(), 3, "test", ["missing audio"])
    assert result.mode == "model"
    assert result.decision == "review"
    assert result.scores.toxic_language is None
    assert result.scores.sexual is None
    assert result.evaluated_labels == ["violence"]


def test_audio_evidence_clipped_to_video_duration():
    result = combine_result(0.1, 0.5, [(-0.1, 8, "test text")], FakeText(), 3, "test", [])
    assert result.decision == "flagged"
    assert result.scores.toxic_language == 0.8
    assert result.evidence[0].start_seconds == 0
    assert result.evidence[0].end_seconds == 3
    assert "toxic_language" in result.evaluated_labels


def test_visual_evidence_is_clip_level():
    result = combine_result(0.9, 0.5, [], FakeText(), 5, "test", [])
    assert result.evidence[0].start_seconds == 0
    assert result.evidence[0].end_seconds == 5
    assert "toàn video" in result.evidence[0].description


def test_temporal_service_reads_video_windows_without_using_static_frame_path(monkeypatch):
    pytest.importorskip("cv2")
    pytest.importorskip("av")
    from types import SimpleNamespace
    from ml import media
    service = VideoService.__new__(VideoService)
    service.visual = SimpleNamespace(is_temporal=True, frame_count=80,
        score_video=lambda path: .9, threshold=.41, version="temporal-test")
    service.text = SimpleNamespace(version="text-test")
    monkeypatch.setattr(media, "inspect_video", lambda path: media.VideoInfo(2, False))
    def unexpected(*args, **kwargs):
        raise AssertionError("Temporal service used the static frame path")
    monkeypatch.setattr(media, "sample_frames", unexpected)
    result = service.analyze("fixture.mp4")
    assert result.scores.violence == .9 and result.scores.toxic_language is None
    assert result.evidence[0].start_seconds == 0 and result.evidence[0].end_seconds == 2
    assert "80 frame" in result.evidence[0].description


def test_missing_artifacts_fail_before_download(tmp_path):
    with pytest.raises(ModelNotReadyError):
        VideoService(tmp_path)


def test_corrupt_media_and_duration_limit(tmp_path):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("av")
    import numpy as np
    from ml.media import inspect_video, sample_frames

    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a video")
    with pytest.raises(InvalidMediaError):
        inspect_video(corrupt)
    path = tmp_path / "silent.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    assert writer.isOpened()
    for _ in range(20):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    info = inspect_video(path)
    assert info.duration == pytest.approx(2, abs=0.1)
    assert not info.has_audio
    assert len(sample_frames(path)) == 8
    with pytest.raises(InvalidMediaError):
        inspect_video(path, max_seconds=1)


def test_group_split_keeps_related_clips_together():
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    pytest.importorskip("av")
    import numpy as np
    from ml.train_visual import make_split, similar_groups

    assert list(similar_groups([[0, 0, 0], [1, 1, 1], [2**64-1]*3])) == [0, 0, 2]
    labels = np.array([i % 2 for i in range(100)])
    groups = np.repeat(np.arange(50), 2)
    splits = make_split(labels, groups)
    assert not set(groups[splits["train"]]) & set(groups[splits["test"]])
    assert not set(groups[splits["valid"]]) & set(groups[splits["test"]])


def test_whisper_audio_decoder_dependency_compatibility(tmp_path):
    pytest.importorskip("faster_whisper")
    import wave
    import numpy as np
    from faster_whisper.audio import decode_audio

    path = tmp_path / "tone.wav"
    samples = (np.sin(np.arange(16000) * 2 * np.pi * 440 / 16000) * 16000).astype("<i2")
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(samples.tobytes())
    decoded = decode_audio(str(path), sampling_rate=16000)
    assert len(decoded) == 16000
    assert np.isfinite(decoded).all()
    assert np.max(np.abs(decoded)) > 0.4


def test_conflicting_duplicate_labels_are_identified_before_split(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    pytest.importorskip("av")
    from ml.train_visual import scan_videos

    paths = []
    for label in ("Violence", "NonViolence"):
        directory = tmp_path / label
        directory.mkdir()
        path = directory / "same.mp4"
        path.write_bytes(b"same video bytes with incompatible labels")
        paths.append(path)
    unique = tmp_path / "NonViolence" / "unique.mp4"
    unique.write_bytes(b"another video")
    entries, conflicts = scan_videos(paths + [unique])
    assert len(conflicts) == 1
    retained = [path for path, _, digest in entries if digest not in conflicts]
    assert retained == [unique]


def test_failed_asr_preserves_visual_and_unknown_toxic(monkeypatch):
    pytest.importorskip("av")
    pytest.importorskip("cv2")
    whisper_audio = pytest.importorskip("faster_whisper.audio")
    from types import SimpleNamespace
    import numpy as np
    from ml import media

    service = VideoService.__new__(VideoService)
    service.visual = SimpleNamespace(score=lambda frames: 0.2, threshold=0.5, version="test-visual")
    service.text = SimpleNamespace(version="test-text")
    monkeypatch.setattr(media, "inspect_video", lambda path: media.VideoInfo(2, True))
    monkeypatch.setattr(media, "sample_frames", lambda path: [])
    monkeypatch.setattr(whisper_audio, "decode_audio", lambda *args, **kwargs: np.array([], dtype=np.float32))
    result = service.analyze("empty-audio.mp4")
    assert result.scores.violence == 0.2
    assert result.scores.toxic_language is None
    assert result.decision == "review"
    assert any("ASR" in warning and "ValueError" in warning for warning in result.warnings)


@pytest.mark.parametrize("language,probability,texts,expected", [
    ("vi", .9, ["lời nói do decoder trả về"], .8),
    ("vi", .9, ["   "], None),
    ("vi", .9, [], None),
    ("en", .9, ["speech"], None),
    ("vi", .49, ["không chắc ngôn ngữ"], None),
])
def test_asr_native_filter_not_overridden_by_no_speech_probability(monkeypatch, language, probability, texts, expected):
    """Whisper already applies joint no-speech/log-probability filtering before yielding."""
    pytest.importorskip("av")
    pytest.importorskip("cv2")
    whisper_audio = pytest.importorskip("faster_whisper.audio")
    from types import SimpleNamespace
    import numpy as np
    from ml import media
    service = VideoService.__new__(VideoService)
    service.visual = SimpleNamespace(is_temporal=True, score_video=lambda path: .1,
                                     frame_count=80, threshold=.41, version="visual-test")
    service.text = FakeText()
    service.text.version = "text-test"
    service.asr_version = "asr-test"
    service.asr = SimpleNamespace(transcribe=lambda *args, **kwargs: (
        iter([SimpleNamespace(start=0., end=2., text=t, no_speech_prob=.9, avg_logprob=-.4) for t in texts]),
        SimpleNamespace(language=language, language_probability=probability)))
    monkeypatch.setattr(media, "inspect_video", lambda path: media.VideoInfo(3, True))
    monkeypatch.setattr(whisper_audio, "decode_audio", lambda *args, **kwargs: np.ones(48000, dtype=np.float32))
    result = service.analyze("fixture.mp4")
    assert result.scores.toxic_language == expected
    assert result.decision == ("flagged" if expected is not None else "review")
    assert ("toxic_language" in result.evaluated_labels) == (expected is not None)


@pytest.mark.parametrize('repeated', [False, True])
def test_service_retries_only_repeated_asr_and_reports_actual_rule_evidence(monkeypatch, repeated):
    pytest.importorskip('av'); pytest.importorskip('cv2')
    whisper_audio=pytest.importorskip('faster_whisper.audio')
    import numpy as np
    from types import SimpleNamespace
    from ml import media
    from ml.toxic_rules import ToxicRules
    service=VideoService.__new__(VideoService)
    service.visual=SimpleNamespace(is_temporal=True,score_video=lambda path:.1,frame_count=80,threshold=.41,version='visual')
    service.text=SimpleNamespace(version='text',predict=lambda text:dict(score=.1,flagged=False))
    service.asr_version='small';service.asr_recovery_enabled=True;service.toxic_rules=ToxicRules()
    calls=[]
    def transcribe(*args,**kwargs):
        calls.append(kwargs)
        texts=['nó chứa trên đầu']*4 if len(calls)==1 and repeated else ['xin chào các bạn']
        if len(calls)>1:texts=['địt mẹ']
        return iter([SimpleNamespace(start=i,end=i+1,text=t,avg_logprob=-.2,compression_ratio=1.,no_speech_prob=.1)
                     for i,t in enumerate(texts)]),SimpleNamespace(language='vi',language_probability=.9)
    service.asr=SimpleNamespace(transcribe=transcribe)
    monkeypatch.setattr(media,'inspect_video',lambda path:media.VideoInfo(5,True))
    monkeypatch.setattr(whisper_audio,'decode_audio',lambda *a,**kw:np.ones(5*16000,dtype=np.float32))
    result=service.analyze('fixture.mp4')
    assert len(calls)==(2 if repeated else 1)
    assert result.scores.toxic_language==.1  # Rule never fabricates a higher model score.
    assert result.decision==('flagged' if repeated else 'review')
    assert 'asr-repeat-recovery-v1' in result.model_version
    assert any('ASR phát hiện transcript lặp' in w for w in result.warnings)==repeated
    if repeated:
        assert calls[1]['language']=='vi'
        assert result.evidence[0].start_seconds==0 and result.evidence[0].end_seconds==1
