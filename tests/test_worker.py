from concurrent.futures import ThreadPoolExecutor

import pytest

from apps.api.worker import AnalysisError, AnalyzerWorker
from tests.worker_fixtures import crashed_worker, impossible_audio_worker, invalid_model_worker, slow_media, slow_model, synthetic_model_worker


@pytest.fixture
def worker():
    runner = AnalyzerWorker()
    yield runner
    runner.close()


def test_process_reused_and_mock_honest(worker, video_files):
    first = worker.analyze(video_files["silent.mp4"], "mock")
    pid = worker._process.pid
    second = worker.analyze(video_files["audio.mp4"], "mock")
    assert worker._process.pid == pid
    assert first.mode == second.mode == "mock"
    assert first.decision == second.decision == "review"
    assert any("không có luồng âm thanh" in warning for warning in first.warnings)
    assert not any("không có luồng âm thanh" in warning for warning in second.warnings)
    assert any("đã được kiểm tra kỹ thuật" in warning for warning in first.warnings)


@pytest.mark.parametrize("target,setting", [(slow_media, "KIDSAFE_MEDIA_TIMEOUT"), (slow_model, "KIDSAFE_MODEL_TIMEOUT")])
def test_timeout_kills_process_before_return(target, setting, monkeypatch, video_files):
    monkeypatch.setenv(setting, "1")
    runner = AnalyzerWorker(target)
    try:
        with pytest.raises(AnalysisError) as error:
            runner.analyze(video_files["silent.mp4"], "mock")
        assert error.value.status_code == 504
        assert runner._process is None
        # A fresh process can serve the next request after timeout.
        from apps.api.worker import worker_loop
        runner._target = worker_loop
        monkeypatch.setenv(setting, "30")
        assert runner.analyze(video_files["silent.mp4"], "mock").mode == "mock"
    finally:
        runner.close()


def test_busy_worker_rejects_without_queue(worker, video_files):
    with worker._lock:
        with ThreadPoolExecutor() as pool:
            future = pool.submit(worker.analyze, video_files["silent.mp4"], "mock")
            with pytest.raises(AnalysisError) as error:
                future.result(timeout=5)
    assert error.value.status_code == 429


@pytest.mark.parametrize("backend", ["model", "unknown"])
def test_no_silent_mock_fallback(worker, video_files, backend):
    with pytest.raises(AnalysisError) as error:
        worker.analyze(video_files["silent.mp4"], backend)
    assert error.value.status_code == 503


def test_worker_crash_is_recoverable(video_files):
    runner = AnalyzerWorker(crashed_worker)
    try:
        with pytest.raises(AnalysisError) as error:
            runner.analyze(video_files["silent.mp4"], "mock")
        assert error.value.status_code == 500
        assert runner._process is None
    finally:
        runner.close()


def test_model_contract_in_isolated_worker(video_files):
    runner = AnalyzerWorker(synthetic_model_worker)
    try:
        result = runner.analyze(video_files["silent.mp4"], "model")
        assert result.mode == "model" and result.evidence
        assert "KIỂM THỬ" in result.warnings[0]
    finally:
        runner.close()


def test_worker_rejects_evidence_outside_video(video_files):
    runner = AnalyzerWorker(invalid_model_worker)
    try:
        with pytest.raises(AnalysisError) as error:
            runner.analyze(video_files["silent.mp4"], "model")
        assert error.value.status_code == 502
    finally:
        runner.close()


def test_worker_rejects_audio_evidence_when_no_audio(video_files):
    runner = AnalyzerWorker(impossible_audio_worker)
    try:
        with pytest.raises(AnalysisError) as error:
            runner.analyze(video_files["silent.mp4"], "model")
        assert error.value.status_code == 502
    finally:
        runner.close()


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "601", "invalid"])
def test_bad_timeout_configuration(worker, video_files, monkeypatch, value):
    monkeypatch.setenv("KIDSAFE_MEDIA_TIMEOUT", value)
    with pytest.raises(AnalysisError) as error:
        worker.analyze(video_files["silent.mp4"], "mock")
    assert error.value.status_code == 503
    assert worker._process is None
