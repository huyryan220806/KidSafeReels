from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from apps.api import main
from apps.api.worker import AnalysisError
from kidsafe.schemas import Evidence


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("KIDSAFE_BACKEND", "mock")
    with TestClient(main.app) as connection:
        yield connection


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "backend": "mock"}


def test_mock_never_claims_safety(client, video_files):
    response = client.post("/v1/analyze", files={"file": ("clip.mp4", video_files["silent.mp4"].read_bytes(), "video/mp4")})
    assert response.status_code == 200
    result = response.json()
    assert result["mode"] == "mock"
    assert result["decision"] == "review"
    assert all(score is None for score in result["scores"].values())
    assert result["evaluated_labels"] == []
    assert result["warnings"]
    assert result["processing_seconds"] > 0


@pytest.mark.parametrize("name,content,expected", [("clip.txt", b"abc", 415), ("clip.mp4", b"", 400), ("clip.mp4", b"fake", 422)])
def test_invalid_upload(client, name, content, expected):
    assert client.post("/v1/analyze", files={"file": (name, content)}).status_code == expected


def test_missing_file(client):
    assert client.post("/v1/analyze").status_code == 422


def test_upload_limit(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 3)
    assert client.post("/v1/analyze", files={"file": ("clip.mp4", b"1234")}).status_code == 413


def test_long_video(client, video_files):
    assert client.post("/v1/analyze", files={"file": ("clip.mp4", video_files["long.mp4"].read_bytes())}).status_code == 422


def test_model_is_not_silently_replaced_by_mock(client, monkeypatch, video_files):
    monkeypatch.setenv("KIDSAFE_BACKEND", "model")
    assert client.post("/v1/analyze", files={"file": ("clip.mp4", video_files["silent.mp4"].read_bytes())}).status_code == 503


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_temporary_file_removed_on_failure(client, monkeypatch, code):
    captured = []

    def broken_worker(path, **kwargs):
        captured.append(Path(path))
        assert path.exists()
        assert path.name == "video.mp4"
        raise AnalysisError(code, "Test error")

    monkeypatch.setattr(main.app.state.analyzer, "analyze", broken_worker)
    response = client.post("/v1/analyze", files={"file": ("../../private.mp4", b"abc")})
    assert response.status_code == code
    assert captured and not captured[0].exists()
    assert not captured[0].parent.exists()
    if code == 429:
        assert response.headers["Retry-After"] == "5"


def test_temporary_file_removed_on_success(client, monkeypatch, video_files):
    captured = []
    original = main.app.state.analyzer.analyze

    def record_path(path, **kwargs):
        captured.append(Path(path))
        return original(path, **kwargs)

    monkeypatch.setattr(main.app.state.analyzer, "analyze", record_path)
    assert client.post("/v1/analyze", files={"file": ("CLIP.MP4", video_files["silent.mp4"].read_bytes())}).status_code == 200
    assert captured and not captured[0].parent.exists()


def test_evidence_rejects_backwards_timestamp():
    with pytest.raises(ValidationError):
        Evidence(modality="visual", start_seconds=5, end_seconds=2, label="violence", score=0.8, description="test")
