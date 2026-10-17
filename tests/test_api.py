from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from apps.api import main
from kidsafe.schemas import Evidence


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("KIDSAFE_BACKEND", "mock")
    return TestClient(main.app)


def test_mock_never_claims_safety(client):
    response = client.post("/v1/analyze", files={"file": ("clip.mp4", b"mock bytes", "video/mp4")})
    assert response.status_code == 200
    result = response.json()
    assert result["mode"] == "mock"
    assert result["decision"] == "review"
    assert all(score is None for score in result["scores"].values())
    assert result["evaluated_labels"] == []
    assert result["warnings"]


@pytest.mark.parametrize("name,content,expected", [("clip.txt", b"abc", 415), ("clip.mp4", b"", 400)])
def test_invalid_upload(client, name, content, expected):
    assert client.post("/v1/analyze", files={"file": (name, content)}).status_code == expected


def test_upload_limit(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 3)
    assert client.post("/v1/analyze", files={"file": ("clip.mp4", b"1234")}).status_code == 413


def test_model_is_not_silently_replaced_by_mock(client, monkeypatch, tmp_path):
    monkeypatch.setenv("KIDSAFE_BACKEND", "model")
    monkeypatch.setenv("KIDSAFE_MODEL_DIR", str(tmp_path))
    assert client.post("/v1/analyze", files={"file": ("clip.mp4", b"abc")}).status_code == 503


def test_temporary_file_removed_on_pipeline_failure(client, monkeypatch):
    captured = []

    def broken_pipeline(path, **kwargs):
        captured.append(Path(path))
        assert path.exists()
        raise main.ModelNotReadyError("Not ready")

    monkeypatch.setattr(main, "analyze_video", broken_pipeline)
    assert client.post("/v1/analyze", files={"file": ("clip.mp4", b"abc")}).status_code == 503
    assert captured and not captured[0].exists()


def test_evidence_rejects_backwards_timestamp():
    with pytest.raises(ValidationError):
        Evidence(modality="visual", start_seconds=5, end_seconds=2, label="violence", score=0.8, description="test")


def test_invalid_media_returns_422_and_cleans_upload(client, monkeypatch):
    captured = []

    def invalid_media(path, **kwargs):
        captured.append(path)
        raise main.InvalidMediaError("Không giải mã được video.")

    monkeypatch.setattr(main, "analyze_video", invalid_media)
    response = client.post("/v1/analyze", files={"file": ("broken.mp4", b"bad")})
    assert response.status_code == 422
    assert "video" in response.json()["detail"]
    assert not captured[0].exists()
