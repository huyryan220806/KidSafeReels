import pytest

from apps.api.media import MediaError, inspect_video


@pytest.mark.parametrize("name", ["silent.mp4", "audio.mp4", "clip.mov", "clip.webm", "boundary.mp4"])
def test_real_decoding(video_files, name):
    info = inspect_video(video_files[name])
    assert info.frames > 0
    assert 0 < info.duration <= 60.001
    assert info.has_audio == (name == "audio.mp4")


@pytest.mark.parametrize("name,status", [("broken.mp4", 422), ("long.mp4", 422), ("wrong-container.mp4", 415)])
def test_bad_media(video_files, name, status):
    with pytest.raises(MediaError) as error:
        inspect_video(video_files[name])
    assert error.value.status_code == status


def test_frame_limit(video_files, monkeypatch):
    monkeypatch.setattr("apps.api.media.MAX_FRAMES", 1)
    with pytest.raises(MediaError, match="khung hình"):
        inspect_video(video_files["silent.mp4"])


def test_resolution_limit(video_files, monkeypatch):
    monkeypatch.setattr("apps.api.media.MAX_PIXELS", 100)
    with pytest.raises(MediaError, match="phân giải"):
        inspect_video(video_files["silent.mp4"])
