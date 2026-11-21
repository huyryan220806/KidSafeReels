import pytest

from scripts.make_demo_videos import make_video


@pytest.fixture(scope="session")
def video_files(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-videos")
    files = {}
    for name, options in {
        "silent.mp4": {}, "audio.mp4": {"audio": True}, "clip.mov": {}, "clip.webm": {},
        "boundary.mp4": {"seconds": 60, "fps": 2}, "long.mp4": {"seconds": 61, "fps": 2},
        "wrong-container.mp4": {"container_format": "matroska"},
    }.items():
        files[name] = make_video(root / name, **options)
    files["broken.mp4"] = root / "broken.mp4"
    files["broken.mp4"].write_bytes(b"not a video")
    return files
