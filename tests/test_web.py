from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from kidsafe.pipeline import analyze_video

APP = Path(__file__).resolve().parents[1] / "apps" / "web" / "app.py"


def test_initial_page_requires_upload():
    page = AppTest.from_file(str(APP)).run(timeout=30)
    assert not page.exception
    assert page.button[0].disabled


def test_mock_result_displays_unknown_scores(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"mock payload")
    result = analyze_video(video).model_dump()
    upload = SimpleNamespace(name="sample.mp4", size=12, type="video/mp4", getvalue=lambda: b"mock payload")
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: result)
    with patch("streamlit.file_uploader", return_value=upload), patch("requests.post", return_value=response):
        page = AppTest.from_file(str(APP)).run(timeout=30)
        page.button[0].click().run(timeout=30)
        page.run(timeout=30)  # Results survive widget reruns, e.g. downloading JSON.
    assert not page.exception
    assert page.subheader[0].value == "Cần người lớn kiểm tra"
    assert "MÔ PHỎNG" in page.warning[0].value
    assert list(page.table[0].value["Điểm model"]) == ["Chưa đánh giá"] * 4
