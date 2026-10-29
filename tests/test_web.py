import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests
import streamlit as st
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from apps.api.main import app
from apps.web.client import error_message
from kidsafe.pipeline import analyze_video
from tests.worker_fixtures import synthetic_result

APP = Path(__file__).resolve().parents[1] / "apps" / "web" / "app.py"


@pytest.fixture(autouse=True)
def health(monkeypatch):
    st.cache_data.clear()
    monkeypatch.setenv("KIDSAFE_BACKEND", "mock")
    monkeypatch.setattr("apps.web.client.check_health", lambda url: {"status": "ok", "backend": "mock"})


def analyze_button(page):
    return next(button for button in page.button if button.label == "Kiểm tra video")


def page_with_upload(video_files):
    page = AppTest.from_file(str(APP), default_timeout=30).run(timeout=30)
    page.file_uploader[0].set_value(("sample.mp4", video_files["silent.mp4"].read_bytes(), "video/mp4")).run()
    return page


def response_with(payload):
    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)


def test_initial_page_requires_upload():
    page = AppTest.from_file(str(APP), default_timeout=30).run(timeout=30)
    assert not page.exception
    assert analyze_button(page).disabled


def test_mock_result_persists_and_new_upload_resets(video_files):
    result = analyze_video(video_files["silent.mp4"]).model_dump()
    with patch("requests.post", return_value=response_with(result)) as request:
        page = page_with_upload(video_files)
        analyze_button(page).click().run()
        page.run()
        assert request.call_count == 1
        assert not page.exception
        assert any("Cần người lớn kiểm tra" in block.value for block in page.markdown)
        assert any("KẾT QUẢ MÔ PHỎNG" in warning.value for warning in page.warning)
        assert list(page.table[0].value["Chỉ số phân tích"]) == ["Chưa đánh giá"] * 4
        assert len(page.get("download_button")) == 2
        details = next(expander for expander in page.expander if expander.label.startswith("Chi tiết phân tích"))
        assert not details.proto.expanded
        page.file_uploader[0].set_value(("new.mp4", video_files["audio.mp4"].read_bytes(), "video/mp4")).run()
        assert "analysis_result" not in page.session_state
        assert not page.table
        assert not page.get("download_button")


def test_model_evidence_seeks_video_and_zero_is_not_null(video_files):
    result = synthetic_result().model_dump()
    result["scores"]["toxic_language"] = 0
    result["evaluated_labels"].append("toxic_language")
    result["evidence"].append({**result["evidence"][0], "start_seconds": 1.1, "end_seconds": 1.5})
    with patch("requests.post", return_value=response_with(result)):
        page = page_with_upload(video_files)
        analyze_button(page).click().run()
        assert not page.exception
        assert list(page.table[0].value["Chỉ số phân tích"]) == ["0.800", "Chưa đánh giá", "0.000", "Chưa đánh giá"]
        page.selectbox[0].select_index(1).run()
        assert page.get("video")[0].proto.start_time == 1
        assert "analysis_result" in page.session_state


def test_hybrid_rule_warning_is_visible_even_with_low_model_score(video_files):
    result = synthetic_result().model_dump()
    result["model_version"] += "+toxic-hybrid-v1-test"
    result["scores"]["violence"] = .01
    result["scores"]["toxic_language"] = .02
    result["evaluated_labels"].append("toxic_language")
    result["evidence"] = [{"modality": "audio_text", "label": "toxic_language", "score": .02,
                           "start_seconds": .2, "end_seconds": .8,
                           "description": "Bộ luật từ/cụm từ phát hiện: ví dụ kiểm thử tổng hợp."}]
    with patch("requests.post", return_value=response_with(result)):
        page = page_with_upload(video_files)
        analyze_button(page).click().run()
    assert not page.exception
    assert page.session_state["analysis_result"]["decision"] == "flagged"
    assert any("bộ luật" in block.value for block in page.caption)
    assert any("Bộ luật từ/cụm từ phát hiện" in block.value for block in page.markdown)
    assert "0.020" in list(page.table[0].value["Chỉ số phân tích"])
    assert len(page.get("download_button")) == 2


@pytest.mark.parametrize("exception,text", [
    (requests.Timeout(), "Hết thời gian"),
    (requests.ConnectionError(), "Chưa kết nối"),
    (requests.RequestException(), "Không gửi được"),
])
def test_network_errors(video_files, exception, text):
    with patch("requests.post", side_effect=exception):
        page = page_with_upload(video_files)
        analyze_button(page).click().run()
    assert not page.exception
    assert any(text in error.value for error in page.error)
    assert "analysis_result" not in page.session_state


@pytest.mark.parametrize("code", [413, 422, 429, 500, 502, 503, 504])
def test_http_errors_are_readable(video_files, code):
    response = requests.Response()
    response.status_code = code
    response._content = json.dumps({"detail": f"Thông báo kiểm thử {code}"}).encode()
    with patch("requests.post", return_value=response):
        page = page_with_upload(video_files)
        analyze_button(page).click().run()
    assert not page.exception
    assert any(f"Thông báo kiểm thử {code}" in block.value for block in [*page.error, *page.markdown])
    assert not page.get("download_button")


def test_invalid_response_never_renders_result(video_files):
    with patch("requests.post", return_value=response_with({"mode": "model"})):
        page = page_with_upload(video_files)
        analyze_button(page).click().run()
    assert not page.exception
    assert any("Chưa đọc được kết quả" in error.value for error in page.error)
    assert not page.table


def test_empty_file_rejected_before_network(video_files):
    with patch("requests.post") as request:
        page = page_with_upload(video_files)
        page.file_uploader[0].set_value(("empty.mp4", b"", "video/mp4")).run()
        analyze_button(page).click().run()
        request.assert_not_called()
    assert any("rỗng" in error.value for error in page.error)


def test_error_does_not_leave_old_result(video_files):
    with patch("requests.post", return_value=response_with(analyze_video(video_files["silent.mp4"]).model_dump())):
        page = page_with_upload(video_files)
        analyze_button(page).click().run()
    with patch("requests.post", side_effect=requests.Timeout()):
        analyze_button(page).click().run()
    assert "analysis_result" not in page.session_state
    assert not page.table


def test_api_offline_is_visible(monkeypatch):
    monkeypatch.setattr("apps.web.client.check_health", lambda url: None)
    page = AppTest.from_file(str(APP), default_timeout=30).run(timeout=30)
    assert not page.exception
    assert any("Chưa kết nối được dịch vụ" in error.value for error in page.error)


def test_non_json_error_hides_html():
    response = requests.Response()
    response.status_code = 413
    response._content = b"<html>proxy error</html>"
    assert "50 MiB" in error_message(response)
    assert "<html>" not in error_message(response)


def test_web_to_real_api_to_real_decoder(video_files):
    with TestClient(app) as client:
        def post(url, files, timeout):
            return client.post("/v1/analyze", files=files)
        with patch("requests.post", side_effect=post):
            page = page_with_upload(video_files)
            analyze_button(page).click().run(timeout=30)
            assert not page.exception
            assert page.session_state["analysis_result"]["mode"] == "mock"
            assert any("không có luồng âm thanh" in block.value for block in page.markdown)
