"""Streamlit AppTest against real loopback HTTP and CPU models, no fake model.

This validates widget state and result delivery, not browser rendering or clicks.
"""
import json
import math
import os
from pathlib import Path
import socket
from tempfile import TemporaryDirectory

from streamlit.testing.v1 import AppTest

from apps.api.worker import AnalysisError, AnalyzerWorker
from apps.web.design import report_text

ROOT = Path(__file__).resolve().parents[1]


def analyze_button(page):
    return next(button for button in page.button if button.label == "Kiểm tra video")


def main():
    base = "http://127.0.0.1:8000"
    os.environ["KIDSAFE_API_URL"] = base
    app_path = str(ROOT / "apps/web/app.py")
    private = ROOT / "data/private/app-acceptance-tien"
    speech = ROOT / "data/processed/smoke/vietnamese-speech.mp4"
    rows = []
    page = AppTest.from_file(app_path, default_timeout=240).run(timeout=240)
    assert not page.exception and analyze_button(page).disabled
    rows.append({"case": "initial-requires-upload", "passed": True})
    page.file_uploader[0].set_value(("fleurs-vietnamese-speech.mp4", speech.read_bytes(), "video/mp4")).run()
    assert not analyze_button(page).disabled
    assert len(page.get("video")) == 1
    analyze_button(page).click().run(timeout=240)
    assert not page.exception
    result = page.session_state["analysis_result"]
    assert result["mode"] == "model"
    assert result["scores"]["violence"] is not None and result["scores"]["toxic_language"] is not None
    assert result["scores"]["sexual"] is None and result["scores"]["dangerous_acts"] is None
    rows.append({"case": "ui-real-http-both-modalities", "passed": True, "mode": result["mode"]})
    assert len(page.get("download_button")) == 2
    assert all(button.proto.url for button in page.get("download_button"))
    (private / "ui-report.txt").write_text(report_text(result, "fleurs-vietnamese-speech.mp4"), encoding="utf-8")
    (private / "ui-report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    rows.append({"case": "download-widgets-with-current-txt-json", "passed": True,
                 "limit": "Widgets and report generation, not browser download."})
    if result["evidence"]:
        target = len(result["evidence"]) - 1
        page.selectbox[0].select_index(target).run()
        expected = math.floor(result["evidence"][target]["start_seconds"])
        assert page.get("video")[0].proto.start_time == expected
        rows.append({"case": "evidence-player-start-time", "passed": True, "start_seconds": expected,
                     "limit": "Widget start_time, not visual playback in a browser."})
    else:
        rows.append({"case": "evidence-player-start-time", "passed": None, "limit": "No evidence returned for this fixture."})
    original_id = result["analysis_id"]
    page.run()
    assert page.session_state["analysis_result"]["analysis_id"] == original_id
    rows.append({"case": "result-survives-widget-rerun", "passed": True})
    page.file_uploader[0].set_value(("silent.mp4", (private / "silent.mp4").read_bytes(), "video/mp4")).run()
    assert "analysis_result" not in page.session_state and not page.get("download_button")
    rows.append({"case": "new-file-clears-old-result", "passed": True})
    page.file_uploader[0].set_value(("broken.mp4", b"INVALID VIDEO", "video/mp4")).run()
    analyze_button(page).click().run()
    assert page.error and "analysis_result" not in page.session_state
    rows.append({"case": "corrupt-file-ui-error", "passed": True})
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        unused_port = sock.getsockname()[1]
    os.environ["KIDSAFE_API_URL"] = f"http://127.0.0.1:{unused_port}"
    offline = AppTest.from_file(app_path, default_timeout=30).run()
    assert not offline.exception and any("Chưa kết nối" in error.value for error in offline.error)
    rows.append({"case": "api-offline-ui-message", "passed": True})
    os.environ["KIDSAFE_API_URL"] = base
    old_directory = os.environ.get("KIDSAFE_MODEL_DIR")
    with TemporaryDirectory(prefix="kidsafe-empty-model-") as empty:
        os.environ["KIDSAFE_MODEL_DIR"] = empty
        runner = AnalyzerWorker()
        try:
            try:
                runner.analyze(private / "silent.mp4", "model")
            except AnalysisError as error:
                assert error.status_code == 503
                rows.append({"case": "missing-artifacts-real-worker-no-fallback", "passed": True, "http_mapping": 503})
            else:
                raise AssertionError("Missing model was silently accepted")
        finally:
            runner.close()
    if old_directory is None:
        os.environ.pop("KIDSAFE_MODEL_DIR", None)
    else:
        os.environ["KIDSAFE_MODEL_DIR"] = old_directory
    report = {"scope": "Actual Streamlit AppTest + real HTTP API + real CPU model, no pilot inference. Not browser/manual acceptance.",
              "passed": sum(row["passed"] is True for row in rows),
              "not_run": sum(row["passed"] is None for row in rows), "total": len(rows), "cases": rows}
    path = ROOT / "reports/app_ui_acceptance_tien.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
