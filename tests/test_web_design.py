from apps.web.design import filename_card, report_text
from tests.worker_fixtures import synthetic_result


def test_uploaded_name_cannot_inject_html():
    rendered = filename_card('<img src=x onerror="alert(1)">.mp4', 123)
    assert '<img' not in rendered
    assert '&lt;img' in rendered


def test_readable_report_preserves_unknown_and_zero():
    result = synthetic_result().model_dump()
    result["scores"]["toxic_language"] = 0
    result["evaluated_labels"].append("toxic_language")
    report = report_text(result, "video.mp4")
    assert "Nội dung gợi dục: Chưa đánh giá" in report
    assert "Ngôn ngữ độc hại: Có kết quả; chỉ số phân tích 0.000" in report
    assert "Bằng chứng tổng hợp chỉ dùng cho test." in report
    assert "không bảo đảm an toàn tuyệt đối" in report


def test_mock_report_always_marks_simulation(video_files):
    from kidsafe.pipeline import analyze_video

    result = analyze_video(video_files["silent.mp4"]).model_dump()
    report = report_text(result, "video.mp4")
    assert "KẾT QUẢ MÔ PHỎNG" in report
    assert "chưa đánh giá nội dung video" in report


def test_hybrid_report_explains_low_model_score_and_rule_evidence():
    from apps.web.design import toxic_method_note
    from ml.inference import combine_result
    from ml.toxic_rules import load_rules
    from types import SimpleNamespace
    text = SimpleNamespace(predict=lambda text: {"score": .02, "flagged": False})
    result = combine_result(.1, .41, [(0, 1, "đồ ngu", {"avg_logprob": -.4, "compression_ratio": 1.2})],
                            text, 2, "fixture+" + load_rules().version, [], toxic_rules=load_rules()).model_dump()
    report = report_text(result, "fixture.mp4")
    assert "0.020" in report and "Bộ luật từ/cụm từ" in report
    assert "Chỉ số hiển thị là điểm model" in toxic_method_note(result)
    result["mode"] = "mock"
    assert toxic_method_note(result) is None
