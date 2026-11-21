"""Process targets used only by tests; no test backend is exposed by the app."""

import time


def slow_media(connection):
    connection.recv()
    time.sleep(60)


def slow_model(connection):
    connection.recv()
    connection.send(("media",))
    time.sleep(60)


def crashed_worker(connection):
    connection.recv()
    connection.close()


def synthetic_result(*args, **kwargs):
    from kidsafe.schemas import AnalysisResult, Evidence, RiskScores

    return AnalysisResult(
        analysis_id="synthetic-test-only", mode="model", model_version="synthetic-test-only",
        decision="flagged", scores=RiskScores(violence=0.8), evaluated_labels=["violence"],
        evidence=[Evidence(modality="visual", start_seconds=0.2, end_seconds=1.0,
                           label="violence", score=0.8, description="Bằng chứng tổng hợp chỉ dùng cho test.")],
        warnings=["DỮ LIỆU KIỂM THỬ TỔNG HỢP — không phải kết quả model."], processing_seconds=0,
    )


def synthetic_model_worker(connection):
    import kidsafe.pipeline
    from apps.api.worker import worker_loop

    kidsafe.pipeline.analyze_video = synthetic_result
    worker_loop(connection)


def invalid_result(*args, **kwargs):
    result = synthetic_result()
    result.evidence[0].end_seconds = 100
    return result


def invalid_model_worker(connection):
    import kidsafe.pipeline
    from apps.api.worker import worker_loop

    kidsafe.pipeline.analyze_video = invalid_result
    worker_loop(connection)


def impossible_audio_result(*args, **kwargs):
    result = synthetic_result()
    result.evidence[0].modality = "audio_text"
    return result


def impossible_audio_worker(connection):
    import kidsafe.pipeline
    from apps.api.worker import worker_loop

    kidsafe.pipeline.analyze_video = impossible_audio_result
    worker_loop(connection)


def rejected_media_result(*args, **kwargs):
    from kidsafe.errors import InvalidMediaError

    raise InvalidMediaError("Luồng âm thanh không hợp lệ trong kiểm thử tích hợp.")


def rejected_media_worker(connection):
    import kidsafe.pipeline
    from apps.api.worker import worker_loop

    kidsafe.pipeline.analyze_video = rejected_media_result
    worker_loop(connection)
