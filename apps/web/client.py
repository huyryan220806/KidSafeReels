"""HTTP boundary and display labels for the web app."""

import requests

from apps.validation import validate_result

LABELS = {
    "violence": "Bạo lực",
    "sexual": "Nội dung gợi dục",
    "toxic_language": "Ngôn ngữ độc hại",
    "dangerous_acts": "Hành vi nguy hiểm",
}
MODALITIES = {"visual": "Hình ảnh", "audio_text": "Lời nói", "ocr": "Chữ trên video"}
DECISIONS = {
    "no_risk_detected": "Chưa phát hiện rủi ro trong phạm vi đã đánh giá",
    "review": "Cần người lớn kiểm tra",
    "flagged": "Phát hiện dấu hiệu rủi ro",
}


def check_health(api_url):
    try:
        response = requests.get(f"{api_url}/health", timeout=(2, 3))
        response.raise_for_status()
        health = response.json()
        if not isinstance(health, dict) or health.get("status") != "ok":
            return None
        return health
    except (requests.RequestException, ValueError):
        return None


def error_message(response):
    defaults = {
        400: "File video rỗng. Hãy chọn file khác.",
        413: "Video vượt giới hạn 50 MiB.",
        415: "Định dạng video chưa được hỗ trợ. Hãy dùng MP4, MOV hoặc WebM.",
        422: "Video hoặc yêu cầu không hợp lệ. Hãy kiểm tra file đã chọn.",
        429: "Máy đang bận phân tích video khác. Hãy chờ rồi thử lại.",
        500: "Phân tích thất bại. Kiểm tra log API rồi thử lại.",
        502: "Kết quả chưa đúng hợp đồng API. Cần kiểm tra tích hợp model.",
        503: "Model chưa sẵn sàng hoặc cấu hình API chưa hợp lệ.",
        504: "Quá thời gian xử lý. Hãy thử video ngắn hơn.",
    }
    try:
        payload = response.json()
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if isinstance(detail, str) and detail.strip():
            return detail[:500]
    except ValueError:
        pass
    return defaults.get(response.status_code, "Không thể hoàn tất yêu cầu. Kiểm tra kết nối API rồi thử lại.")


def analyze_upload(api_url, upload, timeout):
    response = requests.post(
        f"{api_url}/v1/analyze",
        files={"file": (upload.name, upload.getvalue(), upload.type or "application/octet-stream")},
        timeout=(5, timeout),
    )
    response.raise_for_status()
    return validate_result(response.json()).model_dump(mode="json")


def timestamp(seconds):
    minutes, remaining = divmod(seconds, 60)
    return f"{int(minutes):02d}:{remaining:04.1f}"
