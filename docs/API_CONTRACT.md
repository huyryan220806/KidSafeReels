# Hợp đồng API v1.0

Nguồn schema: `kidsafe/schemas.py`. Schema này là điểm nối giữa hai người, thay đổi cần cập nhật cả app và model.

## Endpoints

- `GET /health` → `{"status":"ok","backend":"mock"}`. Đây là kiểm tra API sống, chưa phải readiness của model.
- `POST /v1/analyze`: multipart/form-data, trường `file`, đuôi `.mp4/.mov/.webm`, không rỗng, tối đa 50 MiB. Đồng bộ một request, trả JSON theo `AnalysisResult`.
- `GET /docs`: tài liệu OpenAPI tương tác.

Ví dụ PowerShell (dùng curl.exe, không dùng alias curl):

```powershell
curl.exe -F "file=@C:\videos\sample.mp4" http://127.0.0.1:8000/v1/analyze
```

Ví dụ mô phỏng hợp lệ:

```json
{
  "schema_version": "1.0",
  "analysis_id": "example-only",
  "mode": "mock",
  "model_version": "mock-v1",
  "decision": "review",
  "scores": {"violence": null, "sexual": null, "toxic_language": null, "dangerous_acts": null},
  "evaluated_labels": [],
  "evidence": [],
  "warnings": ["MÔ PHỎNG: chưa giải mã hoặc phân tích nội dung video."],
  "processing_seconds": 0.01
}
```

## Quy tắc ngữ nghĩa khi tích hợp model thật

- `scores`: điểm [0,1] theo nhãn; không gọi là xác suất đã hiệu chỉnh nếu chưa calibration. `null` = chưa đánh giá/không hỗ trợ; `0` = model đã đánh giá và trả điểm 0.
- `evaluated_labels`: đúng các nhãn có điểm khác null. Model adapter phải giữ nhất quán; starter chưa kiểm tra bất biến này ngoài bộ mock.
- `decision`: `flagged` nếu ít nhất một nhãn được hỗ trợ vượt ngưỡng; `review` nếu bất định, lỗi modality cần thiết hoặc chưa có model; `no_risk_detected` chỉ khi các kiểm tra trong phạm vi model hoàn thành và dưới ngưỡng. UI luôn nói rõ nhãn chưa được hỗ trợ; không hiển thị “an toàn tuyệt đối”.
- `evidence[]`: `{modality, start_seconds, end_seconds, label, score, description}`; modality thuộc visual/audio_text/ocr, timestamp tính từ đầu video. Khi model thật hoạt động, evidence phải đến từ frame/đoạn transcript thật; khoảng thời gian nằm trong thời lượng clip. Starter mới validate start ≥ 0 và end ≥ start.
- `model_version`: định danh artifact và ngưỡng dùng để tái hiện; `processing_seconds` là thời gian server xử lý sau khi framework nhận multipart, không phải toàn bộ thời gian upload.
- Thiếu audio: vẫn có thể đánh giá hình ảnh, nhưng ghi warning và text score = null nếu không có OCR thay thế. Không biến “không nghe thấy” thành điểm an toàn.

## Lỗi hiện có và phần còn thiếu

| HTTP | Hiện tại |
|---|---|
| 400 | File rỗng |
| 413 | Nội dung file > 50 MiB |
| 415 | Đuôi file ngoài whitelist |
| 422 | Thiếu trường file / request sai |
| 503 | Model chưa tích hợp hoặc cấu hình backend không hợp lệ |

**Chưa có:** kiểm tra container/codec/thời lượng và tính hợp lệ media, model readiness, hàng đợi, retry, rate limit, timeout giải mã. Mock chấp nhận byte bất kỳ mang đuôi hợp lệ để test giao tiếp; không đồng nghĩa video hợp lệ. App owner bổ sung giải mã/kiểm tra ≤60 giây và mã lỗi thống nhất trong D2, phối hợp ML lead. Giới hạn 50 MiB trong endpoint áp dụng sau khi framework parse multipart; triển khai online cần giới hạn body ở ingress/proxy.

**Điểm bàn giao:** app owner giữ HTTP/upload/tempfile; ML lead triển khai `kidsafe.pipeline.analyze_video(path, backend="model") -> AnalysisResult`, dùng module trong `ml/`. Không đổi signature độc lập. Không được gọi mock khi model thật lỗi.
