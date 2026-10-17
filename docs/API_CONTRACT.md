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
- `evaluated_labels`: đúng các nhãn có điểm khác null. Pipeline model tạo danh sách từ các điểm thực có; schema chưa tự kiểm tra bất biến này với mọi dữ liệu bên ngoài.
- `decision`: `flagged` nếu ít nhất một nhãn được hỗ trợ đạt/vượt ngưỡng; `review` nếu bất định, lỗi modality cần thiết hoặc chưa có model. Pipeline v1 hiện trả `review` cho mọi ca dưới ngưỡng vì chưa đánh giá trên video đích; chưa dùng `no_risk_detected`. UI luôn nói rõ nhãn chưa được hỗ trợ; không hiển thị “an toàn tuyệt đối”.
- `evidence[]`: `{modality, start_seconds, end_seconds, label, score, description}`; modality thuộc visual/audio_text/ocr, timestamp tính từ đầu video. Visual v1 là điểm toàn clip, interval `[0, duration]` chỉ thể hiện phạm vi đánh giá, chưa định vị thời điểm bạo lực. Audio_text lấy từ đoạn ASR bị đánh dấu, giới hạn interval trong clip; timestamp có thể sai lệch do ASR. Schema validate start ≥ 0 và end ≥ start.
- `model_version`: định danh artifact và ngưỡng dùng để tái hiện; `processing_seconds` là thời gian server xử lý sau khi framework nhận multipart, không phải toàn bộ thời gian upload.
- Thiếu audio: vẫn có thể đánh giá hình ảnh, nhưng ghi warning và text score = null nếu không có OCR thay thế. Không biến “không nghe thấy” thành điểm an toàn.

## Lỗi hiện có và phần còn thiếu

| HTTP | Hiện tại |
|---|---|
| 400 | File rỗng |
| 413 | Nội dung file > 50 MiB |
| 415 | Đuôi file ngoài whitelist |
| 422 | Thiếu trường file / request sai; ở model: media hỏng, không có video, không xác định duration, quá 60 giây hoặc độ phân giải quá giới hạn |
| 503 | Thiếu artifact/phụ thuộc model, model không nạp được hoặc cấu hình backend không hợp lệ |

**Đã có ở model:** kiểm tra container/thời lượng ≤60 giây, diện tích frame ≤3840×2160 và giải mã frame lấy mẫu, xử lý audio thiếu/không phải tiếng Việt/ASR lỗi với Toxic=null và warning. Model được cache, inference trong một tiến trình được tuần tự hóa. **Chưa có:** readiness endpoint, hàng đợi bền vững, retry, rate limit, timeout giải mã cứng. Mock chấp nhận byte bất kỳ mang đuôi hợp lệ để test giao tiếp; không đồng nghĩa video hợp lệ. Giới hạn 50 MiB trong endpoint áp dụng sau khi framework parse multipart; triển khai online cần giới hạn body ở ingress/proxy. Chi tiết cấu hình: `docs/VIDEO_PIPELINE.md`.

**Điểm bàn giao:** app owner giữ HTTP/upload/tempfile; ML lead triển khai `kidsafe.pipeline.analyze_video(path, backend="model") -> AnalysisResult`, dùng module trong `ml/`. Không đổi signature độc lập. Không được gọi mock khi model thật lỗi.
