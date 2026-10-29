# Hợp đồng API v1.0

Schema dùng chung giữ nguyên tại `kidsafe/schemas.py`; app không đổi tên trường hay signature của pipeline. Nhánh app bổ sung kiểm tra đầu vào và kiểm tra ngữ nghĩa kết quả ở `apps/validation.py`.

## Endpoints

- `GET /health` → `{"status":"ok","backend":"mock"}`. Chỉ xác nhận API sống và cấu hình backend, không xác nhận model đã nạp/checkpoint sẵn sàng.
- `POST /v1/analyze`: multipart/form-data, trường `file`. Chỉ nhận MP4/MOV/WebM, không rỗng, tối đa 50 MiB và 60 giây. Trả đồng bộ JSON `AnalysisResult`.
- `GET /docs`: OpenAPI tương tác.

```powershell
curl.exe -F "file=@D:\KidSafeReels\data\private\app-demo\01-khong-am-thanh.mp4" http://127.0.0.1:8000/v1/analyze
```

Ví dụ JSON minh họa hợp đồng (ID/thời gian chỉ là ví dụ, không phải kết quả đo):

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
  "warnings": ["MÔ PHỎNG: video đã được kiểm tra kỹ thuật; chưa phân tích rủi ro nội dung."],
  "processing_seconds": 0.01
}
```

## Kiểm tra đầu vào và xử lý

1. Kiểm tra đuôi file, copy theo khối 1 MiB vào thư mục tạm; không dùng tên upload làm đường dẫn lưu trữ.
2. Worker dùng PyAV kiểm tra container khớp đuôi, có đúng một video stream, giới hạn 8.294.400 pixel/khung và 18.000 khung.
3. Kiểm tra thời lượng từ metadata và mốc thời gian khung đã giải mã; tối đa 60 giây, dung sai làm tròn 0,001 giây. Giải mã luồng video, từ chối lỗi decoder/khung hỏng/thiếu khung khi có frame count khai báo. Không xác nhận chất lượng audio hoặc ASR tại bước này.
4. Gọi pipeline đúng backend đã cấu hình; kiểm tra kết quả trước khi trả.
5. Xóa file và thư mục tạm trên cả thành công/lỗi. Với timeout, dừng worker trước khi dọn file.

Mock cũng phải nhận video thật hợp lệ. Những test cũ dùng byte tùy ý mang đuôi `.mp4` phải đổi sang fixture video tổng hợp.

Worker dùng lại trong một tiến trình API để cache model; một lượt xử lý tại một thời điểm, không có hàng đợi. Chạy một uvicorn worker cho baseline CPU. Timeout mặc định: media 30 giây, pipeline 180 giây; cấu hình `KIDSAFE_MEDIA_TIMEOUT`/`KIDSAFE_MODEL_TIMEOUT` trong khoảng (0,600]. Thời gian khởi động worker/import PyAV thuộc giai đoạn media. Lượt sau timeout tạo worker mới.

## Ngữ nghĩa kết quả

- `scores`: điểm [0,1]; `null` = chưa đánh giá, `0` = đã đánh giá và trả điểm 0. Không gọi là xác suất nếu chưa calibration.
- `evaluated_labels`: đúng tập nhãn có điểm khác null, không trùng lặp; app kiểm tra bất biến này.
- `decision`: `flagged` khi vượt ngưỡng do ML chọn hoặc bộ luật Toxic trong pipeline tạo cảnh báo; `review` khi bất định/thiếu đầu vào cần thiết; `no_risk_detected` khi các kiểm tra trong phạm vi hoàn tất và dưới ngưỡng. Không có nhãn được đánh giá thì bắt buộc `review`. App không chọn hoặc đổi ngưỡng/quyết định.
- Model CPU v1/v2/v3 chỉ trả `flagged` hoặc `review`; chưa trả `no_risk_detected` trước đánh giá pilot có nhãn. Thiếu Toxic vẫn có thể `flagged` nếu Violence vượt ngưỡng. V3 thay classifier nội bộ bằng PhoBERT/R3D-18, giữ nguyên mọi trường schema v1 và evidence visual ở cấp toàn clip.
- `mode` phải khớp backend yêu cầu. Mock luôn `review`, mọi điểm null, evidence rỗng. Không chuyển sang mock khi model lỗi.
- `evidence`: `{modality, start_seconds, end_seconds, label, score, description}`; modality `visual/audio_text/ocr`. Timestamp hữu hạn, không âm, end ≥ start và không vượt thời lượng clip. Nhãn evidence phải được đánh giá. Bằng chứng phải do ML tạo từ frame/transcript thật.
- Không có track audio: app thêm warning, từ chối `audio_text` evidence. Huy trả text score null nếu không có nguồn OCR thay thế; quyết định `review` nếu chưa có nhãn khác vượt ngưỡng. Có track audio không đồng nghĩa có lời nói.
- `model_version`: định danh artifact + cấu hình/ngưỡng; cần tái hiện được. App hiển thị nguyên giá trị.
- Từ bản sửa ASR ngày 28/10, kết quả có hậu tố `+asr-native-filter-v2`: pipeline dùng segment không rỗng được decoder chấp nhận, không lọc lại chỉ bằng `no_speech_prob`; vẫn kiểm tiếng Việt/VAD. Không đổi trường hoặc trọng số/ngưỡng. Xem [báo cáo sửa lỗi](../reports/ASR_FILTER_FIX_2026-10-28.md).
- Từ30/10, cấu hình small/beam5/temperature0 thêm `+asr-repeat-recovery-v1`: sau gate tiếng Việt, chỉ nhận dạng lại theo cửa sổ khi transcript lặp bất thường. Retry không rỗng và hết lặp thì dùng transcript/mốc mới; lỗi hoặc vẫn lặp thì giữ bản cũ, thêm warning. Không chọn theo điểm Toxic, không sửa trọng số/ngưỡng hoặc schema. [Điều kiện và giới hạn](ASR_RECOVERY_2026-10-30.md).
- Từ 29/10, mặc định Toxic hybrid thêm hậu tố `+toxic-hybrid-v1-<12 ký tự SHA lexicon>`. `scores.toxic_language` giữ max điểm PhoBERT, `evidence.score` giữ điểm PhoBERT của đoạn tương ứng. Luật có thể tạo `flagged` khi điểm model thấp: không suy lại quyết định từ score trên app. `evidence.description` ghi nguồn model/bộ luật/gợi ý nghe lại và đoạn ASR; gợi ý review không tự cảnh báo. Thông báo UI/TXT giải thích rõ ý nghĩa điểm. Không thêm trường hoặc đổi schema 1.0; [quy tắc, giới hạn và mode baseline](TOXIC_HYBRID.md).
- `processing_seconds`: thời gian endpoint xử lý sau khi framework nhận multipart, gồm kiểm tra media/pipeline. UI đo riêng thời gian từ lúc gọi HTTP đến lúc nhận và validate kết quả.

## Mã lỗi

Lỗi nghiệp vụ có dạng `{"detail":"Thông báo tiếng Việt"}`. Lỗi validation do framework có thể trả `detail` dạng danh sách; UI dùng thông báo phù hợp, không in raw JSON/HTML.

| HTTP | Ý nghĩa |
|---|---|
| 400 | File rỗng |
| 413 | Nội dung file >50 MiB; proxy có thể từ chối toàn body >52 MiB |
| 415 | Đuôi hoặc container không được hỗ trợ/không khớp |
| 422 | Thiếu file, file hỏng, không có video, quá 60 giây hoặc giới hạn media khác |
| 429 | Worker đang bận; header Retry-After: 5; client không tự retry |
| 500 | Pipeline lỗi ngoài dự kiến hoặc worker crash; thông báo chung, không lộ stack trace |
| 502 | Kết quả pipeline sai schema/ngữ nghĩa hoặc evidence ngoài clip |
| 503 | Model chưa sẵn sàng hoặc cấu hình backend/timeout không hợp lệ |
| 504 | Quá thời gian media hoặc pipeline; worker đã được dừng |

Uvicorn có thể trả 503 khi chạm giới hạn số kết nối cấu hình. Giới hạn file 50 MiB trong endpoint áp dụng sau multipart parsing. Compose/nginx chặn body 52 MiB ở trước API và web; bản local uvicorn trực tiếp không thay thế ingress cho triển khai Internet.

## Bàn giao với ML

Giữ `kidsafe.pipeline.analyze_video(path: Path, *, backend="model") -> AnalysisResult`. App sở hữu HTTP/upload/tempfile/worker; Huy sở hữu pipeline, ASR, classifier, ngưỡng và artifact. Schema v1 không nhận metadata media; Huy vẫn tự decode để lấy feature. Muốn tái sử dụng phần decode cần thống nhất contract riêng trước khi đổi. Xem `HANDOFF_ML_APP.md`.
