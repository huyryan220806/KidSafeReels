# Bàn giao phần app cho Huy

Nhánh: `feat/app`. App có thể chạy mock ngay, không cài torch/ASR. Không thay đổi `kidsafe/`, `ml/`, checkpoint, nhãn, ngưỡng hoặc metric ML.

## Điểm nối giữ nguyên

```python
from pathlib import Path
from kidsafe.schemas import AnalysisResult

def analyze_video(path: Path, *, backend: str = "model") -> AnalysisResult:
    ...
```

`apps/api/worker.py` gọi hàm này sau khi video đã qua kiểm tra kỹ thuật. Worker dùng lại giữa các request; Huy có thể cache model ở cấp module/hàm. Worker là process `spawn` dạng daemon; inference CPU nên dùng DataLoader `num_workers=0`. Không tự tải checkpoint trong request. Restart API sau khi thay code/artifact/cấu hình; timeout sẽ kết thúc worker, lượt sau phải nạp model lại.

App đã giải mã video để kiểm tra, nhưng schema v1 chưa có chỗ truyền metadata/frame vào ML. Huy tiếp tục decode để lấy feature theo cấu hình của mình. Việc tối ưu chia sẻ decode là hạng mục cần thống nhất riêng, không đổi signature độc lập.

## Các kiểm tra app đã thêm

- Backend trả về phải đúng `mock` hoặc `model` đã yêu cầu.
- `evaluated_labels` khớp chính xác các score khác null; không lặp nhãn.
- Evidence phải thuộc nhãn đã đánh giá và nằm trong thời lượng clip, không có NaN/Infinity.
- Không có nhãn nào được đánh giá thì phải `review`.
- Không có audio track thì không được trả evidence `audio_text`; app thêm cảnh báo.
- Mock không được có score/evidence thật hoặc kết luận khác `review`.

Điểm sai contract trả 502; model chưa sẵn sàng trả 503; hết thời gian trả 504; exception ngoài dự kiến trả 500. Không fallback sang mock. App không sửa điểm/kết luận của model để qua validator.

Không có audio: Huy trả `toxic_language=null` nếu không có OCR thay thế. Nếu thiếu modality cần thiết, trả `review`. Có audio track không chứng minh có lời nói, ASR cần xử lý riêng.

## Phụ thuộc và contract cần review khi ghép PR

- Thêm `av>=19,<20` cho app và nâng Streamlit tối thiểu 1.65 để hỗ trợ kiểm thử uploader thật.
- `requirements-app.lock.txt` khóa môi trường app đã chạy Windows/Python 3.12. Cài phần app bằng `-r requirements-dev.txt -c requirements-app.lock.txt`.
- Huy nên tách `requirements-ml.txt`; thử tương thích với NumPy/PyAV/PyTorch của model trước khi hợp nhất lock. App lock không tuyên bố tương thích với stack ML chưa bàn giao.
- JSON/schema v1 không thay đổi. Hành vi upload thay đổi: byte giả đuôi `.mp4` bị từ chối ngay cả ở mock. Dùng fixture video thật trong test.
- Mã lỗi bổ sung 429/500/502/504 và media 422/415 đã ghi trong `API_CONTRACT.md`.

## Huy cần cung cấp để nghiệm thu model

1. Code nhánh model của pipeline, dependencies và cách lấy checkpoint hợp lệ.
2. Version/checksum, mapping nhãn, cấu hình preprocessing và ngưỡng đã chọn.
3. Ít nhất 5 video có quyền sử dụng để thử đầu-cuối; gồm có tiếng/không tiếng.
4. RAM/thời gian CPU đo thật để điều chỉnh timeout nếu cần.
5. Metrics và giới hạn thực; app không tạo thay.

```powershell
$env:KIDSAFE_BACKEND = 'model'
powershell -ExecutionPolicy Bypass -File scripts/run.ps1
```

Chỉ chạy lệnh trên sau khi cài stack ML và đặt artifact đúng vị trí. Trong trạng thái hiện tại, video hợp lệ sẽ nhận 503; đó là hành vi đúng.

## Phần app đã kiểm thử

Xem `APP_VALIDATION.md`. Test mode=model dùng fixture tổng hợp rõ nhãn trong `tests/worker_fixtures.py`, chỉ nằm trong test, không có chế độ trả kết quả giả trên app. Những ca này kiểm tra giao tiếp/hiển thị, không chứng minh độ chính xác model.
