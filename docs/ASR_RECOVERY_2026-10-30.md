# Bàn giao nhãn đã duyệt và sửa ASR lặp

App tiếp tục dùng **Toxic v3 + bộ luật, Violence v3**. Code mới thêm nhận dạng lại có điều kiện cho transcript lặp, hậu tố `asr-repeat-recovery-v1`. Không cần nhận ZIP model mới. Sau khi lấy code mới, dừng phiên cũ và chạy lại launcher để worker nạp code.

Mở web tại http://127.0.0.1:8501. Nếu chưa chạy, nhấp đúp `Chay-KidSafeReels.cmd`, hoặc tại gốc repo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run-model.ps1
```

Kiểm phần `model_version` trong JSON tải từ app có `asr-repeat-recovery-v1`. Clip có retry sẽ có cảnh báo giải thích ASR đã nhận dạng lại. Điểm Toxic vẫn là điểm PhoBERT thật, có thể thấp dù bộ luật cảnh báo; không suy quyết định chỉ từ điểm này. Lượt retry cần thêm thời gian CPU, chỉ chạy khi câu dài bị lặp nhiều lần.

## Dữ liệu đã lưu

Thư mục local `runs/toxic-reviewed-2026-10-29/`:

- `annotations-original.json`: file Huy tải từ Chrome, giữ nguyên.
- `chat-adjudications.json`: hai xác nhận của Huy, có nguồn.
- `annotations-resolved.json`:10nhãn đã giải quyết,5dương/5âm; bằng chứng cũ của câu được xác nhận âm vẫn lưu dưới dạng ngữ cảnh.
- `manifest.json`:4train/6validation cùng nhóm nguồn/split đã khóa.
- `asr/`, `recovery/`, `http/`: cache thí nghiệm và trace thật; không Git.
- `CONTINUE.md`: trạng thái cuối và việc tiếp theo, đọc trước khi chạy lại.

Phiếu cũ chưa tự đổi theo xác nhận trong chat. Muốn xem bản đã giải quyết trên giao diện, dùng **Nạp lại JSON** với `annotations-resolved.json`; không cần làm lại10clip. File tải gốc được bảo toàn dù bản giải quyết khác hai trường nhãn.

## Tái tạo

Các stage của run hiện tại đã khóa, không chạy lại để dò ngưỡng hoặc đổi selection. `ml.speech_review_followup` hỗ trợ `prepare`, `train-dev`, `valid-dev`, `recovery-audit` và `summarize`. Chuẩn bị một run mới trong code khi thiết kế thí nghiệm khác; không ghi đè protocol cũ.

`prepare --input FILE --corrections FILE` kiểm manifest/IDs/bằng chứng và lưu bản gốc; chỉ chấp nhận sửa nhãn có nguồn xác nhận người dùng. `train-dev` là thí nghiệm ASR trên phần train, **không huấn luyện trọng số**. `valid-dev` chỉ dùng cấu hình được chọn từ train. `summarize` dựng lại bảng/hình từ kết quả đã lưu.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ml.release_verify
```

264tests pass trước bàn giao;17artifact CPUv3 giữ checksum. Xem [báo cáo và giới hạn](../reports/TOXIC_REVIEW_ASR_2026-10-30.md). Không đánh đồng10clip chọn theo lỗi với test đại diện. Ứng viên Toxic precision chưa kích hoạt vì Recall trên ViCTSD validation giảm; việc giảm FP classifier vẫn đang mở.
