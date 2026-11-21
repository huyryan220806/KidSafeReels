# Tiếp tục Toxic — 15/11/2026

**Đã hoàn tất.** Thử thêm ngữ cảnh ASR trên 170 clip train/validation, giữ nguyên hai checkpoint/ngưỡng, không train thêm và không chấm test. Không có ứng viên đạt điều kiện: FP validation giảm 2→1 nhưng so với checkpoint 13/11 thì FN tăng 2→3. App giữ v3, Violence v3. [Báo cáo, CSV và đồ thị](../reports/TOXIC_CONTEXT_2026-11-15.md).

## Mã và dữ liệu

- `ml/toxic_context.py`: khóa nguồn, tạo cửa sổ giữ đoạn giữa, suy luận 2 checkpoint × 2 cách đọc; kiểm so với cache CPU cũ, tính metric và selection.
- `ml/toxic_dev_audit.py`: phân loại lỗi trên dev từ nhãn Huy và ASR; không tạo transcript hoặc thay nhãn.
- `tests/test_toxic_context.py`: kiểm giữ token đoạn giữa, split và điều kiện chọn ứng viên.
- `runs/toxic-context-2026-11-15/`: manifest chỉ gồm dev, hash đầu vào, cache từng clip, log, bảng lỗi chi tiết, `CONTINUE.md` và `continuation-state.json`. Dữ liệu này private.
- `reports/toxic-context-2026-11-15/`: protocol, CSV tổng hợp, selection, evaluation, PNG/SVG, verification.

Môi trường `.venv-train` dùng CUDA cho thí nghiệm; `.venv` dùng kiểm thử và audit. Cả hai đã có trên máy Huy. Clone GitHub không tự có nhãn/video/checkpoint; không gửi raw qua Git.

## Lệnh tái hiện quy trình

Các lệnh dưới mô tả thứ tự đã chạy, **không chạy lại prepare/score trên run hoàn tất**:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m ml.toxic_context prepare
.\.venv-train\Scripts\python.exe -X utf8 -m ml.toxic_context score
.\.venv\Scripts\python.exe -X utf8 -m ml.toxic_dev_audit
.\.venv\Scripts\python.exe -X utf8 -m pytest -q
.\.venv\Scripts\python.exe -X utf8 -m ml.release_verify
```

Nếu một run chưa hoàn tất bị ngắt, xác nhận không còn tiến trình rồi chạy lại `score`; cache từng clip có protocol hash. Khi đã có `evaluation.json`, code chặn chấm lại. Muốn thử thuật toán khác cần run/protocol riêng, không sửa đầu vào đã khóa hoặc dùng test để chọn cấu hình.

## Việc còn lại

1. Ưu tiên ASR trên các ca dev thiếu lời nói/lặp trong một đoạn; dùng mốc bằng chứng Huy đã ghi để đối chiếu, không tự tạo nhãn hoặc transcript chuẩn.
2. Cải thiện classifier trên câu bình thường. Ngữ cảnh có giảm FP nhưng chưa đủ giữ Recall; chưa đưa vào runtime.
3. Sau khi chọn ứng viên đủ tốt trên dev, kiểm CPU và pipeline trước phát hành; có tập mới chưa dùng để phát triển thì đánh giá độc lập. Không coi validation đã thử nhiều lần hoặc 30 test cũ là test mới.

284 tests pass; 17 artifact active đúng checksum. Không có job train/evaluate còn chạy sau hoàn tất. Nhãn 200 đã nhận đủ; không yêu cầu Huy gán lại. Đợt train 5 epoch và đánh giá CPU trước đó được lưu riêng tại [13/11](TOXIC_HUMAN_2026-11-13.md).
