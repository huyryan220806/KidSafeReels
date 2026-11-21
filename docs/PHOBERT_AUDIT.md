# Chạy audit PhoBERT độc lập

Audit ngày 15/11 đã hoàn tất, không cần train lại để xem báo cáo. Mở [kết luận và bằng chứng](../reports/PHOBERT_AUDIT_2026-11-15.md), [báo cáo tự sinh + R0–R7](../reports/phobert-audit-2026-11-15/AUDIT.md).

## Chạy trên máy Huy

Từ thư mục repo, muốn tái hiện một lượt mới, dùng **run ID khác** để không ghi đè audit đã có:

```powershell
.\.venv-train\Scripts\python.exe -X utf8 -m ml.phobert_audit all --device cuda --run-id phobert-audit-reproduce-01
.\.venv-train\Scripts\python.exe -X utf8 -m ml.phobert_audit_probe --run-id phobert-audit-reproduce-01
```

`all` kiểm dữ liệu, checkpoint gốc, tái chạy đúng hàm train 5 epoch có recorder, rồi tạo CSV/JSON/Markdown/đồ thị. Phần replay cần môi trường CUDA hiện có; giữ thêm khoảng 5 GB checkpoint/artifact private. Không tải model qua mạng, không train ASR, không sửa dữ liệu và model active.

Có thể chạy từng stage:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m ml.phobert_audit data --run-id phobert-audit-reproduce-02
.\.venv-train\Scripts\python.exe -X utf8 -m ml.phobert_audit checkpoints --device cuda --run-id phobert-audit-reproduce-02
.\.venv-train\Scripts\python.exe -X utf8 -m ml.phobert_audit replay --run-id phobert-audit-reproduce-02
.\.venv\Scripts\python.exe -X utf8 -m ml.phobert_audit report --run-id phobert-audit-reproduce-02
.\.venv\Scripts\python.exe -X utf8 -m ml.phobert_audit_probe --run-id phobert-audit-reproduce-02
```

`checkpoints --device cpu` cũng được hỗ trợ, nhưng CPU/GPU có thể lệch số nhỏ; kiểm so sánh trọng số vẫn chính xác. Lượt đã ghi nhận dùng CUDA FP32 khi export, CUDA FP16 khi train theo trainer gốc. `report`/probe chỉ đọc artifact, không chạy test hoặc train lại.

Nguồn mặc định là run đã khóa `toxic-human-2026-11-13`. Script kiểm hash source, ASR, nhãn, code và checkpoint; nếu nguồn đổi, dừng để tránh trộn kết quả. Máy clone repo chưa có private dataset/model sẽ không chạy đủ audit; cần nhận gói riêng từ Huy, không đưa raw lên Git.

## Đầu ra và cách đọc

- `reports/<run-id>/AUDIT.md`: báo cáo tự sinh, label counts, overlap, metrics và kế hoạch R0–R7.
- `validation-metrics.csv`: binary Toxic P/R/F1, macro-F1, weighted-F1, loss, AP và TN/FP/FN/TP tại 0,5.
- `probability-drift.csv`: so sánh clip bằng ID; `segment-drift.csv` và `threshold-margins.csv` do probe tạo.
- `runs/<run-id>/validation/<snapshot>/validation.npz`: logits `[162,2]`, probability 2 lớp, prediction, segment_id/video_id, label video và sentinel `-1` cho nhãn đoạn chưa biết.
- `segments.csv` không giả nhãn đoạn: `segment_label` trống; `video_label` riêng. `clips.csv` là 28 mẫu có nhãn dùng cho metrics. CSV raw và checkpoints chỉ ở `runs/` đã gitignore.
- `original_epoch0/2/5`: export mới từ artifact lịch sử có sẵn. `replay_epoch0..5` và `replay_selected`: bản tái chạy, không gọi là logits lịch sử gốc.
- `replay-comparison.json`: đối chiếu điểm và từng tensor ở checkpoint gốc còn giữ. `history-comparison.csv`: đối chiếu loss/metrics lịch sử qua 5 epoch.
- `preprocessing-issues.json`, `oov-details.json`, `timing-issues.json`, `duplicate-details.json`: chi tiết private để nghe/đọc đối chiếu.

Test chỉ được đọc cho số lượng/nhãn/hash/trùng ASR; **không có inference test, test loss, test F1 hoặc lựa chọn ngưỡng từ test**. Stage replay chỉ đưa train/valid vào trainer. Script không sửa pipeline production; các path/hook được chuyển riêng trong tiến trình audit rồi khôi phục.

## Tiếp tục khi gián đoạn

Kiểm `runs/<run-id>/replay-reports/progress.json`, log terminal và tiến trình Python trước khi chạy lại. Nếu trainer chưa hoàn tất, lệnh `replay` tiếp tục từ `resume.pt` đã lưu sau epoch, giữ các snapshot đã hoàn tất. Nếu đã có `replay-reports/selection.json`, dùng `report` để hoàn tất đối chiếu; không train lại run đó.

Nếu bị ngắt ngay lúc recorder đang ghi snapshot nhưng chưa lưu resume cùng epoch, giữ nguyên run lỗi và dùng run ID mới; không xóa artifact để giả một lượt liên tục. Audit sẽ từ chối ghi đè snapshot đã có.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest -q tests/test_phobert_audit.py tests/test_experiment_metrics.py tests/test_toxic_reviewed.py
.\.venv\Scripts\python.exe -X utf8 -m pytest -q
```

Tiến độ audit đã lưu ở `runs/phobert-audit-2026-11-15/CONTINUE.md`. 200 nhãn Huy giữ nguyên; speaker ID và reference transcript vẫn là phần thiếu dữ liệu, không tự sinh từ ASR.
