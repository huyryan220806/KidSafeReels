# Toxic từ 200 nhãn Huy — 13/11/2026

Đã nhận `annotations-huy-toxic-full200-2026-11-05.json`: 200/200 nhãn, 102 Toxic và 98 bình thường. Tất cả xác nhận nghe đủ và tiếng Việt; chưa có transcript đầy đủ được đối chiếu nên không tính WER/CER. Đây là nhãn một người duyệt, chưa phải bộ nhãn độc lập đã phân xử giữa hai người.

Run mới: `runs/toxic-human-2026-11-13/` (private), kết quả tổng hợp: `reports/toxic-human-2026-11-13/`. File gốc được sao nguyên byte; manifest mới giữ nhãn nguồn cũ cạnh nhãn Huy để truy vết. Không ghi đè các run tháng 10.

**Trạng thái cuối: train 5 epoch và evaluate CPU đã hoàn tất.** Chọn epoch 2 / ngưỡng 0,80; validation hybrid F1 82,76% → 86,67%, nhưng test cũ F1 giữ 64,52%, FP5/FN6 và không đổi quyết định nào. Giữ v3 trong app, ứng viên chưa phát hành. 280 tests pass. [Báo cáo, Accuracy/Loss, CSV và đồ thị](../reports/TOXIC_HUMAN_2026-11-13.md). Không chạy lại các stage đã khóa bên dưới để dò ngưỡng.

## Thí nghiệm

- Split speech200 giữ 142 train / 28 validation / 30 test theo nhóm nguồn. Chỉ train/validation được dùng trước khi khóa checkpoint/ngưỡng. 30 test đã được đánh giá trong đợt trước, lần này chỉ là holdout cũ được chấm lại với nhãn người; không công bố như test độc lập mới.
- Dùng transcript CPU Whisper small/beam5 hiện có, có phục hồi lặp đã triển khai. Các lời bằng chứng ngắn không được coi là transcript đầy đủ hoặc tự gán nhãn cho mọi câu trong clip Toxic.
- Khởi tạo PhoBERT v3, cập nhật hai encoder block cuối và classifier, 5 epoch. Loss gồm BCE cấp clip (max logit các đoạn ASR) và 0,5 lần CE trên bình luận ViCTSD **train** được lấy mẫu cân bằng 4 âm + 4 dương mỗi bước. Mục đích kiểm tra liệu học lại bình luận có giữ Recall trong lúc thích nghi lời nói hay không; không mặc định sẽ cải thiện.
- Chọn epoch/ngưỡng trên validation lời nói và ViCTSD validation: hybrid FP/FN không tăng; model-only F1 lời nói không giảm; Recall/F1 bình luận không giảm so với v3. Báo riêng quét ngưỡng v3 để phân biệt tác dụng train với chỉ đổi ngưỡng.
- Nếu không có ứng viên đạt, giữ v3 và báo nguyên kết quả. Ứng viên được lưu riêng trước kiểm CPU; không sửa Violence v3, bộ luật hoặc schema API.

## Lệnh và tiếp tục khi gián đoạn

Đã prepare run; không chạy lại prepare để ghi đè nhãn/protocol. Mỗi lệnh chỉ chạy một instance:

```powershell
.\.venv-train\Scripts\python.exe -X utf8 -u -m ml.toxic_reviewed_train train
.\.venv\Scripts\python.exe -X utf8 -u -m ml.toxic_reviewed_train evaluate
```

Train lưu `resume.pt` sau từng epoch cùng optimizer/RNG, kiểm protocol trước resume. Khi có `selection.json`, không train lại cùng run. Evaluate chỉ chạy sau selection; lưu cache CPU từng clip và có thể tiếp tục phần thiếu. Khi có `evaluation.json`, không mở test lại để dò ngưỡng.

Mốc tiếp tục: `runs/toxic-human-2026-11-13/CONTINUE.md`. `progress.json` báo epoch; `selection.json` ghi checkpoint/ngưỡng đã khóa; `evaluation.json` ghi kết quả CPU thật. Đồ thị history dùng ngưỡng 0,5 cố định, khác bảng ở ngưỡng chọn. Model-only log-loss chỉ trên clip có ASR; clip thiếu ASR được báo riêng, không giả điểm 0. Quyết định hybrid không có xác suất hay loss gộp.

Nhãn/transcript/media/checkpoint/dự đoán từng clip giữ private. CSV/hình công khai chỉ chứa tổng hợp. Chỉ cập nhật app sau khi có bằng chứng đủ và kiểm tra artifact; chạy train không tự thay model đang dùng.
