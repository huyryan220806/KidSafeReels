# Giữ Violence v3, cải thiện Toxic theo ASR CPU

## Phạm vi và nhãn

Violence v3 được khóa checksum, giữ ngưỡng 0,41. Nhánh Toxic thử nghiệm riêng, chưa đổi model app. Giữ split speech200 theo nhóm nguồn: train142, validation28, test30. Đợt này chỉ dùng train/validation; test30 đã xem ở đợt trước không được mở để chọn model/ngưỡng hay công bố như test độc lập mới.

Nhãn clip vẫn là nhãn yếu do người cung cấp mô tả; không tự điền nhãn người đã nghe duyệt. Dữ liệu, transcript, phiếu nhãn và checkpoint nằm trong `runs/` được Git bỏ qua.

## Phiếu 10 clip cho Huy

Mở [phiếu local](http://127.0.0.1:8767/review-huy.html). Nếu máy chủ đã tắt, từ gốc repo chạy:

```powershell
.\.venv\Scripts\python.exe -m http.server 8767 --bind 127.0.0.1 --directory runs/toxic-precision-2026-10-29/review10
```

Cũng có thể mở trực tiếp `runs/toxic-precision-2026-10-29/review10/review-huy.html`. Gói dùng lại giao diện có lưu nháp, nạp/xuất JSON và kiểm bằng chứng. Video và HTML phải ở cùng cấu trúc thư mục.

- 10 clip, tổng 175 giây, từ10 nhóm nguồn:4train+6validation, không có test. Chọn5 ca nghi báo nhầm +5 ca nghi bỏ sót theo baseline và nhãn nguồn; đây là bất đồng cần nghe xác nhận, chưa phải10 lỗi đã được người xác minh.
- Phiếu ẩn nhãn nguồn, dự đoán và ASR. Nghe hết clip, xác nhận ngôn ngữ, chọn0/1/chưa rõ. Nhãn1 cần mốc thời gian và câu nghe được. Không nghe rõ thì không đoán nhãn0.
- Khuyến khích chép lời nghe được. Chỉ tick đã đối chiếu toàn bộ transcript nếu thực sự chép và đối chiếu toàn bộ; nếu chép một phần, ghi rõ trong ghi chú.
- Bấm **Tải JSON nhãn hiện tại**, gửi đường dẫn file cho Codex. Nhãn validation chỉ dùng đánh giá/chẩn đoán phát triển, không đưa vào fit. Mẫu được chọn theo lỗi không dùng tính Accuracy đại diện.

Kiểm cấu trúc file tải về (không phải tự xác minh nội dung nhãn):

```powershell
.\.venv\Scripts\python.exe -m ml.toxic_precision review-check --input "DUONG_DAN_JSON"
```

## Thí nghiệm đã khóa trước khi chạy

Protocol: `reports/toxic-precision-2026-10-29/protocol.json`. ASR Whisper small CPU INT8, beam5, VAD; đồng bộ kiểu tính toán với app, dùng segment được decoder chấp nhận. Cache mỗi clip để tiếp tục khi gián đoạn. Model PhoBERT khởi tạo từ v3, fine-tune hai block encoder cuối + classifier5epoch, MIL max logit cấp clip. Tăng trọng số loss clip bình thường lên2; không gán tất cả câu trong clip Toxic thành câu Toxic.

Chọn epoch/ngưỡng chỉ trên validation: hybrid FP không tăng, hybrid recall không giảm và model-only FP không tăng so với v3 trên cùng transcript CPU. Trong các cấu hình đạt điều kiện, ưu tiên ít hybrid FP rồi recall, log-loss và epoch sớm. Bộ luật v1 giữ checksum. Nếu không đạt điều kiện, giữ v3; dù đạt cũng lưu ứng viên riêng chờ nhãn người và regression rộng hơn.

Không dựng loss cho quyết định hybrid. Log-loss là xác suất model trên clip có ASR; các clip thiếu ASR giữ review/null và được tính riêng trong thống kê quyết định.

## Tiếp tục và tái tạo

Run đã prepare; **không prepare lại để ghi đè**. Trong hai cửa sổ terminal có thể chạy:

```powershell
.\.venv\Scripts\python.exe -m ml.toxic_precision asr --worker 0
.\.venv\Scripts\python.exe -m ml.toxic_precision asr --worker 1
```

Sau khi đủ170 cache train/validation:

```powershell
.\.venv-train\Scripts\python.exe -m ml.toxic_precision_train train
.\.venv\Scripts\python.exe -m ml.toxic_precision_train cpu-check
```

Train resume từ epoch cuối đã lưu optimizer/RNG; không chạy lại khi selection đã khóa. CPU check chỉ đánh giá28validation cùng transcript, không phải test mới. Checkpoint ứng viên trong `runs/toxic-precision-2026-10-29/toxic/selected/`; không thay `models/artifacts.json`. Bảng tổng hợp/đồ thị không có media hoặc transcript nằm trong `reports/toxic-precision-2026-10-29/`.
