# Kế hoạch cải thiện Toxic và Violence

Cập nhật 20/10/2026: **đã hoàn tất đợt thực nghiệm benchmark, CSV/đồ thị và tích hợp CPU v2**. Nâng ASR small beam5; giữ hai classifier v1 vì ứng viên không cải thiện đủ trên test. Xem [báo cáo thực nghiệm](../reports/MODEL_IMPROVEMENT_2026-10-20.md), [progress](PROGRESS.md) và [hướng dẫn tái lập](MODEL_EXPERIMENTS.md). Huy xác nhận chưa lưu các video báo lỗi, tiếp tục với dữ liệu benchmark hiện có. Các mục cần clip lỗi/nhãn người bên dưới vẫn chưa hoàn tất.

## Phản hồi cần xử lý

1. **Toxic cảnh báo nhầm:** một số video nói chuyện bình thường vẫn bị đánh dấu độc hại. Huy nhận thấy transcript sai, nhận dạng từ chưa đúng. Cần kiểm chứng riêng lỗi ASR và lỗi classifier; chưa kết luận mọi cảnh báo nhầm đều do ASR. Các video cụ thể Huy vừa thử chưa được lưu thành bộ ca lỗi có nhãn.
2. **Violence bỏ sót:** có video bạo lực rõ nhưng model không đánh dấu. Cần đo false negative và phân tích đoạn bạo lực ngắn, lấy mẫu frame và khác biệt so với dữ liệu huấn luyện.
3. **Phạm vi:** chỉ tập trung `toxic_language` và `violence`. Hoãn Sexual/Dangerous khỏi đợt cải thiện này do thời gian ngắn; giữ trạng thái chưa đánh giá và score `null` theo contract.
4. **Đầu ra bắt buộc:** kết quả train, validation và test; Accuracy, Loss và các chỉ số phù hợp; đồ thị và CSV để thấy model đang ở mức nào, so sánh trước/sau bằng số đo thật.

## Mốc đối chiếu đã có

Các số dưới đây thuộc baseline ngày 16/10, không phải kết quả của đợt cải thiện mới.

| Model / tập test | Số mẫu test | Accuracy | Precision lớp dương | Recall lớp dương | F1 lớp dương | FP / FN |
|---|---:|---:|---:|---:|---:|---:|
| Toxic TF-IDF + Logistic Regression / ViCTSD | 988 | 0,8684 | 0,4173 | 0,4862 | 0,4492 | 74 / 56 |
| Violence ResNet18 frozen + Logistic Regression / RLVS | 386 | 0,9585 | 0,9550 | 0,9646 | 0,9598 | 9 / 7 |

Nguồn: [Toxic baseline](../reports/TOXIC_BASELINE.md), [Violence baseline](../reports/VISUAL_BASELINE.md) và hai JSON metrics tương ứng trong `reports/`.

- Toxic có accuracy thấp hơn bộ luôn đoán Non-toxic (0,8897); phải xem Precision/Recall/F1, không chỉ accuracy.
- Điểm RLVS cao chưa chứng minh model tốt trên video người dùng; baseline dùng tối đa 8 frame, chưa mô hình hóa chuyển động và chưa định vị cảnh.
- ASR hiện dùng Whisper base CPU/int8, `beam_size=1`, VAD, tắt nối ngữ cảnh trước; `ml/inference.py` gọi `transcribe`. Toxic lấy điểm cao nhất theo đoạn ASR và ngưỡng vốn chọn trên bình luận. Cần kiểm tra từng khâu bằng dữ liệu, không mặc định chỉ sửa một từ hay tăng ngưỡng là giải quyết được.
- Chưa có loss hoặc lịch sử train từng epoch được ghi trong hai report cũ. Chưa có metric toàn pipeline trên video ngắn tiếng Việt có nhãn; 25 clip pilot vẫn chờ nhãn người thật.

## Thứ tự thực hiện khi tiếp tục

### 1. Chuẩn bị đo và tập ca lỗi

- [x] Kiểm tra checkpoint, cache feature, split và code hiện có để tái sử dụng, tránh tải/train lại không cần thiết.
- [ ] Lưu các clip người dùng báo lỗi vào tập phân tích riêng khi có quyền sử dụng; ghi source/group, nhãn người thật và mốc lỗi. Clip đã xem dự đoán dùng cho phát triển/regression, không trở thành test giữ riêng mới.
- [ ] Với Toxic, đối chiếu audio, transcript ASR và transcript được người nghe sửa; so classifier trên cả hai để tách lỗi ASR khỏi lỗi phân loại. Nếu có transcript tham chiếu, đo WER/CER và ghi cách chuẩn hóa tiếng Việt.
- [ ] Với Violence, đối chiếu khung hình đã lấy mẫu với mốc bạo lực do người xem ghi; phân biệt không lấy trúng cảnh và lấy trúng nhưng classifier sai.

### 2. Cải thiện hai nhánh trên train/validation

- [x] Toxic: kiểm tra nhận dạng tiếng Việt, cấu hình giải mã/VAD, ngữ cảnh câu và cách gộp đoạn. Thử cấu hình có kiểm soát để giảm cảnh báo nhầm mà vẫn theo dõi recall Toxic; đo chi phí CPU.
- [x] Toxic classifier: so baseline với các phương án cải thiện phù hợp CPU/Colab, chọn trên validation; ghi tác động riêng của ASR và classifier trước khi ghép.
- [x] Violence: thử lấy mẫu dày hơn hoặc theo cửa sổ thời gian cho cảnh ngắn; xem xét đặc trưng chuyển động/model khác nếu chi phí cho phép. Khi đổi feature, bảo đảm train và inference dùng cùng tiền xử lý.
- [x] Ghi run ID, seed, cấu hình, phiên bản dữ liệu/model, thời gian train/inference và tiêu chí chọn cấu hình/ngưỡng trước khi đánh giá test cuối.

### 3. Kết quả train/test, CSV và đồ thị

- [x] Xuất cho từng model và split: số mẫu từng lớp, accuracy, balanced accuracy, precision/recall/F1 từng lớp, macro-F1, log loss (binary cross-entropy từ score xác suất), confusion matrix, ROC-AUC và average precision khi split đủ hai lớp. Chỉ số không xác định để trống kèm lý do, không điền 0 thay thế.
- [x] CSV tổng hợp: `metrics_by_split.csv`, `comparison.csv`, `confusion_matrix.csv`; có run ID, dataset/split, model, threshold, support và tên chỉ số rõ ràng.
- [x] CSV dự đoán từng mẫu: sample/group ID, split, nhãn thật, score, dự đoán, lỗi FP/FN. Giữ ở `runs/`; không đưa transcript/video/nhãn người dùng lên Git. Bảng tổng hợp và hình không chứa nội dung riêng tư nằm trong `reports/`.
- [x] Đồ thị PNG/SVG: confusion matrix, so sánh Accuracy/F1/Precision/Recall và Loss giữa train/validation/test; PR curve; ROC curve; đồ thị ảnh hưởng ngưỡng trên validation; so sánh baseline với ứng viên mới.
- [x] Ghi loss cuối và `learning_curve.csv` theo số mẫu train. Không có lịch sử epoch cho Logistic Regression; không dựng đường epoch hoặc dùng `max_iter` như số epoch đã đo.
- [x] Tóm tắt điểm mạnh/yếu, FP/FN, ảnh hưởng ASR và đánh đổi latency trên benchmark. Phản hồi dùng app vẫn chưa kiểm chứng theo từng clip vì chưa lưu ca lỗi.

### 4. Đánh giá test cuối và đưa vào app

- [x] Giữ split theo nhóm video hiện có và dữ liệu test khỏi vòng chọn model/ngưỡng; ghi giới hạn nhóm nguồn dHash. Không điều chỉnh bằng test cũ đã xem rồi gọi đó là một đánh giá độc lập mới; ghi rõ test benchmark đã từng được báo cáo.
- [ ] Có nhãn người thật trước khi báo metric video/pilot. Nếu cần thêm tập đích, chốt nhóm/split trước, dành phần chưa dùng cho đánh giá cuối và báo support thực tế.
- [x] Chốt cấu hình bằng validation, đánh giá test của cấu hình đã chọn; so trên cùng mẫu/định nghĩa và tách metric text, video benchmark, toàn pipeline.
- [x] Chỉ thay bundle khi có bằng chứng cải thiện phù hợp mục tiêu: nâng ASR, giữ classifier v1; lưu baseline để quay lại, cập nhật checksum/version, chạy regression API/UI và ghi progress.

## Điểm tiếp tục

Đọc trạng thái mới nhất ở [PROGRESS.md](PROGRESS.md) và [báo cáo](../reports/MODEL_IMPROVEMENT_2026-10-20.md) trước khi chạy lại. Không lặp vòng chọn cấu hình hoặc đánh giá test của run đã hoàn tất. Cache và checkpoint tiếp tục nằm local tại `runs/model-improvement-2026-10-20/`.

Các mục ca lỗi người dùng và nhãn pilot vẫn chờ người thực hiện. Đã đo WER/CER bằng transcript tham chiếu FLEURS và audit riêng tác động lên Toxic; chưa có transcript được người sửa cho các clip lỗi của Huy hoặc mốc bạo lực do người xem gán. Không đánh dấu hai việc đó hoàn tất bằng kết quả benchmark.
