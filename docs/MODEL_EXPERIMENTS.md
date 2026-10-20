# Chạy và đọc thực nghiệm cải thiện model

Run ngày 20/10/2026: `model-improvement-2026-10-20`. Phạm vi chỉ Toxic, Violence và ASR tiếng Việt. Kết quả nằm tại `reports/model-improvement-2026-10-20/`; dữ liệu từng mẫu và checkpoint ứng viên nằm trong `runs/` (không lên Git).

## Quy trình tái lập

Máy đã có dữ liệu/checkpoint từ baseline. Cài thư viện đồ thị một lần:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-evaluation.txt -c requirements-evaluation.lock.txt
```

Môi trường model giữ bộ phụ thuộc bàn giao trong `docs/VIDEO_PIPELINE.md`, gồm PyAV 16.1.0 và PyTorch CPU. Mỗi lượt thí nghiệm dùng run ID mới. Ví dụ:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m ml.model_experiments select --task toxic --run-id my-run-01
.\.venv\Scripts\python.exe -X utf8 -m ml.model_experiments evaluate --task toxic --run-id my-run-01
.\.venv\Scripts\python.exe -X utf8 -m ml.model_experiments select --task violence --run-id my-run-01
.\.venv\Scripts\python.exe -X utf8 -m ml.model_experiments evaluate --task violence --run-id my-run-01
.\.venv\Scripts\python.exe -X utf8 -m ml.asr_experiments --run-id my-run-01
.\.venv\Scripts\python.exe -X utf8 -m ml.asr_toxic_audit --run-id my-run-01
.\.venv\Scripts\python.exe -X utf8 -m ml.experiment_diagnostics --task toxic --run-id my-run-01
.\.venv\Scripts\python.exe -X utf8 -m ml.experiment_diagnostics --task violence --run-id my-run-01
.\.venv\Scripts\python.exe -X utf8 -m ml.summarize_experiments --run-id my-run-01
```

`select` ghi protocol trước khi fit, chỉ dùng train/validation để chọn cấu hình và ngưỡng; `evaluate` kiểm checksum cấu hình đã khóa rồi mới dự đoán test. Không ghi đè checkpoint v1 hoặc report baseline cũ. Lệnh từ chối đánh giá test lại trên run đã hoàn thành. Cache feature dense và ASR cho phép tiếp tục phần tính toán bị gián đoạn.

Để vẽ lại hình từ số đo đã lưu mà không fit hay dự đoán test lần nữa: `python -m ml.model_experiments render --task toxic --run-id my-run-01` (tương tự cho Violence). `experiment_diagnostics` tạo `error_review.csv` ở thư mục run private để xem text/đường dẫn video và FP/FN; cột ghi chú để trống cho người xem.

ASR tải 16 câu dev và 16 câu test FLEURS tiếng Việt, một câu mỗi prompt ID, không trùng ID giữa hai tập; lấy theo thứ tự archive ở revision cố định, không chọn theo chất lượng dự đoán. Whisper small cũng được khóa revision và checksum. Không có bước fine-tune Whisper trong thí nghiệm này.

## Các file kết quả

| File trong thư mục Toxic/Violence | Nội dung |
|---|---|
| `protocol.json` | Seed, cấu hình định trước, tiêu chí chọn trên validation, môi trường |
| `validation_search.csv` | Kết quả từng cấu hình/ngưỡng trên validation |
| `selection.json` | Lựa chọn đã khóa và checksum checkpoint |
| `metrics_by_split.csv` | Baseline và ứng viên trên train/validation/test, support, Accuracy, Loss, Precision/Recall/F1 và AUC |
| `comparison.csv` | Hai hàng test để so trước/sau |
| `confusion_matrix.csv` | TN/FP/FN/TP từng split |
| `learning_curve.csv` | Các lượt fit tập train con độc lập; trục số mẫu huấn luyện |
| `evaluation.json` | Kết quả máy đọc cùng định nghĩa loss và giới hạn diễn giải |
| `*.png`, `*.svg` | Đồ thị để xem hoặc đưa vào báo cáo/slide |

Thư mục `asr/` có `asr_metrics.csv`, đồ thị WER/CER/thời gian và `toxic_asr_output_changes.csv`. Bảng cuối so đầu ra classifier trên transcript ASR với đầu ra trên transcript tham chiếu; **không phải nhãn Toxic do người gán**, không gọi đây là accuracy hoặc false-positive rate của moderation.

`summarize_experiments` ghép ba CSV `metrics_by_split.csv`, `comparison.csv`, `confusion_matrix.csv` ngay dưới thư mục run public. Chỉ đọc số đo đã lưu, không chạy lại test. Cột `selected` là ứng viên chọn bằng validation; quyết định đưa vào app nằm riêng trong `release_decision.json`, vì ứng viên không nhất thiết tốt hơn baseline trên test.

## Diễn giải đúng số đo

- Accuracy là tỷ lệ dự đoán đúng tại ngưỡng đã chọn. Toxic mất cân bằng lớp nên cần đọc cùng Precision/Recall/F1, support và baseline luôn đoán lớp âm.
- Loss là binary cross-entropy không trọng số, tính trên score xác suất ở từng split; không phải giá trị mục tiêu có regularization/class weight bên trong solver Logistic Regression.
- Learning curve dùng những model fit trên các lượng dữ liệu train khác nhau. Không có epoch history cho solver cũ và không dựng đường epoch từ kết quả cuối. Ngưỡng cố định theo cấu hình chọn trên validation.
- WER tiếng Việt ở đây dùng token cách bằng khoảng trắng (gần mức âm tiết), NFC/casefold/bỏ dấu câu; CER giữ dấu tiếng Việt và bỏ khoảng trắng. Báo corpus tổng lỗi/tổng token, không lấy trung bình đơn giản WER từng câu.
- ID `base-greedy` chỉ cấu hình v1 `beam_size=1`, vẫn giữ temperature fallback mặc định của faster-whisper 1.2.1; các ứng viên beam5 dùng `temperature=0`. Mỗi audio/cấu hình được đo một lần và lưu kết quả, không chọn lượt có WER đẹp nhất.
- Runtime trong các lượt thí nghiệm có thể trùng lúc trích feature trên CPU, là số đo của lần chạy này; không coi là benchmark tốc độ cô lập hoặc độ trễ toàn app.
- Test ViCTSD/RLVS từng được báo cáo ở baseline; lần này không dùng test để chọn cấu hình mới nhưng không gọi chúng là một tập test đích chưa từng xem. ASR test 16 câu và dữ liệu đọc thành tiếng rất nhỏ, chưa đại diện hội thoại/nhạc/nhiễu trên video mạng xã hội.
- Các video Huy đã báo lỗi chưa được lưu; pilot còn chờ nhãn người thật. Không tự sinh nhãn hoặc coi điểm benchmark là độ chính xác của app trên TikTok/Reels/Shorts.

Nguồn: [ViCTSD](https://github.com/tarudesu/ViCTSD), [RLVS](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset), [FLEURS](https://huggingface.co/datasets/google/fleurs), [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Hash/split chi tiết kế thừa report baseline và protocol của run; raw media/transcript/checkpoint chỉ ở local.
