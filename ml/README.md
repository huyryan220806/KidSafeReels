# Công việc ML lead

Đã có Toxic tiếng Việt, visual Violence và ASR trên CPU (16/10/2026). Chạy model thật theo [VIDEO_PIPELINE.md](../docs/VIDEO_PIPELINE.md); metric tại [Toxic](../reports/TOXIC_BASELINE.md), [Visual](../reports/VISUAL_BASELINE.md), smoke test tại `reports/video_smoke_results.json`.

## Chạy baseline văn bản

Tại thư mục gốc repo, sau khi tạo `.venv` theo README:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv\Scripts\python.exe -m ml.toxic_baseline download
.\.venv\Scripts\python.exe -m ml.toxic_baseline train
.\.venv\Scripts\python.exe -m ml.toxic_baseline predict --text "Chúc bạn một ngày tốt lành"
```

Downloader lấy 3 CSV của ViCTSD từ commit cố định của tác giả; lưu dữ liệu và checksum trong `data/raw/victsd/` (gitignore). Train tạo `models/checkpoints/toxic-tfidf-v1.joblib` (gitignore) và `reports/toxic_baseline_metrics.json`. Chạy lại train sẽ ghi đè hai đầu ra này; giữ bản report cũ trước khi thử cấu hình khác. Model dùng TF-IDF ký tự 3–5 gram + Logistic Regression, seed 42; fit vectorizer chỉ trên train và chọn ngưỡng bằng macro-F1 validation. Chưa thử nhiều cấu hình sau khi xem test.

Nhánh text trong `ml.inference` dùng `ToxicClassifier` và gọi `classifier.predict(segment_text)` theo từng đoạn ASR. Chỉ nạp joblib tự huấn luyện/nguồn tin cậy. Model text không tự có timestamp hay kết luận an toàn video; adapter giữ timestamp từ ASR và kết hợp kết quả visual.

Test ML: `python -m pytest tests/test_toxic_baseline.py -q` (cần cài pytest). Full test cần thêm `requirements-dev.txt`. App owner chỉ chạy mock không cần cài requirements ML.

## Việc còn lại sau baseline

Ưu tiên tập video tiếng Việt có nhãn chung và kiểm thử máy thứ hai. Các mục bên dưới là định hướng ban đầu; bản v1 dùng 8 frame/clip để giảm thời gian CPU, hai benchmark riêng, và chưa có calibration ở cấp video. Không coi việc ghép hai nhánh là đã đánh giá fusion.

1. **Dữ liệu:** chốt nhãn đa nhãn và manifest theo `data/README.md`; định nghĩa nguy cơ dựa trên nội dung quan sát được. Dùng clip pilot để kiểm tra feasibility trước khi mở rộng.
2. **Preprocess:** giữ video_id/timestamp; lấy mẫu 1 frame/giây (tối đa 60 frame); lấy transcript phân đoạn với ASR; xử lý không có audio. Cache theo hash video + cấu hình + version model. Cần thống nhất trình giải mã và FFmpeg với app owner.
3. **Baseline visual:** dùng ResNet18 pretrained làm frozen feature extractor; train logistic regression hoặc MLP nhỏ cho violence trên dữ liệu được gán nhãn. Trọng số backbone không thay cho classifier của bài toán. Đo frame/segment evidence; nhãn clip không tự biến mọi frame thành nhãn thật.
4. **Baseline text:** TF-IDF + logistic regression cho toxic_language; cùng split video như visual. Không dùng transcript chứa thông tin từ tập test khi fit vectorizer.
5. **Thử nghiệm tiếp:** PhoBERT frozen embedding + classifier để so sánh TF-IDF; làm đúng bước word segmentation và giới hạn độ dài theo model card. Fine-tune chỉ khi baseline, dữ liệu và thời gian cho phép.
6. **Fusion:** gom điểm theo video từ các đoạn; dùng mỗi nhãn đúng nhánh có năng lực dự đoán. Khi có hai nhánh cùng nhãn, so sánh max/weighted fusion; chọn ngưỡng trên validation. Ghi cấu hình ngưỡng và lớp hỗ trợ vào artifact.
7. **Inference:** triển khai nhánh model của `kidsafe/pipeline.py`, nạp model một lần, inference eval/no-grad, trả schema hợp lệ; thiếu checkpoint thì lỗi rõ ràng, không tự tải hàng GB trong request.
8. **Evaluation:** precision/recall/F1 và support từng lớp, macro/micro-F1 trên các lớp thực hỗ trợ, confusion từng lớp, latency median/p95 và phần cứng. So sánh visual-only/text-only/fusion trên cùng test và nói rõ modality chỉ hỗ trợ lớp nào.

## Module đã triển khai

- `toxic_baseline.py`: tải, train, đánh giá và suy luận Toxic văn bản.
- `download_rlvs.py`, `train_visual.py`: tải RLVS, dedup/group split, cache features, train/evaluate Violence.
- `visual.py`, `media.py`: backbone/head, kiểm tra video và lấy frame.
- `video_assets.py`: tải trước backbone/ASR; `inference.py`: API adapter + CLI.
- `smoke_video.py`: kiểm chứng model qua HTTP; `package_models.py`: tạo gói artifact local.

Phụ thuộc tách `requirements-ml.txt` và `requirements-video.txt`; torch CPU cài theo hướng dẫn riêng. Chưa có notebook Colab vì đã chạy được trên CPU.

## Bàn giao cho app cuối D4

- Classifier/checkpoint và cách tải được phép chia sẻ, kèm checksum/version.
- Cấu hình preprocessing, label mapping, ngưỡng và danh sách lớp thực sự hỗ trợ.
- Hàm `analyze_video` chạy trên 5 clip thật, evidence có nguồn/timestamp.
- Đo RAM/thời gian CPU và hướng dẫn cài phụ thuộc model.
- Danh sách giới hạn thực; chưa có metric thì ghi chưa đo.

Tài liệu: [Torchvision weights](https://docs.pytorch.org/vision/stable/models.html), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [PhoBERT](https://huggingface.co/vinai/phobert-base-v2), [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html). Group split giúp tách theo nhóm, không tự đảm bảo cân bằng lớp; phải kiểm tra phân bố sau chia.
