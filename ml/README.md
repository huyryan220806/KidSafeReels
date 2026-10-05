# Công việc ML lead

Thư mục này chưa có model/training code. Triển khai tuần tự để mỗi ngày có đầu ra đo được; ưu tiên CPU, dùng Colab khi cần batch trích đặc trưng.

## Lộ trình kỹ thuật

1. **Dữ liệu:** chốt nhãn đa nhãn và manifest theo `data/README.md`; định nghĩa nguy cơ dựa trên nội dung quan sát được. Dùng clip pilot để kiểm tra feasibility trước khi mở rộng.
2. **Preprocess:** giữ video_id/timestamp; lấy mẫu 1 frame/giây (tối đa 60 frame); lấy transcript phân đoạn với ASR; xử lý không có audio. Cache theo hash video + cấu hình + version model. Cần thống nhất trình giải mã và FFmpeg với app owner.
3. **Baseline visual:** dùng ResNet18 pretrained làm frozen feature extractor; train logistic regression hoặc MLP nhỏ cho violence trên dữ liệu được gán nhãn. Trọng số backbone không thay cho classifier của bài toán. Đo frame/segment evidence; nhãn clip không tự biến mọi frame thành nhãn thật.
4. **Baseline text:** TF-IDF + logistic regression cho toxic_language; cùng split video như visual. Không dùng transcript chứa thông tin từ tập test khi fit vectorizer.
5. **Thử nghiệm tiếp:** PhoBERT frozen embedding + classifier để so sánh TF-IDF; làm đúng bước word segmentation và giới hạn độ dài theo model card. Fine-tune chỉ khi baseline, dữ liệu và thời gian cho phép.
6. **Fusion:** gom điểm theo video từ các đoạn; dùng mỗi nhãn đúng nhánh có năng lực dự đoán. Khi có hai nhánh cùng nhãn, so sánh max/weighted fusion; chọn ngưỡng trên validation. Ghi cấu hình ngưỡng và lớp hỗ trợ vào artifact.
7. **Inference:** triển khai nhánh model của `kidsafe/pipeline.py`, nạp model một lần, inference eval/no-grad, trả schema hợp lệ; thiếu checkpoint thì lỗi rõ ràng, không tự tải hàng GB trong request.
8. **Evaluation:** precision/recall/F1 và support từng lớp, macro/micro-F1 trên các lớp thực hỗ trợ, confusion từng lớp, latency median/p95 và phần cứng. So sánh visual-only/text-only/fusion trên cùng test và nói rõ modality chỉ hỗ trợ lớp nào.

## Cấu trúc sẽ bổ sung khi triển khai

`ml/preprocess.py`, `ml/features.py`, `ml/train.py`, `ml/evaluate.py`, `ml/inference.py`, `ml/configs/` và notebook Colab nếu cần. Mỗi script có seed/cấu hình/split rõ ràng; tránh logic chỉ tồn tại trong notebook. Tách requirements ML riêng sau khi thử tương thích để người làm app không phải cài toàn bộ stack.

## Bàn giao cho app cuối D4

- Classifier/checkpoint và cách tải được phép chia sẻ, kèm checksum/version.
- Cấu hình preprocessing, label mapping, ngưỡng và danh sách lớp thực sự hỗ trợ.
- Hàm `analyze_video` chạy trên 5 clip thật, evidence có nguồn/timestamp.
- Đo RAM/thời gian CPU và hướng dẫn cài phụ thuộc model.
- Danh sách giới hạn thực; chưa có metric thì ghi chưa đo.

Tài liệu: [Torchvision weights](https://docs.pytorch.org/vision/stable/models.html), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [PhoBERT](https://huggingface.co/vinai/phobert-base-v2), [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html). Group split giúp tách theo nhóm, không tự đảm bảo cân bằng lớp; phải kiểm tra phân bố sau chia.
