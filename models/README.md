# Model artifacts

Artifact thật được lưu local tại `models/checkpoints/` (đã gitignore), không nằm trong Git. Gồm `toxic-tfidf-v1.joblib`, `violence-resnet18-v1.joblib`, `resnet18-imagenet1k-v1.pth` và `faster-whisper-base/`. [Hướng dẫn tải/huấn luyện/đóng gói](../docs/VIDEO_PIPELINE.md), [manifest checksum](artifacts.json), [model card](../reports/MODEL_CARD.md).

Chạy `python -m ml.package_models` tạo `models/checkpoints/bundle/kidsafe-cpu-v1.zip` cùng file SHA-256; gói chỉ chứa artifact inference, không chứa dữ liệu huấn luyện. Chuyển gói này trực tiếp cho thành viên app trong phạm vi sử dụng dữ liệu được phép. Khi tái huấn luyện, cần cập nhật manifest và chuyển cả classifier lẫn backbone tương ứng. API kiểm tra phiên bản scikit-learn và SHA của backbone so với classifier.

Không commit torch weights, pickle/joblib hay token Hugging Face. Chỉ nạp artifact từ nguồn tin cậy và kiểm tra checksum; thành viên app chỉ cần những artifact dùng inference, không cần toàn bộ dữ liệu huấn luyện.
