# Model artifacts

**Hiện hành: CPU v3** gồm PhoBERT fine-tune + R3D-18/LR + Whisper small beam5, ZIP `models/checkpoints/bundle/kidsafe-cpu-v3.zip`, 17 tệp theo [manifest](artifacts.json). [Cài v3/khôi phục](../docs/MODEL_UPDATE_2026-10-28.md). Bản Toxic hybrid 29/10 thêm code/lexicon trong Git, không đổi ZIP/trọng số; sau cập nhật phải khởi động lại API. [Cách chạy](../docs/TOXIC_HYBRID.md). Nội dung dưới ghi lịch sử v2.

Artifact thật được lưu local tại `models/checkpoints/` (đã gitignore), không nằm trong Git. CPU v2 dùng `toxic-tfidf-v1.joblib`, `violence-resnet18-v1.joblib`, `resnet18-imagenet1k-v1.pth`, `faster-whisper-small/` và `inference-config.json`. Hai classifier giữ v1 vì ứng viên mới không đạt yêu cầu test. [Cài v2/khôi phục v1](../docs/MODEL_UPDATE_2026-10-20.md), [manifest đang dùng](artifacts.json), [model card](../reports/MODEL_CARD.md).

Gói mới là `models/checkpoints/bundle/kidsafe-cpu-v2.zip`, kèm SHA-256; 9 tệp inference, không chứa dữ liệu huấn luyện. `ml.package_improved stage` tạo bundle từ quyết định thực nghiệm, `activate` yêu cầu báo cáo kiểm thử pass. Giữ `artifacts-v1.json` và ZIP v1 để khôi phục. `ml.package_models` là công cụ baseline cũ, không chạy để đóng lại v2. API kiểm tra phiên bản scikit-learn và SHA của backbone so với classifier.

Không commit torch weights, pickle/joblib hay token Hugging Face. Chỉ nạp artifact từ nguồn tin cậy và kiểm tra checksum; thành viên app chỉ cần những artifact dùng inference, không cần toàn bộ dữ liệu huấn luyện.
