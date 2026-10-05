# Model artifacts

Chưa có checkpoint trong repo. Để file local tại `models/checkpoints/` (đã gitignore). Trước khi bàn giao, ghi nơi tải mà nhóm có quyền chia sẻ, checksum SHA-256, version, phần cứng, phụ thuộc, label mapping và ngưỡng. Checkpoint backbone và classifier đã học là hai artifact khác nhau nếu pipeline tách riêng.

Không commit torch weights, pickle/joblib hay token Hugging Face. Chỉ nạp artifact từ nguồn tin cậy và kiểm tra checksum; thành viên app chỉ cần những artifact dùng inference, không cần toàn bộ dữ liệu huấn luyện.
