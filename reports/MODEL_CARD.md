# Model card — baseline text local, chưa phát hành model video

Cập nhật 16/10/2026: `toxic-tfidf-v1` đã huấn luyện CPU trên ViCTSD và lưu local tại `models/checkpoints/toxic-tfidf-v1.joblib`. Xem [báo cáo](TOXIC_BASELINE.md) và [metadata/checksum](toxic_baseline_metrics.json). Phạm vi chỉ Toxic văn bản; F1 Toxic=0,4492, recall=0,4862 trên 988 bình luận test sau lọc trùng. Không dùng cho quyết định an toàn thực tế. Các mục dưới đây mô tả model video còn chưa hoàn thành.

- Version/checksum: chưa có.
- Phạm vi/nhãn được hỗ trợ: chưa có; kế hoạch P0 là violence và toxic_language.
- Nguồn dữ liệu, giấy phép, số lượng và split: chưa chốt.
- Backbone/classifier/preprocess/ngưỡng: chưa huấn luyện và chọn trên validation.
- Phần cứng, thời gian train/inference và phiên bản thư viện: chưa đo.
- Metrics và support từng nhãn: chưa đo.
- Giới hạn cần đánh giá: sai ASR, ngữ cảnh văn hóa/ngôn ngữ, thiên lệch dữ liệu, thiếu audio, bỏ sót cảnh ngắn, lớp không được hỗ trợ.
- Mục đích: hỗ trợ người lớn kiểm tra nội dung video ngắn; không bảo đảm an toàn hay thay thế giám sát.
- Mock hiện tại: luôn review, điểm null, không có evidence thật.
