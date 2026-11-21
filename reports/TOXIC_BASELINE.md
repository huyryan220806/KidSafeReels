# Baseline Toxic tiếng Việt — 16/10/2026

Đã chạy thực trên CPU Windows/Python 3.12.10. Nguồn số liệu máy đọc: [toxic_baseline_metrics.json](toxic_baseline_metrics.json). Đây là phân loại bình luận, chưa phải đánh giá video hoặc transcript ASR.

## Dữ liệu và phương pháp

[ViCTSD của tác giả](https://github.com/tarudesu/ViCTSD), commit `e1ffb854ef7b7b7ae45a56d132c1f125e02e55e2`. Tải ba CSV, dùng `Comment` và `Toxicity`; không dùng Title/Topic/Constructiveness làm feature. Nguồn chưa có LICENSE rõ ràng khi kiểm tra; dữ liệu không được đưa lên repo của nhóm.

Giữ split tác giả, chuẩn hóa NFC + casefold + whitespace. Loại trùng text chính xác trong từng split; loại text ở validation/test nếu xuất hiện trong split trước đó. Chính sách ưu tiên train → validation → test. Xử lý nhãn mâu thuẫn nội bộ bằng cách bỏ text đó (thực tế không gặp). Không dùng nhãn test để sửa train. Chưa loại được gần trùng hoặc trùng nguồn bài viết; không tuyên bố đây là phép chia độc lập hoàn toàn theo tác giả/chủ đề.

| Split | Gốc | Giữ lại | Toxic |
|---|---:|---:|---:|
| Train | 7.000 | 6.965 | 759 |
| Validation | 2.000 | 1.975 | 232 |
| Test | 1.000 | 988 | 109 |

TF-IDF char_wb 3–5 gram, min_df=2, tối đa 100.000 features, sublinear TF. Logistic Regression C=1, balanced class weights, liblinear, seed=42. Chọn ngưỡng trên validation từ 0,05 đến 0,95 theo macro-F1: **0,59**. Chỉ fit train; không refit trên validation. Đây là một cấu hình baseline cố định.

## Kết quả test sau loại trùng

| Chỉ số | TF-IDF + LR | Luôn đoán Non-toxic |
|---|---:|---:|
| Macro-F1 | 0,6872 | 0,4708 |
| Precision Toxic | 0,4173 | 0 |
| Recall Toxic | 0,4862 | 0 |
| F1 Toxic | 0,4492 | 0 |
| Accuracy | 0,8684 | 0,8897 |

Confusion: TN=805, FP=74, FN=56, TP=53. Baseline bỏ sót **56/109** bình luận Toxic. Accuracy của bộ luôn đoán Non-toxic còn cao hơn do mất cân bằng lớp; không dùng accuracy để tuyên bố model tốt.

Fit khoảng 2,52 giây, predict cả batch 988 bình luận khoảng 0,30 giây trên máy chạy hiện tại; không gồm tải dữ liệu, startup Python hoặc load artifact. Chưa đo độ trễ request/p95 và chưa benchmark máy khác.

## Việc tiếp theo

1. Phân tích lỗi và thử word+char TF-IDF / PhoBERT features bằng validation; không dùng test hiện tại để chọn cấu hình.
2. Bổ sung tập transcript video có nhãn và đánh giá ảnh hưởng ASR; text bình luận khác lời nói thực tế.
3. Baseline RLVS và ASR đã được tích hợp ở nhánh video; tiếp tục tạo tập video chung để đánh giá hai nhánh, xem [pipeline](../docs/VIDEO_PIPELINE.md).
4. Giữ Sexual/Dangerous ở trạng thái chưa hỗ trợ cho đến khi có dữ liệu và đánh giá.

Chưa đủ chất lượng để dùng làm bộ lọc bảo vệ trẻ em thực tế. Score chưa calibration và không phải xác suất an toàn.
