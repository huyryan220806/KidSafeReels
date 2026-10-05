# Khung báo cáo

**Trạng thái: chưa có kết quả thực nghiệm.** Điền các mục sau bằng dữ liệu thật, không dùng số giả làm kết quả.

1. Bài toán, người dùng và tình huống thực tế; phạm vi so với ý tưởng gốc.
2. Dữ liệu: nguồn/quyền sử dụng, số video và phân bố lớp theo split, quy tắc nhãn, cách chống trùng nguồn, bất đồng gán nhãn.
3. Phương pháp: sampling, ASR, visual/text classifier, loss/training, fusion, ngưỡng và lý do lựa chọn.
4. Thực nghiệm: phần cứng, seed, cấu hình, baseline, train/val/test, precision/recall/F1 và support từng lớp, macro/micro-F1, thời gian.
5. So sánh visual-only, text-only, fusion trên cùng test; không so hai cấu hình đánh giá trên hai tập khác nhau.
6. Sản phẩm: sơ đồ, luồng sử dụng, ảnh giao diện thật, xử lý lỗi và cách chạy.
7. Phân tích lỗi: ASR tiếng Việt, cảnh ngắn bị bỏ qua, ngữ cảnh, nhãn thiếu dữ liệu, thiếu modality, false negative/false positive.
8. Đóng góp từng thành viên, giới hạn và hướng phát triển.

| Cấu hình | Nhãn đánh giá | Test support | Macro-F1 | Recall mỗi lớp | Median/p95 latency | Phần cứng |
|---|---|---|---|---|---|---|
| Visual-only | Chưa đo | — | — | — | — | — |
| Text-only | Chưa đo | — | — | — | — | — |
| Multimodal | Chưa đo | — | — | — | — | — |

Kịch bản demo 3–5 phút: bài toán → upload clip kiểm thử đã có quyền dùng → xem kết quả/evidence → ví dụ thiếu audio hoặc lỗi → trình bày metric/giới hạn → phân công nhóm. Không dùng mock như kết quả model thật.
