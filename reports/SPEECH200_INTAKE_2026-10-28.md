# Tiếp nhận speech200 — 28/10/2026

Đã kiểm tra 200 video local Huy cung cấp, tổng **3.941,284 giây (65 phút 41 giây)**, dung lượng 188.822.285 byte (~180,08 MiB). Đây là báo cáo chất lượng đầu vào, **không phải kết quả huấn luyện hoặc độ chính xác model**. Số liệu máy đọc: [JSON intake](SPEECH200_INTAKE_2026-10-28.json).

| Hạng mục | Kết quả |
|---|---:|
| Nhóm được mô tả là Toxic | 100 clip / 74 video nguồn / 1.940,909 giây |
| Nhóm được mô tả là bình thường | 100 clip / 58 video nguồn / 2.000,375 giây |
| Tổng ID YouTube nguồn | 132 |
| Khớp SHA-256, dung lượng, thời lượng nguồn | 200/200 |
| Giải mã đầy đủ hình và tiếng | 200/200 |
| Trùng SHA trong speech200 | 0 |
| ID nguồn chung giữa hai nhóm | 0 |
| Khoảng cắt chồng nhau trong cùng ID nguồn | 0 |
| Cặp dHash gần nhau cần xem lại | 28 |
| Trong đó cặp khác ID nguồn | 7 |
| Trong đó cặp khác lớp mô tả | 0 |
| Phiếu đã nghe/gán nộp vào bộ mới | 0 |
| Giao Huy / app | 200 / 40 clip |

Độ dài clip 10,790–59,8 giây, đều có một luồng hình và một luồng tiếng, trong giới hạn đầu vào app. Kiểm kỹ thuật không xác nhận clip có nội dung đúng lớp, đúng ngôn ngữ hoặc transcript chính xác.

Metadata cả 200 clip ghi `labels.toxic_language=null`; 100 nhãn gợi ý 1 và 100 gợi ý 0 được lọc bằng ASR/từ khóa. Mô tả của Huy lưu riêng ở `user_reported_clip_label`. Không coi việc thiếu từ tục trong ASR là bằng chứng chắc chắn clip bình thường. Bộ cân bằng do tuyển chọn có thiên lệch; không đại diện phân bố người dùng app.

Audit hình lấy dHash 64 bit ở 1 fps, khoảng cách Hamming ≤2, cần ≥3 hash khác nhau/ảnh không phẳng ở mỗi clip. 7 cặp khác ID cần rà nguồn; cảnh cùng người/bố cục có thể tạo ứng viên giả. Reupload cắt/crop/đổi bố cục hoặc chỉ trùng tiếng có thể không được phát hiện. Không dùng số 0 trùng SHA để khẳng định đã hết leakage. Chưa khóa train/validation/test, chưa trích đặc trưng model hoặc chạy ASR mới.

50 MP4 trong `data/Video_Train/Violence` khớp SHA với tập AIRTLab đã đánh giá ở bản v3. Kết quả AIRTLab 53% Accuracy vẫn là kết quả cũ; bộ tải lại không bổ sung 50 mẫu độc lập và không dùng để train rồi báo lại test cũ.

## Đầu ra và kiểm chứng

- Manifest mới `speech200-intake-v2`, SHA-256 `034eef5128a02a21ead3ae76045868b162bb928f9cba059bc27bd5407f459439`; giữ manifest speech100 cũ nguyên trạng.
- Phiếu Huy/app dùng mã trung tính `review-001`… và thứ tự trộn. App nhận 40 ID nguồn khác nhau, 20 clip mỗi nhóm mô tả, chưa phải test. Không đưa nhãn gợi ý, tiêu đề nguồn, transcript ASR hoặc dự đoán vào phiếu.
- `python -m pytest -q tests/test_speech_intake.py`: 31 tests pass; kiểm null/uncertain, mốc bằng chứng, nhãn mâu thuẫn, phiếu sai ID/hash, đường dẫn vượt thư mục, bảo toàn bản nháp và gói app không rò nhãn.
- Trình duyệt local: nạp video có `readyState=4`, 640×360, không lỗi; chuyển clip/khôi phục bản nháp được; xuất bị chặn khi thiếu xác nhận hoặc bằng chứng, nhãn 0 mâu thuẫn bằng chứng. Xuất rồi nạp JSON trống thành công; Python xác nhận 200 assigned/0 completed. Console không có lỗi ở lượt kiểm.
- Kiểm UI dùng trang/storage QA riêng; phiếu thật Huy/app vẫn trống. Không tuyên bố đã nghe/gán 200 clip. Chưa kiểm nghiệm người dùng trên máy Tiến.
- Raw video, fingerprint/nguồn/nhãn/ZIP/ảnh QA giữ local. Code, hướng dẫn và thống kê tổng hợp được lưu trên nhánh ML; model CPU v3 và API không thay đổi.

[Hướng dẫn mở phiếu và bước tiếp theo](../docs/SPEECH200_ANNOTATION.md). Hoàn tất nhãn/nhóm nguồn, chia theo nguồn trước xử lý, chọn model/ngưỡng trên validation rồi mới xuất Accuracy/Loss/F1 và đồ thị/CSV test. Không có metric speech200 để công bố ở giai đoạn này.
