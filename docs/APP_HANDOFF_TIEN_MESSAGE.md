# Tiến bàn giao lại bản tích hợp - 18/10/2026

Đã nhận `feat/ml-release` f5d9310, ZIP khớp hash ngoài, 17 tệp bàn giao và 8 artifact đều kiểm tra thành công. Giao diện mới của app giữ nguyên, chạy với model CPU trên máy Intel Core i5-12450H / Windows 11 / Python 3.12.5 của Tiến.

- 118 test tự động qua, không skip; pip check và verifier qua.
- 13/13 ca HTTP qua worker/model thật và 9/9 ca UI AppTest gọi API/model thật qua.
- Mẫu tiếng Việt FLEURS ghép nền tổng hợp có score cả Violence/Toxic, evidence lời nói và start_time đúng ở widget. Video không tiếng giữ Toxic null.
- Sửa ánh xạ `InvalidMediaError` từ pipeline thành 422; thêm regression process thật. Không thay model/ngưỡng/schema.
- Có `Chay-KidSafeReels.cmd`, setup/run model, biên bản riêng, PDF và PPTX cập nhật thông tin Tiến cùng số đo máy app.

Chi tiết và lệnh tái hiện: `APP_ACCEPTANCE_TIEN.md`. Chạy `scripts/run-model.ps1` cho backend model; không dùng `scripts/smoke_api` vốn chỉ dành cho mock.

Chưa chạy classifier trên pilot ở máy Tiến. 6 nhãn độc lập của app và 25 nhãn của Huy vẫn chờ người thực hiện. Chưa có metric pilot/ablation mới. Công cụ browser từ chối truy cập trang xem trước nên kiểm thử UI ở mức AppTest, chưa xác nhận bằng tay trong browser hoặc quay video màn hình mới. Cần cả hai hoàn thiện các mục này trước khi coi là bản nộp cuối.

Thông tin app đã được Tiến xác nhận: Lê Quyết Tiến, 2474802010386, lớp 03_02, giảng viên Nguyễn Thái Anh. Thông tin đầy đủ của Huy còn cần điền khi nhóm chốt hồ sơ. Checkpoint, raw media, transcript, nhãn và log chi tiết không lên Git.
