# Cập nhật giao diện — 17/10/2026

Đối tượng: phụ huynh/giáo viên chưa học công nghệ thông tin. Nhánh làm việc `feat/app`; thay đổi chỉ ở giao diện, theme, tài liệu và kiểm thử UI.

## Thay đổi

- Nhận diện KidSafeReels màu tím dịu, minh họa gia đình bằng SVG nội bộ; không tải hình/font bên ngoài.
- Chia ba bước, hiển thị bước hiện tại theo upload/chờ/kết quả. Các khối chọn video và xem kết quả tách rõ, tự xếp dọc trên màn hình nhỏ.
- Nút **Chọn video từ máy**, **Kiểm tra video** và lời hướng dẫn bằng tiếng Việt; lỗi kết nối/model được giải thích thành việc người dùng có thể làm tiếp.
- Kết luận có màu và chữ; nhóm chưa đánh giá luôn hiện rõ, không biến null thành 0. Màu không tự tạo nhãn/đổi ngưỡng. Nhóm có điểm chỉ ghi “Có kết quả phân tích”.
- Dời bảng điểm số, phiên bản/schema, mã lượt và JSON vào mục chi tiết đóng mặc định. Lưu ý về độ bao phủ vẫn nằm trong kết quả và mở mặc định.
- Thêm tóm tắt TXT tiếng Việt để người không biết JSON vẫn lưu được kết quả. Báo cáo giữ nhãn mô phỏng, điểm chưa đánh giá và mô tả/timestamp evidence của pipeline.
- Giữ playback tại mốc bằng chứng, session, reset khi đổi file và xử lý lỗi. Nhắc người dùng đọc mô tả khi bằng chứng hình ảnh bao phủ toàn clip.
- Streamlit chưa có tham số công khai cho chữ trên nút uploader: CSS chỉ thay phần chữ nhìn thấy; vẫn dùng input và hành vi bàn phím gốc. Vùng uploader có nhãn tiếng Việt. Cần kiểm tra lại selector khi nâng Streamlit.

## Kiểm chứng

- `python -m pytest -q`: **76 passed**, 36,75 giây; một deprecation warning từ thư viện kiểm thử như trước.
- Test UI xác nhận bảng chỉ số nằm trong phần chi tiết đóng mặc định; null khác 0, reset khi đổi file, timeout/lỗi, evidence playback và luồng UI → API → decoder.
- Kiểm thử thêm bản TXT đánh dấu mock, giữ null/0 và HTML escape tên file upload.
- Trình duyệt: upload video tổng hợp không tiếng, nhận mock, tải TXT và đọc nội dung tải về thành công. Không dùng kết quả tổng hợp để đánh giá ML.
- Bố cục desktop 1366×1000 và điện thoại 390×844; document width trên điện thoại bằng viewport 390px, không tràn ngang.
- Ảnh mới tại `docs/screenshots/app-redesign-desktop.jpg` và `app-redesign-mobile.jpg`.

Chưa ghép nhánh/checkpoint ML của Huy. Không sửa `kidsafe/`, `ml/` hoặc schema API. Chưa đo trải nghiệm bằng người dùng thực tế; cần lấy phản hồi phụ huynh/giáo viên trong buổi thử sản phẩm.
