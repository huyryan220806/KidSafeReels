# Dữ liệu và nhãn

Chưa chọn/tải dataset nào. Đây là kế hoạch và format cần dùng, không phải tuyên bố có sẵn dữ liệu. ML lead kiểm tra nguồn và quyền sử dụng ngay D1.

- Pilot 20–30 clip để thử pipeline; mục tiêu 150–300 clip ngắn nếu có nguồn phù hợp và thời gian gán nhãn. Số lượng này chỉ phục vụ mini project, không chứng minh khả năng tổng quát ngoài thực tế.
- Ưu tiên video được phép sử dụng, tự tạo hoặc benchmark có điều kiện rõ ràng; lưu nguồn, giấy phép và group_id. Không suy ra quyền tái phân phối từ việc xem được công khai.
- Hai người gán nhãn chéo khoảng 20% pilot; thảo luận bất đồng và cập nhật quy tắc trước khi gán phần còn lại.
- Train/validation/test dự kiến 70/15/15 theo group_id của video gốc/nguồn liên quan. Chia trước trích frame; video cắt ngắn hoặc bản reupload gần trùng phải cùng split. Kiểm tra mỗi lớp có support phù hợp, không chỉ tỷ lệ tổng.
- Dùng validation chọn mô hình/ngưỡng; test chỉ để đánh giá cuối. Với test nhỏ, báo số lượng mẫu và kết quả mang tính thăm dò.

## Quy tắc nhãn

| Nhãn | Tiêu chí thao tác ban đầu | Ghi chú |
|---|---|---|
| violence | Hình ảnh có hành vi đánh nhau/tấn công hoặc bạo lực nhìn thấy | Phân biệt cảnh thể thao, hoạt hình và ngữ cảnh; lưu ca mơ hồ |
| toxic_language | Lời nói/văn bản có chửi tục, xúc phạm hoặc đe dọa | ASR có thể sai; nhãn chuẩn cần người nghe/đọc kiểm tra |
| sexual | Nội dung gợi dục không phù hợp theo định nghĩa nhóm | P1; không thu thập nội dung xâm hại trẻ em |
| dangerous_acts | Hành vi/thử thách nguy hiểm quan sát được | P1; một frame đơn có thể không đủ xác định |

Mỗi nhãn có giá trị **1** (có), **0** (đã xem và xác định không có), **trống** (chưa biết/chưa gán). Không ép ô trống thành 0 khi train. Safe chỉ suy ra trong phạm vi các nhãn đã được đánh giá, không phải nhãn thủ công thứ năm. Clip nhiều rủi ro có nhiều giá trị 1.

`manifest.example.csv` chỉ chứa header. Manifest thật và đường dẫn/video riêng tư để trong `data/private/` hoặc `data/raw/` (đã gitignore). `video_path` là đường dẫn tương đối trong kho dữ liệu local; `group_id` nhóm clip cùng nguồn, `split` thuộc train/val/test, `license` mô tả quyền sử dụng. Các timestamp evidence có thể lưu vào JSONL riêng khi gán nhãn đoạn.
