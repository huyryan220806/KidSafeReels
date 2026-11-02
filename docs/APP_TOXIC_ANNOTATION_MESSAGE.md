# Tin nhắn Huy gửi thành viên app

Gửi đoạn dưới kèm file **`app-toxic-dev30-2026-11-02.zip`**. File ZIP ở máy Huy: `data/private/app-toxic-dev30-2026-11-02/app-toxic-dev30-2026-11-02.zip`; không có trong GitHub.

---

Nhờ bạn gán nhãn giúp mình 30 video lời nói để cải thiện Toxic cho KidSafeReels. Tổng thời lượng video khoảng 9 phút 21 giây; việc nghe lại, chép lời và đánh dấu bằng chứng sẽ cần thêm thời gian.

Hướng dẫn: https://github.com/huyryan220806/KidSafeReels/blob/feat/ml-release/docs/APP_TOXIC_ANNOTATION.md

1. Nhận ZIP mình gửi, giải nén toàn bộ rồi mở `review-app.html` bằng **Chrome hoặc Edge bên ngoài Codex**. Không cần chạy app/model hoặc cài thêm thư viện.
2. Nghe hết từng clip, xác nhận ngôn ngữ và chọn **0 / 1 / chưa rõ**. Với nhãn 1, ghi mốc giây và câu nghe được.
3. Ưu tiên chép đầy đủ lời nói ở cả clip bình thường lẫn Toxic; chỉ tích đã đối chiếu toàn bộ transcript khi thực sự kiểm đủ. Chỗ không nghe rõ thì ghi lại, không đoán.
4. Gán độc lập theo audio, chưa xem dự đoán model hoặc nhãn mình đã làm. Nếu đã biết kết quả một clip thì ghi chú giúp mình.
5. Sau mỗi 5 clip, bấm **Tải JSON nhãn hiện tại**. Dùng **Nạp lại JSON** để tiếp tục khi đổi máy/trình duyệt.
6. Xong 30 clip, gửi mình file JSON mới nhất tên dạng `annotations-app-app-toxic-dev30-2026-11-02.json`. Nếu có nhãn chưa rõ thì để nguyên và báo mình; không gửi file `.blank.json`.

Code và hướng dẫn nằm ở nhánh `feat/ml-release`; video nằm trong ZIP gửi riêng. Giữ JSON gốc sau khi gửi để có thể đối chiếu nếu cần.
