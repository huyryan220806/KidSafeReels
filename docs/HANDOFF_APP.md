# Bàn giao cho thành viên làm app

Bạn có thể clone và chạy ngay theo README; hiện chưa cần GPU hay model. Mục tiêu là luồng người dùng tải video lên → chờ → xem nhãn, điểm, cảnh báo và bằng chứng → tải JSON.

## Công việc theo thứ tự

1. Chạy mock trên máy mình, tạo nhánh `feat/app-upload`.
2. Hoàn thiện bố cục dễ dùng; giữ badge mô phỏng rõ ràng. `null` hiển thị “chưa đánh giá”, không hiển thị 0%.
3. Bổ sung kiểm tra media thật và thời lượng ≤60 giây, giới hạn xử lý; hợp tác với ML lead để tránh giải mã hai lần không cần thiết.
4. Kiểm tra việc giữ kết quả trong session và reset khi đổi file (starter đã có). Bổ sung hiển thị video, có thể nhảy đến timestamp evidence.
5. Bổ sung trạng thái API chưa chạy, model chưa sẵn sàng, timeout, file hỏng và thiếu audio. UI không suy diễn nhãn từ tên file.
6. D4 ghép model thật qua cùng schema, kiểm tra ít nhất 5 clip; D5 xử lý các tình huống lỗi và đo thời gian đầu-cuối.
7. D6 chạy lại trên máy thứ hai, chụp UI và quay demo. Local là phương án mặc định; online là tùy chọn.

## Phạm vi file

Sở hữu `apps/api/`, `apps/web/`, test UI/integration và cấu hình chạy app. Đọc `kidsafe/schemas.py` nhưng thống nhất với Huy trước khi đổi. Không tự sửa nhãn/ngưỡng/model để UI đẹp hơn. Không cần làm login, database, extension hay app điện thoại trong tuần này.

## Prompt để bắt đầu phiên làm việc của bạn

> Đọc AGENTS.md, README.md, docs/PLAN.md, docs/API_CONTRACT.md và docs/HANDOFF_APP.md trong repo KidSafeReels. Tôi là thành viên phụ trách app. Triển khai công việc app theo ngày hiện tại, bắt đầu từ việc chạy bộ khung mock và kiểm tra luồng upload. Giữ đúng API contract, ghi rõ mock/model, không tự tạo điểm đánh giá thật. Trước khi sửa, kiểm tra git status và tạo nhánh feat/app-* phù hợp. Kết thúc bằng cách chạy, kết quả kiểm tra và các điểm cần bàn giao cho ML lead.
