# Bàn giao cho thành viên làm app

Bạn có thể clone và chạy ngay theo README; hiện chưa cần GPU hay model. Mục tiêu là luồng người dùng tải video lên → chờ → xem nhãn, điểm, cảnh báo và bằng chứng → tải JSON.

**Bàn giao ML 16/10/2026:** nhánh `feat/ml-video-pipeline` có visual + ASR + Toxic thật. Giữ nguyên schema v1.0 và signature `analyze_video`; API thêm ánh xạ media lỗi sang HTTP 422. Cài model hoặc nhận gói local theo [VIDEO_PIPELINE.md](VIDEO_PIPELINE.md). Điểm dưới ngưỡng vẫn `review`; Sexual/Dangerous luôn null; visual evidence chưa định vị cảnh. Chỉ sửa dòng giới thiệu UI cho khớp trạng thái mới, các phần playback/bố cục vẫn thuộc app owner.

Nếu bắt đầu một nhánh app mới từ bản ML này:

```powershell
git fetch origin
git switch -c feat/app-integration origin/feat/ml-video-pipeline
```

Nếu đã có nhánh app riêng, giữ công việc của bạn và tích hợp nhánh ML theo quy trình Git; không reset hay ghi đè thay đổi local.

**Ba đầu việc app tiếp theo:** (1) nhận ZIP artifact local từ Huy và chạy `KIDSAFE_BACKEND=model`; (2) thêm playback và hiển thị evidence, phân biệt khoảng toàn clip với đoạn ASR; (3) thử upload không tiếng, audio không phải tiếng Việt, video hỏng/quá dài/thiếu model, rồi quay demo. Smoke test ML chỉ gọi API trong tiến trình, chưa thay thế kiểm thử trình duyệt trên máy của bạn.

## Công việc theo thứ tự

1. Chạy mock trên máy mình, tạo nhánh `feat/app-upload`.
2. Hoàn thiện bố cục dễ dùng; giữ badge mô phỏng rõ ràng. `null` hiển thị “chưa đánh giá”, không hiển thị 0%.
3. Pipeline model đã kiểm tra media thật và thời lượng ≤60 giây; kiểm thử lỗi 422 từ UI. Nếu cần kiểm tra ở mock hoặc trước inference, tái dùng `ml.media.inspect_video` sau khi cài phụ thuộc video, không tạo bộ kiểm tra khác lệch quy tắc.
4. Kiểm tra việc giữ kết quả trong session và reset khi đổi file (starter đã có). Bổ sung hiển thị video, có thể nhảy đến timestamp evidence.
5. Bổ sung trạng thái API chưa chạy, model chưa sẵn sàng, timeout, file hỏng và thiếu audio. UI không suy diễn nhãn từ tên file.
6. D4 ghép model thật qua cùng schema, kiểm tra ít nhất 5 clip; D5 xử lý các tình huống lỗi và đo thời gian đầu-cuối.
7. D6 chạy lại trên máy thứ hai, chụp UI và quay demo. Local là phương án mặc định; online là tùy chọn.

## Phạm vi file

Sở hữu `apps/api/`, `apps/web/`, test UI/integration và cấu hình chạy app. Đọc `kidsafe/schemas.py` nhưng thống nhất với Huy trước khi đổi. Không tự sửa nhãn/ngưỡng/model để UI đẹp hơn. Không cần làm login, database, extension hay app điện thoại trong tuần này.

## Prompt để bắt đầu phiên làm việc của bạn

> Đọc AGENTS.md, README.md, docs/PLAN.md, docs/API_CONTRACT.md và docs/HANDOFF_APP.md trong repo KidSafeReels. Tôi là thành viên phụ trách app. Triển khai công việc app theo ngày hiện tại, bắt đầu từ việc chạy bộ khung mock và kiểm tra luồng upload. Giữ đúng API contract, ghi rõ mock/model, không tự tạo điểm đánh giá thật. Trước khi sửa, kiểm tra git status và tạo nhánh feat/app-* phù hợp. Kết thúc bằng cách chạy, kết quả kiểm tra và các điểm cần bàn giao cho ML lead.
