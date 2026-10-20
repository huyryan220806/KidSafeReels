# Tin nhắn gửi thành viên app

## Bản cập nhật để gửi ngày 20/10

Mình đã hoàn tất đợt thử model và xuất Accuracy/Loss/F1, CSV, đồ thị. Bản CPU v2 nâng Whisper small beam5; hai classifier Toxic/Violence giữ v1 vì ứng viên mới không cải thiện đủ trên test. Bạn lấy code mới từ `feat/ml-release`, đọc `docs/MODEL_UPDATE_2026-10-20.md`, nhận `kidsafe-cpu-v2.zip` và SHA-256 từ mình rồi cài/kiểm lại trên máy bạn. Schema và giao diện không đổi. Gói tổng hợp 18/10 chứa v1, không dùng với manifest v2. Nhãn chéo 6 clip và quay demo vẫn cần hoàn tất. Đây là nội dung để Huy gửi; chưa gửi tự động.

## Tin nhắn lịch sử ngày 18/10

Mình đã ghép app của bạn với model thật trên nhánh `feat/ml-release`. Bản tích hợp trên máy mình qua 117 tests và 13/13 ca HTTP/model. UI, worker và schema v1 vẫn giữ theo app; PyAV đã ghim 16.1.0 để chạy chung với ASR.

Bạn đọc bàn giao tại: https://github.com/huyryan220806/KidSafeReels/blob/feat/ml-release/docs/HANDOFF_APP.md

Mình gửi kèm `kidsafe-app-handoff-2026-10-18.zip` và file `.sha256`, gồm model, video/phiếu gán nhãn, hướng dẫn, báo cáo và slide. Code lấy từ GitHub; model không có trong repo nên cần nhận ZIP mới chạy đầy đủ.

Phần bạn làm tiếp:

1. Gán độc lập 6 clip bằng `review-app.html`, xuất `annotations-app.json` trước khi xem dự đoán model trên pilot. Không dùng model để điền nhãn.
2. Cài nhánh tích hợp trên máy bạn, chạy model thật và kiểm thử theo checklist; ghi lỗi/thời gian/bằng chứng vào mẫu nghiệm thu.
3. Sửa lỗi app nếu có trên nhánh `feat/app-*`, chụp UI, quay demo 3–5 phút; gửi lại commit và kết quả cho mình.

Mình phụ trách gán đủ 25 clip theo phần việc của mình, đối chiếu hai phiếu và chạy đánh giá cuối. Hiện chưa có metric pilot; Sexual/Dangerous vẫn chưa hỗ trợ. Mục tiêu đợt này là chốt bản chạy được và nghiệm thu, chưa thêm tính năng mới.

## Prompt cho Codex

```text
Tôi là thành viên phụ trách app của KidSafeReels. Hãy tiếp nhận bản bàn giao trên origin/feat/ml-release và hoàn thiện nghiệm thu trên máy tôi.

Đầu tiên đọc AGENTS.md, README.md, docs/PLAN.md, docs/API_CONTRACT.md, docs/HANDOFF_APP.md và docs/RELEASE_HANDOFF.md. Kiểm tra git status, giữ mọi công việc local; tạo hoặc dùng nhánh feat/app-* phù hợp từ bản tích hợp. Không reset hoặc force-push nhánh của thành viên khác. Nếu nhánh app hiện tại có thay đổi mới, đối chiếu trước khi ghép.

Nhận đường dẫn gói kidsafe-app-handoff-2026-10-18.zip hoặc model ZIP từ tôi nếu chưa có, kiểm tra checksum, cài đủ dependencies CPU và 8 artifact theo bàn giao. Giữ av==16.1.0, schema v1 và ngưỡng ML. Chạy kiểm thử, app backend model, kiểm tra luồng upload/preview/evidence/download/lỗi trên máy này. Ghi kết quả thực tế vào bản sao docs/APP_ACCEPTANCE_TEMPLATE.md, phân biệt ca đã chạy với ca chưa chạy và bằng chứng từ máy Huy.

Việc gán 6 clip cần chính tôi xem/nghe và xuất JSON, không tự tạo nhãn. Trước khi tôi hoàn tất nhãn độc lập, có thể dùng fixture tổng hợp để kiểm thử, chưa hiển thị dự đoán trên pilot. Không chạy classifier trên test pilot 022–025. Sau khi đã gán nhãn, dùng train 005–009 để kiểm tra kỹ thuật; không dùng score làm gold.

Sửa lỗi thuộc apps/ và tests app khi cần, không sửa model/ngưỡng để làm kết quả đẹp. Hoàn thiện ảnh UI, hướng dẫn và kịch bản demo, chuẩn bị quay demo 3–5 phút theo docs/DEFENSE_NOTES.md. Lưu progress và commit, báo rõ kết quả, cách chạy và mục còn cần người thực hiện. Không tự gửi tin sang chat khác.
```
