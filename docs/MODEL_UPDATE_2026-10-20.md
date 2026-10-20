# Cập nhật model CPU v2 — 20/10/2026

Bản `kidsafe-cpu-v2` dùng **Whisper small beam5, temperature=0**, giữ classifier Toxic v1 và Violence v1. Ứng viên classifier mới không đạt yêu cầu trên test, nên không đưa vào app. Giao diện và schema API v1 không đổi. [Báo cáo số đo](../reports/MODEL_IMPROVEMENT_2026-10-20.md), [quyết định release](../reports/model-improvement-2026-10-20/release_decision.json), [kiểm tra tích hợp](../reports/model-improvement-2026-10-20/integration_checks.json).

## Trên máy Huy

Sau khi kích hoạt, nhấp đúp `Chay-KidSafeReels.cmd`, mở http://127.0.0.1:8501. Khi kiểm tra video có tiếng Việt, phiên bản trong JSON chứa `whisper-small-beam5`. Dừng phiên app cũ trước khi đổi model vì worker giữ model trong bộ nhớ; mở lại một phiên đang chạy không tự nạp artifact mới.

Kiểm tra artifact đang cài:

```powershell
.\.venv\Scripts\python.exe -m ml.release_verify
powershell -ExecutionPolicy Bypass -File scripts/run-model.ps1
```

## Máy thành viên app

Lưu công việc local, nhận code mới của `feat/ml-release` và gói `kidsafe-cpu-v2.zip` cùng file SHA-256 từ Huy. Code trên GitHub không chứa trọng số. Gói cũ `kidsafe-app-handoff-2026-10-18.zip` chứa model v1 và tài liệu/slide lịch sử; không cài ZIP v1 với manifest đang trỏ v2.

Nếu đang ở `feat/ml-release` và working tree sạch:

```powershell
git pull --ff-only origin feat/ml-release
```

Nếu tiếp tục nhánh app có công việc riêng, fetch rồi merge `origin/feat/ml-release` vào nhánh đó sau khi lưu thay đổi; không reset hoặc force-push. Dừng app, sau đó cài gói mới:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup-model.ps1 -ModelBundle "D:\BanGiao\kidsafe-cpu-v2.zip"
powershell -ExecutionPolicy Bypass -File scripts/run-model.ps1
```

Gói nằm trên máy Huy tại `models/checkpoints/bundle/kidsafe-cpu-v2.zip`; checksum cạnh file. Bộ cài xác minh 9 tệp theo manifest Git trước khi cài. Không cần train hoặc tải RLVS/FLEURS trên máy app. Cần kiểm thử v2 trên máy Tiến; nghiệm thu v1 ngày 18/10 không tự áp dụng cho v2. Không tự gửi gói hoặc tin nhắn cho thành viên app.

## Khôi phục v1 khi cần

Giữ ZIP v1 và `models/artifacts-v1.json`. Dừng API/web trước khi khôi phục, dùng Python của môi trường đã cài model (`.venv-model` trên máy app hoặc `.venv` trên máy Huy):

```powershell
.\.venv\Scripts\python.exe -m ml.release_verify --manifest models/artifacts-v1.json --install-bundle models/checkpoints/bundle/kidsafe-cpu-v1.zip
Copy-Item -LiteralPath models/artifacts-v1.json -Destination models/artifacts.json
powershell -ExecutionPolicy Bypass -File scripts/run-model.ps1
```

Bộ cài xóa cấu hình inference v2 khi khôi phục v1 để không vô tình dùng Whisper small với manifest cũ. Muốn quay lại v2, cài ZIP với `--manifest models/artifacts-v2.json`, rồi chép manifest v2 thành `models/artifacts.json`, khởi động lại app.

## Phạm vi số đo và phần còn lại

- ASR test FLEURS 16 câu: WER 38,36% → 19,83%, CER 22,40% → 10,70%; small chậm hơn trong lượt đo. Chưa phải test hội thoại video mạng xã hội.
- Toxic v1: F1 test 44,92%, recall 48,62%; còn yếu. Nâng ASR không làm thay đổi metric classifier trên bình luận ViCTSD.
- Violence v1: F1 test RLVS 95,98%; vẫn có phản hồi bỏ sót trên video thực tế. Thử 24 frame chưa cải thiện test.
- Chưa có video lỗi người dùng được lưu, chưa có nhãn pilot. Hai người vẫn cần gán nhãn, đánh giá video đích và cập nhật báo cáo/slide cuối. Sexual/Dangerous tiếp tục `null`.
