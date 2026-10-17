# KidSafeReels

Mini project môn **Học máy và ứng dụng**: hỗ trợ người lớn sàng lọc nội dung video ngắn trước khi cho trẻ xem.

**Dữ liệu pilot D1–D2:** đã có 25 đoạn từ 8 nguồn có giấy phép, chia theo nhóm và tiền xử lý thật. Huy gán 25 clip, thành viên app gán chéo 6; nhãn đang chờ người xác nhận. Xem [progress để tiếp tục](docs/PROGRESS.md), [cách chạy lại/gói gán nhãn](docs/PILOT_DATA.md) và [kết quả D1–D2](reports/PILOT_D1_D2.md).

**Trạng thái 17/10/2026: đã có pipeline video thật cho Violence + Toxic tiếng Việt trên CPU.** Visual dùng ResNet18 đóng băng + classifier học trên RLVS; audio dùng Whisper base + baseline Toxic học trên ViCTSD. Xem [cài và chạy model video](docs/VIDEO_PIPELINE.md), [kết quả visual](reports/VISUAL_BASELINE.md), [kết quả Toxic](reports/TOXIC_BASELINE.md) và [kiểm thử tích hợp/ca lỗi](reports/VIDEO_SMOKE.md). Chưa đánh giá toàn hệ thống trên tập video ngắn tiếng Việt; Sexual/Dangerous chưa được hỗ trợ. Mock vẫn là mặc định để thành viên app chạy ngay.

## Phạm vi một tuần

- Web upload video MP4/MOV/WebM, mục tiêu tối đa 60 giây và 50 MiB.
- MVP học máy: hình ảnh + transcript tiếng Việt, ưu tiên bạo lực và ngôn ngữ độc hại; hiển thị lý do và mốc thời gian.
- Nhãn thiết kế: `violence`, `sexual`, `toxic_language`, `dangerous_acts` (đa nhãn). Nhãn chưa được hỗ trợ phải trả `null`.
- OCR, hai nhóm sexual/dangerous_acts và triển khai online là mở rộng sau khi lõi chạy ổn.
- TikTok/Reels/YouTube Shorts là bối cảnh sử dụng video, chưa có tích hợp chặn feed hay tải video trực tiếp từ nền tảng.

## Chạy bộ khung (Python 3.11 hoặc 3.12)

```powershell
git clone https://github.com/huyryan220806/KidSafeReels.git
cd KidSafeReels
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m uvicorn apps.api.main:app --reload
```

Mở terminal thứ hai tại thư mục repo:

```powershell
.\.venv\Scripts\python.exe -m streamlit run apps/web/app.py
```

Web: http://localhost:8501 · API docs: http://127.0.0.1:8000/docs · Health: http://127.0.0.1:8000/health

macOS/Linux: thay `.\.venv\Scripts\python.exe` bằng `.venv/bin/python`. Chế độ mock không cần GPU, FFmpeg hay tải model.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

`KIDSAFE_BACKEND=mock` là mặc định: luôn trả `review`, điểm `null`, không phân tích video. Đặt `KIDSAFE_BACKEND=model` sau khi chuẩn bị phụ thuộc và artifact theo [hướng dẫn model](docs/VIDEO_PIPELINE.md); thiếu artifact trả HTTP 503. `KIDSAFE_API_URL` cấu hình địa chỉ API cho web, mặc định `http://127.0.0.1:8000`. File `.env.example` là tài liệu cấu hình, chưa được tự động nạp.

## Hai người bắt đầu từ đâu?

| Người | Phụ trách | Đọc trước |
|---|---|---|
| Huy — leader / ML | Dữ liệu, model, fusion, đánh giá; quyết định phạm vi và tích hợp | [Kế hoạch tuần](docs/PLAN.md), [ML](ml/README.md) |
| Thành viên app | FastAPI, Streamlit, upload, hiển thị evidence, lỗi, chạy demo | [Bàn giao app](docs/HANDOFF_APP.md), [API contract](docs/API_CONTRACT.md) |

```text
apps/api/       API upload và gọi pipeline (app owner)
apps/web/       Giao diện Streamlit (app owner)
kidsafe/        Schema dùng chung + giao diện pipeline (hai người thống nhất)
ml/            Chỗ phát triển huấn luyện, trích đặc trưng, suy luận (ML lead)
data/          Hướng dẫn dữ liệu + manifest mẫu; không commit video
models/        Hướng dẫn checkpoint; không commit trọng số
reports/       Khung báo cáo và model card
docs/          Phạm vi, phân công, kiến trúc, hợp đồng và ảnh ý tưởng
tests/         Kiểm tra API và giao diện mock
```

Đọc [quy trình Git](CONTRIBUTING.md) và [hướng dẫn cho Codex](AGENTS.md) trước khi sửa. Chi tiết giới hạn hiện tại nằm ở [API contract](docs/API_CONTRACT.md). Model kiểm tra media/thời lượng và xử lý thiếu audio; còn cần tập video đích có nhãn, kiểm thử UI và cài lại trên máy thứ hai.

Ý tưởng gốc được giữ tại [Idea1.png](docs/ideas/Idea1.png) và [Idea2.png](docs/ideas/Idea2.png); xem [kiến trúc dự kiến](docs/ARCHITECTURE.md) để đối chiếu với phạm vi một tuần.
