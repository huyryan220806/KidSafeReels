# KidSafeReels

Mini project môn **Học máy và ứng dụng**: hỗ trợ người lớn sàng lọc nội dung video ngắn trước khi cho trẻ xem.

**Trạng thái hiện tại: bộ khung chạy được với API mô phỏng; chưa có model đã huấn luyện hay kết quả đánh giá.** Chế độ mock luôn trả `review`, điểm số `null`, không phân tích nội dung video. Không dùng kết quả này để đánh giá độ an toàn.

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

`KIDSAFE_BACKEND=mock` là mặc định. Đặt biến môi trường `KIDSAFE_BACKEND=model` sẽ trả HTTP 503 cho phân tích cho đến khi ML lead nối model thật. `KIDSAFE_API_URL` cấu hình địa chỉ API cho web, mặc định `http://127.0.0.1:8000`. File `.env.example` là tài liệu cấu hình, chưa được tự động nạp.

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

Đọc [quy trình Git](CONTRIBUTING.md) và [hướng dẫn cho Codex](AGENTS.md) trước khi sửa. Chi tiết giới hạn hiện tại nằm ở [API contract](docs/API_CONTRACT.md). Bộ khung chưa giải mã video, kiểm tra thời lượng hay chạy ASR/model; đây là hạng mục phải hoàn thành trong tuần triển khai.

Ý tưởng gốc được giữ tại [Idea1.png](docs/ideas/Idea1.png) và [Idea2.png](docs/ideas/Idea2.png); xem [kiến trúc dự kiến](docs/ARCHITECTURE.md) để đối chiếu với phạm vi một tuần.
