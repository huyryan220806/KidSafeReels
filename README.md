# KidSafeReels

Mini project môn **Học máy và ứng dụng**: hỗ trợ người lớn sàng lọc nội dung video ngắn trước khi cho trẻ xem.

**Dữ liệu pilot D1–D2:** đã có 25 đoạn từ 8 nguồn có giấy phép, chia theo nhóm và tiền xử lý thật. Huy gán 25 clip, thành viên app gán chéo 6; nhãn đang chờ người xác nhận. Xem [progress để tiếp tục](docs/PROGRESS.md), [cách chạy lại/gói gán nhãn](docs/PILOT_DATA.md) và [kết quả D1–D2](reports/PILOT_D1_D2.md).

**Trạng thái 19/10/2026: đã có pipeline video thật cho Violence + Toxic tiếng Việt trên CPU.** Visual dùng ResNet18 đóng băng + classifier học trên RLVS; audio dùng Whisper base + baseline Toxic học trên ViCTSD. Xem [cài và chạy model video](docs/VIDEO_PIPELINE.md), [kết quả visual](reports/VISUAL_BASELINE.md), [kết quả Toxic](reports/TOXIC_BASELINE.md) và [kiểm thử tích hợp/ca lỗi](reports/VIDEO_SMOKE.md). Chưa đánh giá toàn hệ thống trên tập video ngắn tiếng Việt; Sexual/Dangerous chưa được hỗ trợ. Mock vẫn là mặc định để thành viên app chạy ngay.

App đã ghép giao diện mới, worker có timeout, kiểm tra media và evidence từ nhánh app. Bản tích hợp có 13/13 ca HTTP/model thật pass. Xem [bàn giao release](docs/RELEASE_HANDOFF.md), [báo cáo kỹ thuật](reports/TECHNICAL_REPORT.md) và [quy trình đánh giá sau gán nhãn](docs/EVALUATION_GUIDE.md).

## Phạm vi một tuần

- Web upload video MP4/MOV/WebM, mục tiêu tối đa 60 giây và 50 MiB.
- MVP học máy: hình ảnh + transcript tiếng Việt, ưu tiên bạo lực và ngôn ngữ độc hại; hiển thị lý do và mốc thời gian.
- Nhãn thiết kế: `violence`, `sexual`, `toxic_language`, `dangerous_acts` (đa nhãn). Nhãn chưa được hỗ trợ phải trả `null`.
- OCR, hai nhóm sexual/dangerous_acts và triển khai online là mở rộng sau khi lõi chạy ổn.
- TikTok/Reels/YouTube Shorts là bối cảnh sử dụng video, chưa có tích hợp chặn feed hay tải video trực tiếp từ nền tảng.

## Chạy app (ưu tiên Python 3.12)

**Mở nhanh trên máy đã cài model:** nhấp đúp `Chay-KidSafeReels.cmd`, chờ terminal báo địa chỉ rồi mở http://127.0.0.1:8501. Chọn video → bấm **Kiểm tra video** → đọc kết quả. Giữ terminal mở khi dùng web; Ctrl+C để dừng. Xem [kiểm tra bản merge ngày 19/10](docs/APP_MERGE_REVIEW.md).

Máy thành viên app đã có gói model: mở `Chay-KidSafeReels.cmd` hoặc chạy `powershell -ExecutionPolicy Bypass -File scripts/run-model.ps1`. Lệnh này ưu tiên `.venv-model`, dùng `.venv` nếu máy đã cài model ở đó, kiểm checksum rồi khởi động backend model thật. Bản kiểm tra trên máy Tiến được ghi riêng tại [nghiệm thu máy app](docs/APP_ACCEPTANCE_TIEN.md).

Cài model trên máy mới, dùng ZIP `kidsafe-cpu-v1.zip` Huy gửi:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup-model.ps1 -ModelBundle "D:\BanGiao\kidsafe-cpu-v1.zip"
powershell -ExecutionPolicy Bypass -File scripts/run-model.ps1
```

Môi trường model dùng đúng `av==16.1.0`, PyTorch CPU và constraints bàn giao. `.venv-model/`, checkpoint và video được giữ local. Các lệnh dưới đây dành cho phát triển app/mock:

Windows, tại thư mục repo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
powershell -ExecutionPolicy Bypass -File scripts/run.ps1
```

Lệnh đầu tạo `.venv` và cài phiên bản thư viện đã khóa; lệnh sau chạy cả web và API. Mở http://127.0.0.1:8501. Nhấn Ctrl+C trong terminal chạy app để dừng cả hai. Các dịch vụ local chỉ lắng nghe trên máy hiện tại. Mở Codex/VS Code tại đúng thư mục `D:\KidSafeReels` khi làm phần app.

Hoặc chạy từng dịch vụ để phát triển:

```powershell
git clone --branch feat/ml-release https://github.com/huyryan220806/KidSafeReels.git
cd KidSafeReels
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -c requirements-app.lock.txt
.\.venv\Scripts\python.exe -m uvicorn apps.api.main:app --reload
```

Mở terminal thứ hai tại thư mục repo:

```powershell
.\.venv\Scripts\python.exe -m streamlit run apps/web/app.py
```

Web: http://localhost:8501 · API docs: http://127.0.0.1:8000/docs · Health: http://127.0.0.1:8000/health

macOS/Linux: thay `.\.venv\Scripts\python.exe` bằng `.venv/bin/python`; có thể chạy chung bằng `.venv/bin/python -m scripts.run_app`. PyAV wheel chứa thư viện FFmpeg để kiểm tra media; không cần cài executable FFmpeg riêng cho phần app. Không cần GPU, torch hay tải model để chạy mock.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

`KIDSAFE_BACKEND=mock` là mặc định: luôn trả `review`, điểm `null`, không phân tích video. Đặt `KIDSAFE_BACKEND=model` sau khi chuẩn bị phụ thuộc và artifact theo [hướng dẫn model](docs/VIDEO_PIPELINE.md); thiếu artifact trả HTTP 503. `KIDSAFE_API_URL` cấu hình địa chỉ API cho web, mặc định `http://127.0.0.1:8000`. File `.env.example` là tài liệu cấu hình, chưa được tự động nạp.


Giới hạn mặc định: 50 MiB, 60 giây, 8.294.400 pixel/khung, 18.000 khung; timeout kiểm tra media 30 giây, model 180 giây, web 225 giây. Một worker phục vụ một lượt; lượt đến khi worker bận nhận 429, không tự retry. Worker được dùng lại để Huy có thể cache model trong bộ nhớ; timeout chấm dứt worker và lượt sau tạo lại. Thay đổi code model cần khởi động lại API.

```powershell
# Tạo 8 file tổng hợp để thử app; không phải dữ liệu đánh giá ML.
.\.venv\Scripts\python.exe -m scripts.make_demo_videos
```

File nằm tại `data/private/app-demo/`, đã được gitignore. Xem [hướng dẫn sử dụng và demo](docs/APP_GUIDE.md), [bàn giao tích hợp cho Huy](docs/HANDOFF_ML_APP.md), [kiểm chứng app](docs/APP_VALIDATION.md).

## Chạy bằng container (tùy chọn)

```powershell
docker compose up --build
```

Mở http://127.0.0.1:8501. Compose gồm web, API và nginx giới hạn request body 52 MiB (gồm multipart); API vẫn giới hạn nội dung file 50 MiB. Chưa công bố ra Internet. Docker chưa được chạy kiểm chứng trên máy phát triển hiện tại vì chưa có Docker. Khi tích hợp model, Huy cần thêm dependencies/checkpoint tương ứng; image hiện tại chỉ gồm app và mock.

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

Đọc [quy trình Git](CONTRIBUTING.md) và [hướng dẫn cho Codex](AGENTS.md) trước khi sửa. Chi tiết giới hạn hiện tại nằm ở [API contract](docs/API_CONTRACT.md). Model kiểm tra media/thời lượng và xử lý thiếu audio; còn cần tập video đích có nhãn, đánh giá có nhãn và cài lại trên máy thứ hai.

Ý tưởng gốc được giữ tại [Idea1.png](docs/ideas/Idea1.png) và [Idea2.png](docs/ideas/Idea2.png); xem [kiến trúc dự kiến](docs/ARCHITECTURE.md) để đối chiếu với phạm vi một tuần.
