# KidSafeReels

Đồ án môn **Học máy và ứng dụng**. KidSafeReels giúp phụ huynh kiểm tra video ngắn trước khi cho trẻ xem. App nhận MP4, MOV và WebM tối đa **60 giây, 50 MiB**, kiểm tra dấu hiệu bạo lực và lời nói độc hại tiếng Việt, rồi chỉ ra các đoạn cần xem lại khi có bằng chứng.

## Chạy trên Windows

Dùng **Python 3.12** và mở PowerShell tại thư mục dự án.

```powershell
git clone https://github.com/huyryan220806/KidSafeReels.git
cd KidSafeReels
```

**Dùng thử giao diện** bằng backend mô phỏng. Chế độ này kiểm tra luồng tải video và hiển thị kết quả mẫu.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
powershell -ExecutionPolicy Bypass -File scripts/run.ps1
```

**Chạy model thật** bằng gói `kidsafe-cpu-v3.zip` do nhóm cung cấp riêng. Thay đường dẫn ví dụ bằng vị trí ZIP trên máy.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup-model.ps1 -ModelBundle "D:\BanGiao\kidsafe-cpu-v3.zip"
.\Chay-KidSafeReels.cmd
```

Web ở [127.0.0.1:8501](http://127.0.0.1:8501) và API ở [127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Chọn video, bấm **Kiểm tra video**, rồi xem kết quả và các đoạn được chỉ ra. Nhấn **Ctrl+C** trong terminal đã khởi động app để dừng.

Giao diện gia đình có thể tải bản tóm tắt TXT. Để xem điểm, cảnh báo kỹ thuật và tải JSON, dừng phiên đang chạy rồi mở [lối chạy quản trị](scripts/Chay-KidSafeReels-QuanTri.cmd) trong thư mục `scripts/`. Đây là công cụ chẩn đoán trên máy cá nhân. Xem [hướng dẫn sử dụng và xử lý sự cố](docs/APP_GUIDE.md).

**Chạy bản mô phỏng bằng Docker Compose** từ thư mục gốc của repo.

```powershell
docker compose -f deploy/compose.yaml up --build
```

## Model và kết quả thử nghiệm

- App dùng gói CPU v3 với R3D-18 cho bạo lực, Whisper small để nhận dạng lời nói, và PhoBERT kết hợp luật từ/cụm từ cho Toxic. Xem [hướng dẫn model v3](docs/MODEL_UPDATE_2026-10-28.md) và [model card](reports/MODEL_CARD.md).
- Thử nghiệm Toxic ngày 20/11 trên **28 clip validation** đạt **F1 92,86%**, với 0 báo nhầm và 2 bỏ sót, khi kết hợp bộ phân loại ký tự trên toàn transcript với luật. Ứng viên này cần được đánh giá trên video mới trước khi tích hợp vào app. Xem [kết quả và cách tái hiện](reports/toxic-transcript-2026-11-20/README.md).
- Kết quả hỗ trợ người lớn xem lại video. Nội dung gợi dục và hành vi nguy hiểm hiện ở trạng thái chưa kiểm tra.

## Cấu trúc repo

| Thư mục | Nội dung |
| --- | --- |
| `apps/`, `kidsafe/` | FastAPI, Streamlit, schema và pipeline |
| `ml/`, `models/` | Mã ML, hướng dẫn và checksum của gói model |
| `data/` | Hướng dẫn dữ liệu, manifest mẫu và split dùng để tái hiện |
| `docs/`, `docs/assets/` | Hướng dẫn chạy, hợp đồng API, ảnh giao diện và tài liệu dự án |
| `reports/`, `output/` | Báo cáo kỹ thuật, số liệu và tài liệu trình bày chung |
| `scripts/`, `tests/` | Cài đặt, chạy app và kiểm thử |
| `requirements/` | Phụ thuộc và các bản khóa phiên bản cho app và ML |
| `config/`, `deploy/` | Mẫu cấu hình môi trường và cấu hình Docker |

Video, phiếu gán nhãn, transcript riêng và gói bàn giao lưu ở `data/private/`. Trọng số và bộ cài model lưu ở `models/checkpoints/`. Khi cần cấu hình app chạy trực tiếp, đặt biến môi trường trong terminal theo [mẫu cấu hình](config/.env.example).

## Phát triển

Chạy bộ kiểm thử app sau khi cài môi trường mô phỏng:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_api.py tests/test_media.py tests/test_run_app.py tests/test_validation.py tests/test_web.py tests/test_web_design.py tests/test_worker.py
```

CI cài thêm phụ thuộc ML và chạy toàn bộ bộ test trên Windows và Linux. Xem [hướng dẫn đóng góp](.github/CONTRIBUTING.md), [cấu hình CI](.github/workflows/ci.yml), [hợp đồng API](docs/API_CONTRACT.md), [hướng dẫn ML](ml/README.md) và [nhật ký dự án](docs/PROGRESS.md).

**Nhóm thực hiện:** Huy phụ trách ML, Lê Quyết Tiến phụ trách API, giao diện và tích hợp.
