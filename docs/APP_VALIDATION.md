# Kiểm chứng phần app — 05/10/2026

Phạm vi: app trên nhánh `feat/app`, không có model thật. Máy kiểm tra: Windows 11, Python 3.12.5, Intel Core i5-12450H. Phiên bản thư viện được ghi trong `requirements-app.lock.txt`.

## Kết quả đã chạy

| Kiểm tra | Kết quả thực tế |
|---|---|
| `python -m pytest -q` | **73 passed**, 20,58 giây ở lượt cuối |
| `python -m pip check` | Không có dependency bị xung đột |
| `git diff --check` | Qua |
| `scripts/setup.ps1` | Chạy thành công trên môi trường máy hiện tại |
| `scripts/run.ps1` / `python -m scripts.run_app` | Web/API khởi động được; đã kiểm tra dừng và khởi động lại |
| HTTP smoke qua server thật | **11/11 passed**, dữ liệu chi tiết tại `app-smoke.json` |
| Trình duyệt thật | Upload MP4, preview, phân tích mock, tải JSON và kiểm tra nội dung file tải về; đổi file reset kết quả; video 61 giây hiện lỗi |
| Bố cục | Đã xem desktop 1366×900 và màn hình nhỏ 390×844; không có tràn ngang ở 390px |

Pytest còn một deprecation warning từ Starlette TestClient về HTTPX; không có test lỗi. Đây là warning của công cụ kiểm thử, không phải lỗi trả về người dùng.

Ảnh chụp: `screenshots/app-empty.jpg`, `app-mock.jpg`, `app-too-long.jpg`, `app-mobile.jpg`. Video trong ảnh là hình khối tổng hợp, luôn có nhãn mock, không có điểm ML giả.

## Các tình huống được bao phủ

| Tình huống | Bằng chứng |
|---|---|
| MP4/MOV/WebM thật | PyAV decode test và HTTP smoke |
| Có audio/không audio | Decoder/worker test; warning trên UI với file không audio |
| Đúng 60 giây / quá 61 giây | Decoder, API test, HTTP smoke; lỗi 61 giây kiểm tra thêm trên trình duyệt |
| File giả/hỏng, file rỗng, đuôi/container sai | Decoder/API test, HTTP smoke |
| Vượt giới hạn kích thước, thiếu trường file | API test; missing-file HTTP smoke |
| Vượt giới hạn số khung/độ phân giải | Decoder test với giới hạn thu nhỏ có chủ đích |
| Worker bận | Test trả 429 không xếp hàng, API có Retry-After |
| Media/model timeout | Test chấm dứt process, lượt sau tạo worker mới |
| Worker crash | Test trả lỗi và thu hồi process |
| Model chưa có / cấu hình backend sai | Test trả 503, không fallback mock |
| Null và điểm 0 khác nhau | UI test dùng kết quả tổng hợp chỉ trong test |
| Evidence và nhảy mốc video | UI test chọn mốc và kiểm tra start_time; chưa nghiệm thu evidence từ model thật |
| Evidence ngoài clip/không có audio nguồn | Test trả 502 |
| Score/schema bị mutate sau khi tạo object | Test bắt lỗi ở ranh giới app, không tin sẵn object Pydantic |
| Đổi file / rerun / request lỗi sau thành công | UI test; đổi file và tải JSON kiểm tra thêm trên trình duyệt |
| Kết nối lỗi, timeout HTTP, lỗi server, JSON sai | UI test với lỗi có kiểm soát, không lưu kết quả cũ |
| Dọn file tạm khi thành công/lỗi | API test kiểm tra cả file và thư mục đã bị xóa |

Các test dùng `mode=model` chỉ kiểm tra giao tiếp qua fixture có ghi rõ “DỮ LIỆU KIỂM THỬ TỔNG HỢP”. Không có mock trá hình hoặc test backend trên ứng dụng chạy thật.

## Đo thời gian app

- Trên trình duyệt, lượt MP4 tổng hợp 2 giây đầu tiên sau khi khởi động worker: API 0,4407 giây, UI hiển thị toàn lượt khoảng 0,50 giây. File JSON tải về đã được kiểm tra.
- HTTP smoke tiếp theo dùng worker đã khởi động: năm video hợp lệ có thời gian HTTP khoảng 0,009–0,039 giây (xem giá trị chính xác trong JSON). Video 160×96; clip 60 giây chỉ 2 fps.
- Đây là số đo **giải mã + mock**, không phải latency model, không đại diện video 1080p/4K hoặc trải nghiệm Internet. Chưa đo p95 model hay chất lượng phân loại.

## Tái hiện

```powershell
cd D:\KidSafeReels
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.make_demo_videos
powershell -ExecutionPolicy Bypass -File scripts/run.ps1
```

Terminal thứ hai:

```powershell
cd D:\KidSafeReels
.\.venv\Scripts\python.exe -m scripts.smoke_api
```

Kết quả smoke mới mặc định lưu `data/private/app-demo/smoke.json` để không ghi đè bản bằng chứng đã lưu. Sau khi ghép model, dùng checklist tình huống trên để kiểm tra thủ công với video thật có quyền sử dụng.

## Chưa thể nghiệm thu tại máy này

- Tích hợp/đánh giá bằng model thật và tối thiểu 5 clip: chờ Huy bàn giao.
- Thiếu lời nói/ASR lỗi dù có audio track: logic ML chưa có.
- Cài mới và demo trên máy thành viên thứ hai: cần thực hiện trên máy đó; không coi kiểm thử trên máy hiện tại là đã đạt.
- Docker/Compose và GitHub Actions: đã có cấu hình, chưa chạy ở đây; máy chưa có Docker, nhánh chưa được push để chạy CI.
- Deploy công khai và quay demo cuối với model: chưa thực hiện; local là bản dùng ngay. Đã có kịch bản demo và ảnh app.

## Các nhóm file thay đổi

- `apps/api/`: upload, kiểm tra media và quản lý worker.
- `apps/web/`, `.streamlit/`: UI, HTTP client, theme, preview/evidence/download.
- `apps/validation.py`: bảo vệ contract ở phía app; schema dùng chung không đổi.
- `tests/`: API, media, worker, contract, UI và tích hợp.
- `scripts/`: setup/run, video tổng hợp và HTTP smoke.
- `requirements*`, `.github/workflows/ci.yml`, `Dockerfile`, `compose.yaml`, `deploy/`: môi trường/chạy app.
- `README.md`, `docs/`: hành vi thực tế, hướng dẫn sử dụng, bàn giao và bằng chứng kiểm tra.

Không sửa `kidsafe/`, `ml/`, model artifacts hoặc báo cáo metric ML của Huy. Bản ghi ngày 05/10 chưa được push. Ngày 16/10 chuẩn bị bàn giao qua nhánh `feat/app`; không merge vào main.
