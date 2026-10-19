# Nghiệm thu kỹ thuật trên máy Lê Quyết Tiến - 18/10/2026

Sinh viên app: **Lê Quyết Tiến**, MSSV **2474802010386**, lớp **03_02**. Giảng viên: **Nguyễn Thái Anh**. Thông tin do sinh viên xác nhận trong phiên làm việc.

## Môi trường và nguồn bàn giao

| Mục | Kết quả thực tế |
|---|---|
| Máy | Windows 11, Intel Core i5-12450H, RAM khả dụng hệ thống 15,71 GiB |
| So với máy Huy | Máy khác với AMD Ryzen 5 7535HS của Huy, không phải venv mới trên cùng máy Huy |
| Python / môi trường | 3.12.5, tạo mới `.venv-model` |
| Code đầu vào | Nhánh app d82798d nằm trong nhánh bàn giao feat/ml-release f5d9310; ghép fast-forward lên feat/app |
| ZIP ngoài | SHA-256 `617b111e93ae0a8adf376b7fa9ee83df096388c04318879056b30b207f28e519`, khớp file Huy gửi |
| Nội dung ZIP | 17 tệp khớp kích thước và SHA-256 theo HANDOFF_CONTENTS.json |
| Model | kidsafe-cpu-v1, **8/8 artifact** khớp manifest trong repo |
| Thư viện | PyAV 16.1.0, faster-whisper 1.2.1, torch 2.14.1+cpu, torchvision 0.29.1+cpu, Streamlit 1.66.0 |
| Backend thực tế | `model`, xác minh từ kết quả phân tích qua HTTP và giao diện tự động |

Không thay schema, nhãn, ngưỡng hoặc trọng số. Dataset/checkpoint, log đầy đủ, transcript và JSON kết quả chi tiết được giữ local ở `data/private/` và `models/checkpoints/`.

## Kết quả đã chạy trên máy này

| Hạng mục | Trạng thái | Bằng chứng / giới hạn |
|---|---|---|
| Cài môi trường CPU sạch | Đạt | requirements-release.lock.txt làm constraints, PyTorch lấy CPU index |
| `pip check` | Đạt | No broken requirements found |
| `ml.release_verify` | Đạt | kidsafe-cpu-v1, verified_files: 8 |
| `pytest -q` | **118 passed**, 0 failed, 0 skipped | 37,046 giây; reports/app_test_summary.json |
| HTTP qua worker/model thật | **13/13 qua** | reports/app_acceptance_tien_results.json; video tổng hợp và mẫu speech FLEURS, không phải metric ML |
| UI AppTest nối HTTP/model thật | **9/9 qua** | reports/app_ui_acceptance_tien.json; không mock response |
| Hai modality | Đạt trong kiểm thử vận hành | Video nền tổng hợp + audio tiếng Việt FLEURS 14,94 giây có cả score Violence và Toxic |
| Không tiếng / Sexual / Dangerous | Đạt | Toxic null với video không tiếng; hai nhãn ngoài P0 luôn null |
| Preview / evidence start time | Đạt ở widget | Widget video có dữ liệu; evidence chọn mốc cho start_time=4 giây. Chưa xác nhận hình ảnh/phát lại qua trình duyệt bằng tay |
| TXT / JSON | Đạt ở widget và dữ liệu sinh ra | Có 2 nút download với URL media; tạo lại TXT/JSON từ đúng kết quả hiện tại. Chưa kiểm download bằng trình duyệt ở bản model |
| Đổi file / rerun | Đạt | ID kết quả giữ qua rerun, đổi file xóa kết quả và nút tải cũ |
| File rỗng / hỏng / quá 60 giây | Đạt | HTTP 400 / 422 / 422; file hỏng cũng hiển thị lỗi qua AppTest |
| Sai đuôi / sai container | Đạt | HTTP 415 |
| Thiếu model | Đạt ở worker thật | KIDSAFE_MODEL_DIR trỏ thư mục trống riêng, nhận 503, không fallback mock |
| API không kết nối | Đạt ở AppTest | URL loopback không có listener, UI hiện thông báo kết nối |
| Busy / timeout | Đạt trong test tự động | 429, 504 và dừng worker được bao phủ bởi pytest; không ghi là đã thử thủ công |
| 5 clip pilot train 005-009 | Chưa chạy trên máy này | Chờ người app hoàn tất gán nhãn độc lập, chưa gọi classifier trên pilot nào |
| Gán 6 clip / kiểm JSON | Chưa hoàn tất | Player đã giải nén và mở; chưa có annotations-app.json nộp |
| Ảnh UI model và quay màn hình 3-5 phút | Chưa thực hiện | Công cụ browser từ chối quyền truy cập trang xem trước; không tìm cách vòng qua. Có kịch bản và slide, cần người dùng thao tác/ghi hình |

## Thời gian đo qua HTTP

| Fixture | Thời lượng | Lần gọi | Thời gian request |
|---|---:|---|---:|
| silent.mp4, hình khối tổng hợp | 2 giây | Đầu tiên, gồm nạp worker/model | 7,2651 giây |
| tone.mp4, âm sine | 2 giây | Worker đã nạp | 1,2164 giây |
| video.mov | 2 giây | Worker đã nạp | 0,2460 giây |
| video.webm | 2 giây | Worker đã nạp | 0,2575 giây |
| boundary.mp4 | 60 giây, 2 fps | Worker đã nạp | 0,3059 giây |
| FLEURS tiếng Việt ghép nền tổng hợp | 14,94 giây | Worker đã nạp | 1,8200 giây |
| Cùng fixture tiếng Việt | 14,94 giây | Chạy lại một lần | 1,7737 giây |

Mỗi hàng là một lượt đo, kích thước hình rất nhỏ. Đây là độ trễ vận hành của các fixture, không đại diện video 1080p/4K hoặc Internet. Không so trực tiếp với clip pilot 25 giây trên máy Huy. Không có metric accuracy/F1 pilot mới.

## Lỗi đã sửa

Worker trước đó gộp `InvalidMediaError` từ nhánh model vào lỗi hệ thống 500. App nay ánh xạ đúng sang 422, kèm kiểm thử regression qua process thật. Giao diện cũng ghi rõ khi kết quả đến từ bộ phân tích thật. Không chỉnh classifier để làm kết quả tốt hơn.

## Chạy lại

```powershell
cd D:\KidSafeReels
powershell -ExecutionPolicy Bypass -File scripts/run-model.ps1
```

Hoặc mở `Chay-KidSafeReels.cmd`. Trình duyệt: http://127.0.0.1:8501. Dừng bằng Ctrl+C trong terminal app.

Terminal thứ hai, chỉ dùng fixture không phải pilot:

```powershell
.\.venv-model\Scripts\python.exe -X utf8 -m ml.smoke_video --prepare-only
.\.venv-model\Scripts\python.exe -X utf8 -m scripts.acceptance_model --run-dir data/private/app-acceptance-tien-new --machine "Tien, Intel Core i5-12450H"
.\.venv-model\Scripts\python.exe -X utf8 -m scripts.acceptance_ui --run-dir data/private/app-acceptance-tien-new
```

**Cập nhật công cụ 19/10:** các lệnh chạy lại dùng thư mục mới (đổi tên nếu đã tồn tại), ghi `http-summary.json` và `ui-summary.json` tại `--run-dir`. Dữ liệu công bố ngày 18/10 bên trên giữ nguyên. Script không còn tự ghi đè báo cáo máy Tiến khi chạy trên máy khác. Không chạy `ml.release_smoke` để ghi đè số đo máy Huy.

## Phần người app thực hiện tiếp

Mở `data/private/pilot-v1/annotation/review-app.html`, gán độc lập 001, 005, 011, 016, 022, 023 rồi xuất JSON. Có thể dùng `scripts/open-annotation.ps1`. Giữ file gốc và đặt tại `data/private/pilot-v1/submissions/annotations-app.json`, chạy kiểm tra cấu trúc:

```powershell
.\.venv-model\Scripts\python.exe -m ml.pilot_annotation check --annotator app --input data/private/pilot-v1/submissions/annotations-app.json
```

Sau khi hoàn tất nhãn độc lập, thử kỹ thuật 005-009 nếu cần. Không chạy classifier trên pilot test 022-025 để chọn cấu hình/clip demo. Huy đối chiếu hai phiếu, phân xử cùng Tiến rồi chạy đánh giá cuối.

**Kết luận:** đã nghiệm thu kỹ thuật cài môi trường, artifact, API và luồng UI tự động trên máy thứ hai. Nghiệm thu thủ công trình duyệt, gán nhãn pilot, metric chung và bản quay demo cuối vẫn chờ người thực hiện. Không đánh dấu toàn đồ án hoàn thành khi các mục này còn thiếu.
