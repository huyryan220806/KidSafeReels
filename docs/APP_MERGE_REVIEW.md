# Kiểm tra và hợp nhất app — 19/10/2026

Đã nhận `origin/feat/app` tại `47326a6`, nối tiếp `feat/ml-release` tại `f5d9310`. Hợp nhất fast-forward vào `feat/ml-release`, giữ toàn bộ lịch sử app và ML. Không đổi schema v1, checkpoint, nhãn, ngưỡng hoặc split pilot. Nhánh `main` vẫn là bộ khung cũ; dùng `feat/ml-release` để lấy bản tích hợp này.

## Kết quả review và sửa trước bàn giao

- Giữ bản sửa của Tiến: lỗi `InvalidMediaError` từ model trả HTTP 422 và có regression test qua worker. Giao diện phân biệt kết quả model thật.
- Sửa launcher `scripts/run-model.ps1`: ưu tiên `.venv-model`, dùng `.venv` khi không có môi trường thứ nhất. Máy Huy đã có `.venv`, không cần tải/huấn luyện lại. Có thể chọn Python bằng `-PythonPath`; launcher kiểm và chạy cùng bộ checkpoint trong repo, tránh biến môi trường cũ trỏ sang bộ khác.
- Sửa công cụ nghiệm thu: trước đây `acceptance_model.py` cố định tên máy Tiến và hai script ghi đè báo cáo đã công bố khi chạy lại trên máy khác. Nay bắt buộc `--run-dir` riêng dưới `data/private`, không ghi đè lần hoàn tất và ghi đúng môi trường chạy. Giữ nguyên JSON kết quả Tiến ngày 18/10.

## Kiểm tra trên máy Huy

Windows 11, Python 3.12.10, AMD Ryzen 5 7535HS, CPU. Bằng chứng mới giữ riêng ở `data/private/app-merge-review-2026-10-19/`; bản tóm tắt công khai tại `reports/app_merge_review_2026-10-19.json`.

| Kiểm tra | Kết quả / phạm vi |
|---|---|
| Toàn bộ pytest | **118 passed**, 78,12 giây; JUnit local `runs/app-merge-review/pytest.xml` |
| Phụ thuộc / model | `pip check` pass; 8/8 artifact đúng checksum |
| HTTP với worker và model thật | **13/13 pass**; fixture tổng hợp và FLEURS tiếng Việt, không chạy pilot |
| Streamlit AppTest gọi HTTP/model thật | **9/9 pass**, 0 mục chưa chạy |
| Trình duyệt thật qua công cụ điều khiển | Upload FLEURS, preview, bấm kiểm tra, thấy kết quả model thật và hai nhãn đã đánh giá |
| Evidence trong trình duyệt | Player có `currentTime=4`, `readyState=4`, không lỗi media; kiểm mốc tải video, chưa xác nhận độ đúng nội dung ASR |
| Download TXT | Tải thành công `kidsafe-ket-qua.txt`, 1.212 byte, tên video và nội dung khớp kết quả vừa chạy |
| Đổi video | Xóa kết quả/nút tải cũ trước khi phân tích video tiếp theo |
| Video không tiếng | UI hiện Toxic chưa đánh giá, kết luận cần người lớn kiểm tra; có [ảnh chạy thật](screenshots/app-merged-model-20261019.png) |

FLEURS là fixture kiểm tra đường đi audio; ca ASR gây cảnh báo nhầm đã được ghi trong báo cáo ML. Các kiểm tra trên không phải đánh giá chất lượng nội dung hoặc nhãn gold. Browser do công cụ thao tác, không phải biên bản người dùng diễn tập. Download JSON được kiểm trong AppTest, chưa tải trực tiếp qua trình duyệt ở lượt này.

## Mở web trên máy Huy

1. Mở thư mục `D:\VLU\261\ML\BTL\KidSafeReels`.
2. Nhấp đúp **`Chay-KidSafeReels.cmd`**. Chờ kiểm checksum và dòng `Web: http://127.0.0.1:8501`.
3. Mở **http://127.0.0.1:8501**, chọn video rồi bấm **Kiểm tra video**. MP4/MOV/WebM, tối đa 60 giây và 50 MiB.
4. Giữ cửa sổ terminal mở khi dùng web. Ctrl+C trong terminal chạy app để dừng cả web/API.

Hoặc từ PowerShell:

```powershell
cd D:\VLU\261\ML\BTL\KidSafeReels
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run-model.ps1
```

Nếu báo cổng 8000/8501 đã dùng, mở địa chỉ web của phiên đang chạy hoặc dừng phiên cũ trước khi khởi động thêm. Chạy `scripts/run.ps1` thông thường có thể dùng mock; dùng launcher **run-model** ở trên để kiểm tra model thật. Không cần chạy setup lại trên máy Huy đã được kiểm chứng.

Video thử kỹ thuật có sẵn: `data/processed/smoke/silent.mp4` (không tiếng) và `data/processed/smoke/vietnamese-speech.mp4` (FLEURS, có ca ASR cảnh báo nhầm). Không dùng kết quả của fixture để suy ra chất lượng mô hình. Chưa xem dự đoán trên pilot trước khi hoàn tất nhãn độc lập; không mở test 022–025 để chọn cấu hình.

## Chạy lại kiểm thử tích hợp khi cần

Giữ app đang chạy, dùng terminal thứ hai. Chọn tên thư mục mới cho mỗi lượt:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.acceptance_model --run-dir data/private/app-review-new --machine "Huy, AMD Ryzen 5 7535HS, CPU"
.\.venv\Scripts\python.exe -X utf8 -m scripts.acceptance_ui --run-dir data/private/app-review-new
```

## Các mục chưa hoàn tất

Theo bàn giao của Tiến, mới hoàn tất nghiệm thu kỹ thuật tự động trên máy thứ hai; 6 nhãn app, nhãn Huy, metric pilot và video quay demo vẫn chưa hoàn tất. Lượt review này bổ sung kiểm tra trình duyệt trên máy Huy, không thay việc người dùng gán nhãn hoặc quay demo. PDF/slide bản `_Tien` được giữ như bàn giao; chưa cập nhật chúng bằng kết quả review 19/10.
