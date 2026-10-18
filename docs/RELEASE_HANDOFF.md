# Bàn giao bản tích hợp CPU - 18/10/2026

Nhánh `feat/ml-release` ghép ML/pilot từ `32d1ad4` với app `d82798d`. Giữ nguyên schema v1 và signature `analyze_video`; giữ giao diện/worker của app owner. **117 tests pass, 13/13 ca HTTP với model thật pass**, 8 artifact khớp checksum. Đây là bản tích hợp để demo và hoàn thiện nhãn, chưa phải bản đã chứng minh hiệu quả kiểm duyệt trẻ em.

## Chạy trên máy Huy

```powershell
.\.venv\Scripts\python.exe -m ml.release_verify
$env:KIDSAFE_BACKEND = 'model'
.\.venv\Scripts\python.exe -m scripts.run_app
```

Web `http://127.0.0.1:8501`, API `http://127.0.0.1:8000/docs`. Restart API sau khi đổi artifact/code. `.env.example` chỉ là tài liệu; chương trình không tự nạp `.env`. Để app trên máy này chạy model, cần đặt biến backend ở chính terminal khởi động.

## Chuẩn bị máy thành viên app

Lưu/commit công việc đang làm trước khi chuyển nhánh. Clone mới hoặc fetch nhánh tích hợp; không reset nhánh app.

```powershell
git clone --branch feat/ml-release https://github.com/huyryan220806/KidSafeReels.git
cd KidSafeReels
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-pilot.txt
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
```

Gửi gói local `models/checkpoints/bundle/kidsafe-cpu-v1.zip` cho bạn cùng nhóm, không đưa raw dataset lên Git. Sau khi nhận ZIP, cài vào repo bằng lệnh dưới với đường dẫn thực tế:

```powershell
.\.venv\Scripts\python.exe -m ml.release_verify --install-bundle D:/duong-dan/kidsafe-cpu-v1.zip
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
$env:KIDSAFE_BACKEND = 'model'
.\.venv\Scripts\python.exe -m scripts.run_app
```

Bộ cài kiểm tra danh sách file, kích thước và SHA-256 theo manifest trong repo trước khi giải nén. Không tin manifest tự khai trong ZIP. Không nạp joblib từ nguồn khác. Không cần tải/huấn luyện lại khi có đủ 8 artifact.

`requirements-release.lock.txt` ghi môi trường đã kiểm thử trên Windows/Python 3.12 của Huy; dùng làm constraints khi cần tái tạo đúng phiên bản, kèm CPU index cho torch. Không xem việc tạo venv trên cùng máy là nghiệm thu máy thứ hai.

## Những điểm đã giải quyết khi tích hợp

- PyAV ghim 16.1.0 cho cả app và ML vì faster-whisper 1.2.1 dùng API mà PyAV 19 đã bỏ. App vẫn qua bộ test media/worker trên 16.1.0. Lock app được sửa đúng một phiên bản này; nhánh app remote giữ nguyên lịch sử.
- Hai test “model chưa có” dùng thư mục artifact trống riêng, không còn phụ thuộc việc máy Huy đã có checkpoint. Không làm yếu kiểm tra chống fallback.
- Worker giữ model trong tiến trình spawn, một request tại một thời điểm. Media timeout 30 giây, model 180 giây, web 225 giây; không cần tăng timeout trong 13 ca smoke.
- `flagged` khi ít nhất một nhãn đạt ngưỡng, còn lại `review`; pipeline v1 chưa trả `no_risk_detected`. Thiếu Toxic vẫn có thể flagged bởi Violence. Sexual/Dangerous luôn null.
- Decode/app và sampling/model vẫn riêng, chưa thay schema để chia sẻ buffer. Image Docker hiện chỉ dành cho app/mock; chưa nghiệm thu Docker với model.

## Kết quả thật và phần cần người thực hiện

[HTTP smoke](../reports/release_smoke_results.json): 5 clip train thật 25 giây qua HTTP/worker/model, hai nhánh có score; clip không tiếng giữ Toxic null. Các ca file hỏng/rỗng/dài/sai container/đuôi được từ chối đúng. Năm clip đều thuộc nhóm tin bưu điện Thanh Hóa, không đại diện bối cảnh video đích. Public report không chứa score/transcript để tránh làm lệch nhãn; JSON đầy đủ ở thư mục private.

Lần gọi đầu 23,34 giây gồm nạp worker/model. Bốn clip ấm có median 5,20 và p95 mô tả 11,95 giây; mỗi clip chỉ chạy một lần. Không dùng số này làm cam kết hiệu năng.

Huy còn gán 25 clip; app owner gán 6 và thử cài/chạy trên máy thứ hai. Sau đó Huy chạy [quy trình đánh giá](EVALUATION_GUIDE.md), bổ sung metric vào báo cáo/slide rồi cả nhóm diễn tập và quay video demo. Các phần này chưa được đánh dấu hoàn thành.
