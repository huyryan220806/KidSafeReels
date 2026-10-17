# Pipeline video trên CPU

Nhánh `feat/ml-video-pipeline` nối model thật vào `kidsafe.pipeline.analyze_video`. Luồng hiện tại: video → tối đa 8 frame lấy đều → ResNet18 ImageNet đóng băng → mean/max pooling → Logistic Regression Violence; audio → faster-whisper base đa ngôn ngữ → TF-IDF + Logistic Regression Toxic tiếng Việt. Hai nhánh phục vụ hai nhãn khác nhau; không lấy trung bình hai điểm.

## Cài đặt

Tại gốc repo, dùng Python 3.12 và môi trường ảo theo README:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-video.txt
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
```

Đã chọn CPU cho cả hai nhánh, ASR dùng int8, 4 luồng. Máy hiện tại có RTX 4060 Laptop nhưng thí nghiệm này không sử dụng GPU. PyAV kèm thư viện FFmpeg; không cần cài lệnh `ffmpeg` riêng. Ghim PyAV 16.1.0 vì faster-whisper 1.2.1 chưa tương thích với `av.open` của PyAV 19; có test giải mã WAV để bắt lỗi này. Cài gói video chỉ ở máy chạy API/model; máy chỉ chạy giao diện có thể dùng `requirements.txt`.

## Chuẩn bị model

Nếu tái tạo từ đầu, các lệnh sau có tải dữ liệu/trọng số qua Internet:

```powershell
.\.venv\Scripts\python.exe -m ml.toxic_baseline download
.\.venv\Scripts\python.exe -m ml.toxic_baseline train
.\.venv\Scripts\python.exe -m ml.video_assets
.\.venv\Scripts\python.exe -m ml.download_rlvs
.\.venv\Scripts\python.exe -m ml.train_visual
```

RLVS archive version 1 tải thực tế khoảng 3,6 GiB, chứa hai cây thư mục có bản trùng; cần khoảng 10 GB trống cho archive, giải nén, cache và model. Downloader ghi SHA-256; training loại trùng chính xác và gom clip gần giống trước chia tập. Có thể dùng lại cache đặc trưng khi chạy lại cùng backbone/sampling, nhưng train sẽ ghi đè classifier và report. Không thay cấu hình dựa trên test đã xem.

Whisper được ghim tại revision `ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66`. `ml.video_assets` giữ nguyên backbone local nếu đã có để không làm lệch checksum của classifier. Không có tải model ngầm trong request API.

Thành viên app có thể nhận bộ artifact local từ ML lead, không phải tải RLVS hay huấn luyện lại:

```powershell
# ML lead tạo gói local, không tải lên dịch vụ nào
.\.venv\Scripts\python.exe -m ml.package_models
# Máy nhận: đặt file ZIP đã nhận ở models/checkpoints rồi giải nén
Expand-Archive -LiteralPath models/checkpoints/kidsafe-cpu-v1.zip -DestinationPath models/checkpoints
```

Gói tạo tại `models/checkpoints/bundle/`, kèm SHA-256. Đối chiếu file bằng `Get-FileHash -Algorithm SHA256 <đường-dẫn-zip>` và `models/artifacts.json`. Chỉ nạp joblib của nhóm hoặc nguồn tin cậy. Không commit gói/checkpoint; không có video huấn luyện trong gói.

## Chạy video thật

```powershell
$env:KIDSAFE_BACKEND = "model"
.\.venv\Scripts\python.exe -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
# Terminal thứ hai
.\.venv\Scripts\python.exe -m streamlit run apps/web/app.py
```

Hoặc dùng CLI:

```powershell
.\.venv\Scripts\python.exe -m ml.inference C:\videos\sample.mp4 --output data/processed/sample-result.json
```

`KIDSAFE_MODEL_DIR` tùy chọn trỏ tới thư mục artifact khác. API mặc định vẫn mock nếu không đặt `KIDSAFE_BACKEND`. Đổi artifact phải khởi động lại API vì model được cache trong tiến trình. Dùng một worker cho demo; pipeline khóa inference để hạn chế bộ nhớ. `/health` chỉ báo API sống.

## Kết quả và lỗi

- Hỗ trợ video ≤60 giây, diện tích frame ≤3840×2160; API nhận ≤50 MiB. File hỏng/không đọc được/thời lượng vượt giới hạn trả 422 ở chế độ model. Chưa có timeout giải mã cứng, chưa phù hợp triển khai public.
- Violence là điểm toàn clip từ frame thưa. Evidence visual bao phủ `[0, duration]` và nói rõ chưa định vị cảnh; không diễn giải thành toàn bộ khoảng này đều có bạo lực.
- Toxic chỉ chạy khi có audio và ASR nhận diện tiếng Việt với điểm ngôn ngữ ≥0,5. Chỉ dùng đoạn có `no_speech_prob < 0,6`; điểm video là max điểm từng đoạn. Các bộ lọc ASR này là cấu hình ban đầu, chưa hiệu chỉnh trên video tiếng Việt. Timestamp đến từ ASR và được giới hạn trong clip, có thể sai lệch.
- Không tiếng/không nhận ra tiếng Việt/ASR lỗi: Toxic = `null`, kèm lý do. Lỗi ASR không làm mất điểm visual đã có. Sexual và Dangerous Acts luôn `null`.
- Vượt ngưỡng bất kỳ nhánh → `flagged`; còn lại → `review`. Chưa trả `no_risk_detected` ở bản này vì chưa đánh giá trên tập video đích. Score chưa được calibration.
- Thiếu artifact hoặc gói phụ thuộc → 503; không thay bằng kết quả mock. Không có báo cáo transcript đầy đủ, chỉ các đoạn bị đánh dấu xuất hiện trong evidence.

## Kiểm chứng và việc còn lại

```powershell
.\.venv\Scripts\python.exe -m pytest -q
# Tải một WAV FLEURS tiếng Việt có CC-BY-4.0, tạo video nền màu, chạy qua API local
.\.venv\Scripts\python.exe -m ml.smoke_video --prepare
```

Smoke test gọi FastAPI qua TestClient trong tiến trình, chưa kiểm thử trình duyệt hay kết nối mạng. Dùng 5 clip RLVS từ test (chọn theo thứ tự manifest, không chọn theo dự đoán), video không tiếng, video nền màu ghép một mẫu giọng nói FLEURS, video quá dài và file hỏng. Mẫu ghép chỉ kiểm tra đường đi ASR → Toxic; không dùng để đo độ chính xác moderation. JSON kết quả lưu `reports/video_smoke_results.json`.

Cần tiếp tục gán nhãn tập video ngắn tiếng Việt để đo hai lớp trên cùng video, ảnh hưởng ASR, visual-only/text-only/kết hợp và độ trễ đại diện. Hai metric benchmark riêng chưa chứng minh hiệu quả hệ thống trên TikTok/Reels/Shorts. App owner tiếp tục playback/evidence, kiểm thử UI và cài trên máy thứ hai.

Nguồn: [RLVS](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset), [ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [FLEURS](https://huggingface.co/datasets/google/fleurs).
