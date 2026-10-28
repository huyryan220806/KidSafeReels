# Bàn giao CPU v3 — PhoBERT + R3D-18

Bundle mới dùng PhoBERT-base fine-tune (epoch 4/ngưỡng 0,78), R3D-18 Kinetics400 đóng băng + LR (ngưỡng 0,41), Whisper small beam5 giữ nguyên. Toàn bộ inference trong app chạy CPU; không cần NVIDIA trên máy Tiến. Code giữ schema v1, xử lý null và evidence cấp clip. Sexual/Dangerous chưa hỗ trợ.

## Kết quả để trình bày đúng

Toxic test F1 44,92% → 53,39%, FP 74 → 53, FN 56 → 50. Violence RLVS test F1 95,98% → 97,97%, FP 9 → 3, FN 7 → 5. AIRTLab nhãn nguồn chỉ đạt 53% Accuracy, còn 39 false positive/50 bình thường. Toxic loss tăng và có quá khớp. [Báo cáo đầy đủ, đồ thị và CSV](../reports/MODEL_IMPROVEMENT_2026-10-28.md). Chưa có metric toàn pipeline trên pilot tiếng Việt có nhãn.

## Mở trên máy Huy

Sau khi v3 được kích hoạt trong `models/artifacts.json`, nhấp đúp **Chay-KidSafeReels.cmd**, hoặc:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run-model.ps1 -OpenBrowser
```

Web: http://127.0.0.1:8501 · API: http://127.0.0.1:8000/docs. Upload video rồi bấm **Kiểm tra video**. Giao diện kết quả phải có chuỗi `violence-temporal-model-improvement-2026-10-28+toxic-phobert-model-improvement-2026-10-28+whisper-small-beam5`. Dừng phiên API cũ bằng Ctrl+C trước khi đổi model; mở lại launcher khi phiên cũ còn chạy chỉ dùng lại phiên đó, không tự reload trọng số.

## Cài hoặc cập nhật máy app

Nhận code `feat/ml-release` và ZIP `models/checkpoints/bundle/kidsafe-cpu-v3.zip` từ Huy. Chỉ `git pull` chưa có weights, vì checkpoint/ZIP không nằm trên GitHub. Không gửi raw dataset hoặc thư mục Drive cho máy app.

ZIP: **1.080.956.738 byte** (~1,01 GiB), 17 tệp được kiểm bằng manifest; SHA-256:

```text
d5b57016ebd150fd3a663227b97cdf1a96447e9b71d0a4cbdf4622faee7c09aa
```

```powershell
# Trong repo code mới; thay đường dẫn ZIP bằng đường dẫn nhận thực tế.
powershell -ExecutionPolicy Bypass -File scripts/setup-model.ps1 -ModelBundle "D:\BanGiao\kidsafe-cpu-v3.zip"
powershell -ExecutionPolicy Bypass -File scripts/run-model.ps1 -OpenBrowser
```

Setup cài môi trường `.venv-model`, CPU Torch, Transformers 4.57.1/PyVi và Hugging Face Hub 0.36.0 tương thích với cả PhoBERT và Whisper. Khóa trong `requirements-release.lock.txt`; không dùng lock v2 cũ khi cài v3. Tạo môi trường mới cần Internet để cài thư viện; inference sau đó dùng weights local, không tải model ngầm.

Nếu máy Huy dùng `.venv` đã cài sẵn và chưa có `.venv-model`, launcher tự chọn `.venv`. Không cài môi trường CUDA `.venv-train` để chạy app.

## Kiểm tra lại trên máy Tiến

Giữ app model đang chạy rồi dùng một thư mục kết quả mới:

```powershell
.\.venv-model\Scripts\python.exe -m pip check
.\.venv-model\Scripts\python.exe -m ml.release_verify
.\.venv-model\Scripts\python.exe -m pytest -q
.\.venv-model\Scripts\python.exe -m scripts.acceptance_model --run-dir data/private/acceptance-v3-tien --machine "Tien / CPU v3"
.\.venv-model\Scripts\python.exe -m scripts.acceptance_ui --run-dir data/private/acceptance-v3-tien
```

Fixture speech cần được chuẩn bị theo script `ml.smoke_video --prepare-only` nếu chưa có. Các ca acceptance chỉ kiểm kỹ thuật với video tổng hợp/FLEURS, không phải gán nhãn hoặc đo độ chính xác nội dung. Chưa coi số đo máy Huy là kết quả máy Tiến. Không cần sửa endpoint hoặc trường API để dùng v3.

## Khôi phục v2 khi cần

Dừng app. Chọn Python đúng môi trường đang chạy; trên máy Huy là `.venv`, trên máy Tiến thường `.venv-model`. Khôi phục manifest active cùng ZIP cũ:

```powershell
.\.venv\Scripts\python.exe -m ml.release_verify --install-bundle models/checkpoints/bundle/kidsafe-cpu-v2.zip --manifest models/artifacts-v2.json
Copy-Item -LiteralPath models/artifacts-v2.json -Destination models/artifacts.json
.\.venv\Scripts\python.exe -m ml.release_verify
powershell -ExecutionPolicy Bypass -File scripts/run-model.ps1 -OpenBrowser
```

Phụ thuộc CPU mới vẫn nạp được hai classifier v1 của v2; không cần gỡ PhoBERT. `requirements-release-v2.lock.txt` lưu môi trường cũ để đối chiếu. Sau rollback, thay đổi manifest là local có chủ đích, không commit/push nhầm như một bản phát hành mới. Muốn trở lại v3 thì dùng ZIP/manifest v3 tương ứng.

Trạng thái kiểm thử và activation được ghi trong [integration checks](../reports/model-improvement-2026-10-28/integration_checks.json) và [progress](PROGRESS.md). Tệp bàn giao này không gửi tin nhắn tự động cho Tiến.
