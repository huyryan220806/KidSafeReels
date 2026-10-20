# Bàn giao cho thành viên app — 18/10/2026

**Cập nhật 20/10:** app đã merge và nghiệm thu v1 trên hai máy. Bước tiếp theo cho app là nhận [bản model CPU v2](MODEL_UPDATE_2026-10-20.md) và kiểm lại trên máy Tiến. Dùng code mới của `feat/ml-release` cùng `kidsafe-cpu-v2.zip`; gói tổng hợp 18/10 bên dưới chứa v1 và được giữ làm lịch sử. Nhãn chéo và demo vẫn cần người thực hiện.

Bản này thay hướng dẫn app cũ ngày 05/10 và 16/10. **Tiếp tục từ bản đã tích hợp**, ưu tiên nghiệm thu máy thứ hai, sửa lỗi phát sinh và demo. Phạm vi một tuần giữ Violence + Toxic tiếng Việt trên CPU.

## 1. Nhận phiên bản và gói đính kèm

- Repo: https://github.com/huyryan220806/KidSafeReels
- Nhánh bàn giao: `feat/ml-release`. Commit tích hợp `fdec72b`, commit ghi tiến độ `5691c05`; các commit tài liệu bàn giao tiếp theo không đổi model.
- App `d82798d` đã ghép cùng ML/pilot `32d1ad4`; giữ UI, worker và schema v1.0. Có checkpoint thật, không cần huấn luyện lại.
- Đã kiểm chứng trên máy Huy: **117 tests pass, pip check pass, 13/13 ca HTTP/model thật pass, 8 artifact khớp checksum**. Đây là bằng chứng trên máy Huy, chưa thay thế nghiệm thu máy app.
- Gói Huy gửi riêng: `kidsafe-app-handoff-2026-10-18.zip` cùng file `.sha256`. Bên trong có model ZIP, annotation ZIP, tài liệu và báo cáo/slide. Mã nguồn lấy từ GitHub.

| Thành phần trong gói | Dùng để làm gì |
|---|---|
| `kidsafe-cpu-v1.zip` (~170 MiB) | 8 artifact inference; cài bằng lệnh kiểm hash bên dưới |
| `kidsafe-pilot-v1-annotation.zip` (~61 MiB) | Video/player/phiếu trống; không chứa ASR hoặc dự đoán |
| `HANDOFF_CONTENTS.json` | Commit nguồn, danh sách file, kích thước và SHA-256 |
| `docs/` | Bàn giao, API contract, hướng dẫn nhãn, mẫu nghiệm thu và kịch bản demo |
| `output/` | Báo cáo PDF 9 trang, slide PPTX 11 trang |

Raw dataset huấn luyện, kết quả dự đoán private và nhãn người khác không nằm trong gói. Chưa tải gói lên dịch vụ ngoài; Huy gửi qua kênh hai người thống nhất.

## 2. Lấy code và cài trên Windows/Python 3.12

Để kiểm tra cài mới, clone vào một thư mục mới, không đụng checkout đang làm:

```powershell
git clone --branch feat/ml-release https://github.com/huyryan220806/KidSafeReels.git KidSafeReels-app-check
cd KidSafeReels-app-check
git switch -c feat/app-release-check
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-pilot.txt
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
```

Nếu thư mục/nhánh này đã tồn tại, dùng tên mới hoặc tiếp tục nhánh đã tạo; không reset hay force-push công việc cũ. Nếu có thay đổi app mới hơn `d82798d`, xem diff và trao đổi với Huy trước khi ghép. Tên thư mục giải nén dưới đây chỉ là ví dụ, thay bằng đường dẫn thật của gói đã nhận:

```powershell
.\.venv\Scripts\python.exe -m ml.release_verify --install-bundle "D:/BanGiaoKidSafe/kidsafe-cpu-v1.zip"
.\.venv\Scripts\python.exe -m ml.release_verify
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
$env:KIDSAFE_BACKEND = 'model'
.\.venv\Scripts\python.exe -m scripts.run_app
```

Kỳ vọng verifier báo `kidsafe-cpu-v1`, `verified_files: 8`. Mở web http://127.0.0.1:8501 và API http://127.0.0.1:8000/docs. Ctrl+C dừng cả hai. `/health` báo backend, chưa chứng minh checkpoint nạp thành công; cần gửi video hợp lệ và nhận `mode=model`.

**Lưu ý tích hợp:** giữ `av==16.1.0`; PyAV 19 không tương thích faster-whisper 1.2.1 trong pipeline này. `requirements-app.lock.txt` chỉ là lock app; môi trường ML tham chiếu nằm ở `requirements-release.lock.txt`. Không nâng phiên bản tùy ý nếu chưa kiểm thử lại. `.env.example` không tự được nạp; đặt biến môi trường trong terminal khởi động. Docker hiện chỉ gồm app/mock, không phải cách chạy model đã nghiệm thu.

## 3. Việc app cần làm, theo thứ tự

### A. Gán nhãn độc lập 6 clip

Giải nén `kidsafe-pilot-v1-annotation.zip`, giữ `media/` cạnh HTML, mở `review-app.html` bằng Chrome/Edge. Đọc [quy tắc nhãn](ANNOTATION_GUIDE.md), xem/nghe đủ các clip **001, 005, 011, 016, 022, 023**. Xuất `annotations-app.json` và giữ bản gốc. Không dùng Codex/model/ASR để điền nhãn, không xem phiếu Huy trước.

Ưu tiên hoàn tất bước này trước khi xem kết quả model trên clip pilot. Trong lúc chưa gán nhãn, vẫn có thể cài môi trường và thử video tổng hợp. **022/023 là test: chỉ gán nhãn, không dùng chúng để thử hoặc chỉnh model/UI theo kết quả phân loại.** Huy sẽ chạy đánh giá test sau khi chốt gold/cấu hình.

Sau khi tải JSON, lưu vào `data/private/pilot-v1/submissions/` và kiểm tra cấu trúc:

```powershell
.\.venv\Scripts\python.exe -m ml.pilot_annotation check --annotator app --input data/private/pilot-v1/submissions/annotations-app.json
```

Chỉ trao đổi nhãn giữa hai người sau khi cả hai đã hoàn tất độc lập. JSON để private, không commit lên Git. Huy phụ trách đối chiếu, phân xử cùng bạn, tạo gold và metric.

### B. Kiểm thử bản model trên máy của bạn

Tạo fixture khi cần bằng `python -m scripts.make_demo_videos` với Python trong `.venv`; file nằm ở `data/private/app-demo/`. Xem [mẫu nghiệm thu](APP_ACCEPTANCE_TEMPLATE.md). Không lấy kết quả trên máy Huy điền vào mẫu của máy bạn.

| Ca kiểm tra | Kỳ vọng cần ghi nhận |
|---|---|
| Upload video hợp lệ | Có preview, gửi phân tích thật, kết quả `mode=model` |
| 5 clip train 005–009 sau khi gán nhãn xong | Luồng đầu-cuối chạy; ghi score có/null, lỗi và thời gian, không coi là gold |
| Video không tiếng | Toxic null và cảnh báo phù hợp; không biến thành điểm 0 |
| Nhãn chưa hỗ trợ | Sexual/Dangerous hiện chưa đánh giá |
| Kết quả dưới ngưỡng | `review`, không diễn đạt “an toàn tuyệt đối” |
| Evidence khi có | Phát lại đúng khoảng; visual toàn clip không được gọi là cảnh định vị chính xác |
| TXT/JSON và đổi video | Tải đúng kết quả hiện tại; đổi file không giữ kết quả cũ |
| File hỏng, rỗng, quá 60 giây | Hiển thị lỗi rõ ràng; API lần lượt 422, 400, 422 |
| Model thiếu / API dừng | Thông báo lỗi; không chuyển sang mock âm thầm |
| Timeout / request khi bận | Theo contract: 504 / 429; ghi rõ test tự động hay thử thủ công |

Thiếu model có thể thử bằng cách dừng app rồi khởi động với `KIDSAFE_MODEL_DIR` trỏ đến một thư mục trống; khôi phục biến trước khi demo. Không xóa checkpoint để thử. Với ca khó tạo thủ công như timeout, ghi kiểm thử tự động và ca chưa làm, không tự điền pass.

Đo riêng lần gọi đầu và những lần sau, ghi ID clip, CPU/OS/Python và số lượt. Mốc tham khảo của Huy: lần đầu 23,34 giây, median 4 lần sau 5,20 giây; **không phải cam kết tốc độ máy bạn**. `scripts.smoke_api` chỉ dùng cho mock. Không chạy `ml.release_smoke` tùy tiện để ghi đè báo cáo công bố của Huy; lưu kết quả máy bạn riêng.

### C. Sửa lỗi app và chuẩn bị demo

Sửa trong `apps/`, test UI/integration và hướng dẫn chạy thuộc app owner. Giữ signature `analyze_video(path, *, backend="model")`, schema v1 và ngưỡng của Huy. App chỉ trình bày tín hiệu đã nhận; nhãn null khác 0, điểm chưa là xác suất an toàn. Báo Huy khi lỗi thuộc model/artifact; không chỉnh nhãn/ngưỡng để làm kết quả đẹp hơn.

Chụp luồng upload, kết quả thật, thiếu audio và lỗi; bỏ thông tin riêng trước khi commit ảnh/log. Quay demo 3–5 phút theo [kịch bản](DEFENSE_NOTES.md), ghi commit và ngày quay. Clip demo ưu tiên train đã gán nhãn xong hoặc fixture; không trình chiếu test giữ riêng. Phần báo cáo/slide ML đã có, chỉ cập nhật bằng chứng app và máy thứ hai.

## 4. Những thứ gửi lại Huy

- Nhánh `feat/app-*`, commit và cách tái hiện các lỗi/sửa đổi. Không force-push nhánh bàn giao.
- Biên bản nghiệm thu từ [mẫu](APP_ACCEPTANCE_TEMPLATE.md), log pytest/pip check/verifier, ảnh UI hoặc video demo. Phân biệt máy thật thứ hai và venv mới trên máy cũ.
- `annotations-app.json` nguyên bản sau gán nhãn độc lập, gửi riêng.
- Thông tin sinh viên của bạn để điền bản nộp; kế hoạch diễn tập chung.

Chỉ đánh dấu xong các mục có bằng chứng. Chưa có nhãn pilot thì chưa có F1/ablation đa phương thức; hai metric benchmark riêng không thay thế kết quả này. Huy còn phụ trách 25 nhãn, đánh giá cuối và cập nhật nội dung ML.

## 5. Prompt cho Codex của thành viên app

Sao chép phần “Prompt cho Codex” trong [tin nhắn bàn giao](APP_HANDOFF_MESSAGE.md). Đọc tài liệu hiện tại này trước các bàn giao lịch sử; `HANDOFF_ML_APP.md` ghi thời điểm app giao cho Huy, không phải trạng thái tích hợp mới nhất.
