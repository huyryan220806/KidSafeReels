# Hướng dẫn thành viên app gán nhãn Toxic — 02/11/2026

**Việc cần làm: nghe 30 video, gán nhãn lời nói, ghi mốc bằng chứng và chép lời nghe được; gửi lại file JSON cho Huy.** Không cần chạy model hay web kiểm duyệt để làm phiếu này. Chỉ gán Toxic; giữ Violence v3.

## 1. Nhận đúng hai phần

1. Code và hướng dẫn trên nhánh [`feat/ml-release`](https://github.com/huyryan220806/KidSafeReels/tree/feat/ml-release).
2. Gói **`app-toxic-dev30-2026-11-02.zip`** do Huy gửi riêng. Gói có đủ 30 video và trang gán nhãn, không cần tải model.

**Chỉ clone GitHub sẽ chưa có video.** ZIP và file nhãn giữ riêng, không commit lên Git. Gói 30 clip này là đợt mới; không dùng phiếu `speech200` 40 clip hoặc phiếu pilot cũ để nộp cho đợt này.

Nếu chỉ cần gán nhãn, nhận ZIP là có thể bắt đầu. Nếu muốn lấy code trên máy mới:

```powershell
git clone --branch feat/ml-release https://github.com/huyryan220806/KidSafeReels.git
cd KidSafeReels
```

Nếu repo đã có nhánh này và không có thay đổi chưa lưu:

```powershell
git status
git switch feat/ml-release
git pull --ff-only origin feat/ml-release
```

Nếu đang làm dở trên nhánh app, giữ nguyên công việc; mở ZIP riêng để gán nhãn. Không cần reset, ghi đè file hoặc đổi nhánh để mở phiếu.

## 2. Mở phiếu trong Chrome hoặc Edge

1. Nhấp phải ZIP → **Extract All / Giải nén tất cả**. Không mở HTML trực tiếp bên trong ZIP.
2. Trong thư mục đã giải nén, mở **`Mo-Phieu.cmd`**, hoặc nhấp phải **`review-app.html` → Open with → Google Chrome / Microsoft Edge**.
3. Trang phải hiện **Thành viên app**, ban đầu **0 / 30 clip đã điền nhãn**. Nếu từng làm trên cùng trình duyệt, trang có thể khôi phục bản nháp của đợt này.
4. Bấm Play, bật tiếng và kiểm tra video phát được. Giữ thư mục `media/` nằm cạnh `review-app.html`.

**Dùng Chrome/Edge bên ngoài cửa sổ chat Codex.** Lần trước tải JSON trong trình duyệt tích hợp của chat không thành công. Địa chỉ `127.0.0.1` của Huy không dùng được trên máy bạn; phiếu trong ZIP chạy trên máy của chính bạn.

Cấu trúc sau giải nén:

```text
app30/
  Mo-Phieu.cmd
  review-app.html
  review-manifest.json
  annotations-app.blank.json
  CHECKSUMS.json
  HUONG_DAN.md
  media/
    review-....mp4
```

Nếu trình duyệt không phát video qua file trực tiếp, có thể mở server local. Trong Terminal tại thư mục chứa `review-app.html`, nếu máy đã có Python:

```powershell
py -3 -m http.server 8768 --bind 127.0.0.1
```

Mở **http://127.0.0.1:8768/review-app.html** bằng Chrome/Edge, giữ terminal mở trong lúc làm, Ctrl+C để dừng. Nếu cổng bận, dùng `8769` ở cả lệnh và URL. Khi đổi từ mở file sang server, nạp JSON đã tải để tiếp tục vì bản nháp trình duyệt có thể khác.

## 3. Gán nhãn từng video

1. Nghe hết clip, có thể nghe lại; tích **Tôi đã xem/nghe toàn bộ clip** và chọn ngôn ngữ.
2. Chọn nhãn theo lời thật trong audio, không dựa vào hình ảnh, tên file hoặc đoán kết quả model.
3. Với nhãn **1**, bấm **+ Thêm khoảng bằng chứng**, điền giây bắt đầu/kết thúc và câu thực sự nghe được. Có nút **Lấy thời điểm đang xem**. Mốc phải thỏa `0 ≤ bắt đầu < kết thúc ≤ độ dài clip`; có thể thêm nhiều khoảng.
4. Điền lời nghe được vào ô transcript. Ưu tiên chép đầy đủ cả 30 clip, gồm cả clip bình thường, để Huy phân biệt lỗi nhận dạng giọng nói với lỗi classifier. Có thể thêm mốc `[00:03]` để chỉ vị trí.
5. Chỉ tích **Tôi đã đối chiếu toàn bộ transcript với audio** khi đã chép và kiểm đủ. Nếu chỉ chép một phần hoặc còn từ không rõ, ghi `[không rõ]` và mô tả trong ghi chú; để ô xác nhận transcript chưa tích. Không đoán hoặc tự điền lời bằng ASR/AI.
6. Bấm **Sau →** để chuyển clip. Sau mỗi 5 clip, tải JSON dự phòng.

| Nhãn | Chọn khi |
|---|---|
| **1 — Có lời nói độc hại** | Nghe rõ chửi tục, lăng mạ, đe dọa gây hại hoặc miệt thị. Chỉ cần một đoạn có bằng chứng; các câu còn lại không tự thành Toxic. |
| **0 — Không có lời nói độc hại** | Đã nghe đủ và không có nội dung trên. Nói lớn, bất đồng, nhắc đến bạo lực hoặc từ chỉ quan hệ gia đình không tự là Toxic. |
| **Chưa rõ / cần xem lại** | Âm thanh bị che, khó nghe hoặc thiếu ngữ cảnh để quyết định. Ghi rõ lý do ở ô ghi chú. |
| **Chưa gán** | Chưa làm clip đó; không tương đương nhãn 0. |

Phân biệt nói **về** việc chửi bới với **nói ra** lời chửi. Trích nguyên lời chửi vẫn nghe rõ thì gán 1 và ghi bối cảnh `trích dẫn/giáo dục`. Câu chỉ dùng để nhấn mạnh nhưng không có nội dung chửi tục, lăng mạ, đe dọa hoặc miệt thị có thể là 0; trường hợp ranh giới chưa thống nhất thì chọn chưa rõ và ghi nguyên câu để phân xử. Không tự gán 1 chỉ vì có riêng một từ như “mẹ”.

Nhãn 0 không được kèm khoảng **bằng chứng Toxic**. Nếu muốn giải thích vì sao câu đó bình thường, ghi vào **Ghi chú**. Chữ trên màn hình không được đọc thành tiếng chỉ ghi chú, không dùng làm nhãn Toxic audio.

Trong khi gán, không mở dự đoán app, ASR hoặc nhãn Huy để tham khảo. Nếu đã biết kết quả một clip từ trước, ghi việc đó trong ghi chú; vẫn đánh giá theo lời mình nghe được. Không cố điền cho cân bằng số nhãn 0 và 1.

## 4. Lưu và gửi kết quả

1. Bấm **Tải JSON nhãn hiện tại**. Có thể tải bản đang làm dở; các mục đã điền phải hợp lệ.
2. Mở danh sách tải xuống của Chrome/Edge bằng **Ctrl+J**, bấm **Show in folder / Hiển thị trong thư mục**.
3. Tên file dự kiến: **`annotations-app-app-toxic-dev30-2026-11-02.json`**. Trình duyệt có thể thêm `(1)`, `(2)` khi tải nhiều lần; gửi bản mới nhất.
4. Khi làm tiếp hoặc chuyển máy, mở đúng phiếu và dùng **Nạp lại JSON**. Kiểm tra số nhãn và vài câu đã ghi để chắc chắn khôi phục đúng.
5. Khi kết thúc, kiểm tra **30 / 30**, tải bản cuối, gửi riêng file JSON cho Huy. Nếu có clip chưa rõ thì giữ nguyên và báo số lượng; không ép thành 0/1. Không cần gửi lại video.

**Không gửi file `annotations-app.blank.json`: đây là phiếu trống.** “30/30 đã điền” có thể bao gồm nhãn chưa rõ và không tự chứng minh tất cả transcript đã đầy đủ. Giữ JSON gốc sau khi gửi, không chỉ dựa vào lưu nháp trình duyệt.

Nếu bấm tải mà không thấy file: đọc lỗi ngay dưới nút tải, sửa clip được báo lỗi, rồi Ctrl+J. Các lỗi thường gặp là thiếu xác nhận đã nghe, thiếu ngôn ngữ, nhãn 1 thiếu mốc/câu bằng chứng, mốc vượt clip hoặc nhãn 0 còn bằng chứng Toxic. Không xóa dữ liệu trình duyệt khi chưa có JSON dự phòng. Nếu vẫn không xuất được, giữ nguyên tab và báo Huy.

## 5. Kiểm tra bằng code nếu đã có repo — tùy chọn

Không cần cài thư viện ML cho bước này; Python 3.11+ là đủ. Nếu giải nén vào `data/private/app-toxic-dev30-2026-11-02/app30/`, từ gốc repo chạy (thay đường dẫn JSON bằng file thật):

```powershell
py -3 -m ml.toxic_app_review check --pack "data/private/app-toxic-dev30-2026-11-02/app30" --input "C:\Users\TEN_BAN\Downloads\annotations-app-app-toxic-dev30-2026-11-02.json" --require-complete
```

Có thể dùng đường dẫn tuyệt đối khác cho `--pack`, trỏ vào thư mục chứa `review-manifest.json`. Nếu dùng môi trường repo, thay `py -3` bằng `.\.venv\Scripts\python.exe`. Bỏ `--require-complete` để kiểm phiếu dở. Lệnh báo số nhãn còn thiếu, nhãn chưa rõ và transcript đã xác nhận; không xác minh người gán nghe đúng.

## 6. Huy xử lý sau khi nhận

- Lưu nguyên JSON tại `data/private/app-toxic-dev30-2026-11-02/submissions/`, chạy lệnh kiểm tra trên; không sửa đè bản gửi.
- Gói có 10 clip đối chiếu với Huy và 20 clip phát triển bổ sung. Chỉ xem nhãn của nhau sau khi nộp độc lập; lưu hai nhãn gốc, quyết định phân xử và lý do nếu có bất đồng.
- Giữ split đã khóa: **24 train / 6 validation, không có test**. Chỉ nhãn train đủ rõ mới dùng fit; validation dùng đánh giá. Đây là mẫu ưu tiên tìm lỗi, không dùng để công bố Accuracy đại diện.
- Chép lời đầy đủ giúp so transcript người nghe với ASR. Không coi câu bằng chứng ngắn là transcript toàn clip và không tính WER/CER trên bản chép còn thiếu.
- Giữ Violence v3. Đợt bàn giao chỉ chuẩn bị gán nhãn, chưa tạo nhãn người thật hoặc train model mới.

Gói ZIP trên máy Huy: `data/private/app-toxic-dev30-2026-11-02/app-toxic-dev30-2026-11-02.zip`. Checksum và thống kê gói nằm trong [`reports/APP_TOXIC_ANNOTATION_2026-11-02.json`](../reports/APP_TOXIC_ANNOTATION_2026-11-02.json) của repo. `selection-private.json` lưu ánh xạ nguồn và cách chọn mẫu, không nằm trong ZIP gửi người gán nhãn.
