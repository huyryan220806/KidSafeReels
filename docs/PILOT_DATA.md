# Dữ liệu pilot D1–D2

Phiên bản `vi-public-pilot-v1`, ngày 17/10/2026. Xem [tiến độ](PROGRESS.md), [manifest cố định](../data/pilot/manifest.json), [ghi công từng nguồn](../data/pilot/ATTRIBUTION.md) và [audit thực chạy](../reports/pilot_data_audit.json).

## Đã chốt dữ liệu nào?

**25 đoạn × 25 giây từ 8 video gốc, 5 nhóm nguồn**, đều có audio trong nguồn. Nguồn chủ yếu tin dịch vụ bưu điện, sinh hoạt và thông tin sức khỏe tiếng Việt, công khai trên Wikimedia Commons với CC BY 3.0 và bằng chứng YouTubeReviewBot đã kiểm tra tag giấy phép. Manifest lưu tác giả, nguồn gốc, liên kết giấy phép, bản sửa đổi trang quyền, bản lưu lịch sử, SHA-1 và kích thước file gốc. Ghi công phải đi cùng bản sao/clip.

Chọn nguồn/đoạn theo quyền sử dụng và độ đa dạng, không dựa vào điểm model hoặc nhãn. Đây là **tập pilot vận hành**, chưa phải benchmark moderation cân bằng. 25 đoạn không phải 25 nguồn độc lập, không đại diện video dọc hoặc nội dung mạng xã hội; chưa biết số mẫu dương của Violence/Toxic cho tới khi người thật gán. Không huấn luyện lại hay công bố F1 hệ thống trên dữ liệu chưa có gold.

| Split | Clip | Tỷ lệ | Nhóm |
|---|---:|---:|---|
| train | 17 | 68% | Thanh Hóa/bưu điện; Nghệ An/bưu điện; VNC TV/mua sắm |
| validation | 4 | 16% | Bưu điện trung tâm Sài Gòn — hai tác giả cùng địa điểm |
| test | 4 | 16% | SFGovTV — hai video cùng chiến dịch sức khỏe |

Tỷ lệ xấp xỉ 70/15/15; **ưu tiên không tách nhóm**, không stratify bằng nhãn chưa có. Mọi đoạn của một video, các tin bưu điện cùng nhà xuất bản, hai video cùng địa điểm và hai clip cùng chiến dịch nằm chung split. File `manifest.sha256` khóa cấu hình trước khi tải/cắt/trích frame. Metadata nguồn không phải dữ liệu huấn luyện. Nếu phải sửa nhóm/nguồn sau audit, tạo phiên bản mới và tiền xử lý lại; không âm thầm đổi tập test.

`.gitattributes` giữ nguyên byte của manifest này để Git không đổi newline và làm sai checksum khi clone giữa Windows/Linux. Không format lại manifest đã khóa. Công cụ tạo phiên bản mới ghi LF tường minh.

## Nguồn không chọn

- Trailer Galaxy Studio có tag CC trên Commons nhưng các trang được kiểm tra còn chờ license review. Chưa đưa vào pilot, không kết luận rằng tác giả cấm sử dụng.
- Terra X redub tiếng Việt có giấy phép nhưng audio là giọng AI; không trộn vào pilot lời nói tự nhiên này.
- Các file VOA có tag public domain nhưng cần phân biệt tư liệu hãng tin bên thứ ba; không cần đưa vào bản v1.
- ViHarm4Kids chưa có bản tải đã xác minh trong dự án, không dùng làm điều kiện chặn D1–D2.

Thông tin giấy phép dựa trên trang nguồn có lưu phiên bản và lịch sử kiểm tra, không phải một bảo đảm độc lập về toàn bộ chuỗi sở hữu. Khi phát hiện tranh chấp hoặc quyền không phù hợp, loại nguồn khỏi phiên bản tiếp theo và giữ bản ghi lý do.

## Chạy tiếp / tạo lại trên máy khác

Checkout `feat/ml-pilot-data`; chuẩn bị môi trường/model theo [VIDEO_PIPELINE.md](VIDEO_PIPELINE.md). Không cần GPU.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-pilot.txt
.\.venv\Scripts\python.exe -m ml.pilot_data prepare
.\.venv\Scripts\python.exe -m ml.pilot_data preprocess
.\.venv\Scripts\python.exe -m ml.pilot_data audit
.\.venv\Scripts\python.exe -m ml.pilot_annotation pack
```

Lệnh `prepare` dùng manifest đã commit, không cần chạy discovery/freeze. Tải 8 file gốc (~313 MiB), đối chiếu kích thước và SHA-1 trước khi dùng; sai phiên bản thì dừng. FFmpeg từ `imageio-ffmpeg==0.6.0` cắt đoạn, giữ tỷ lệ ảnh trong 640×480, xuất H264/AAC 16 kHz mono, không thay lời nói. FFmpeg seek khi re-encode có sai số cấp frame; mốc nguồn lưu là offset đoạn + timestamp clip, cần nghe/xem lại khi trích dẫn chính xác.

`preprocess` giải mã toàn video, lấy khoảng 1 frame/giây và lưu PTS thật + time_base + thời gian clip/nguồn; không đoán timestamp bằng số frame/FPS. Whisper base local chạy CPU int8, beam 1, VAD; lưu segments, ngôn ngữ và cờ độ tin cậy. ASR là đầu ra tự động, không phải nhãn hoặc transcript chuẩn. Chỉ chuẩn bị dữ liệu, **không gọi hai classifier và không chọn ngưỡng**. Test được xử lý bằng cùng cấu hình cố định, không đọc nội dung transcript/frame test để điều chỉnh.

Mỗi clip lưu tiến độ và cache riêng. Chạy lại sẽ dùng file đã qua checksum; cache preprocessing khóa theo manifest, clip, model và cấu hình. Nếu phiên bản mã tiền xử lý thay đổi phải tăng `CONFIG.version`. Lệnh audit kiểm tra hash frame/media, nguồn/nhóm/split, timestamp; tìm ứng viên gần trùng bằng dHash 1fps. dHash là heuristic, không chứng minh hoàn toàn không có cảnh trùng. Audit có ứng viên giao split sẽ thoát lỗi để xem xét trước huấn luyện/đánh giá.

Whisper có thể trả mốc vượt cuối clip: lưu `raw_start_seconds/raw_end_seconds`, giới hạn mốc playback trong thời lượng media và đánh dấu `timing_clipped`. Đoạn nằm hoàn toàn ngoài media được ghi riêng trong `rejected_intervals`, không dùng như evidence. Không đổi mốc gốc âm thầm. Cần người nghe đối chiếu khi đánh giá vị trí evidence.

## Đầu ra và gán nhãn

| Đường dẫn | Nội dung | Git |
|---|---|---|
| `data/pilot/` | Manifest, checksum, ghi công, phân công nhãn | Có |
| `data/raw/pilot/` | Video gốc và metadata discovery | Không |
| `data/processed/pilot-v1/media/` | 25 clip MP4 | Không |
| `data/processed/pilot-v1/features/<id>/` | Frame JPG và preprocess.json chứa ASR | Không |
| `data/processed/pilot-v1/media-state.json` | Checksum, duration, phiên bản FFmpeg, mốc hoàn tất từng clip | Không |
| `data/private/pilot-v1/` | Player/phiếu gán nhãn local; các bản nộp đặt ở submissions/ | Không |
| `data/private/kidsafe-pilot-v1-annotation.zip` | Gói đưa cho hai người gán; clip + nguồn + hướng dẫn, không có ASR/nhãn đã nộp | Không |
| `reports/pilot_data_audit.json` | Kết quả kiểm tra số lượng/checksum/thời gian, không transcript/ảnh | Có |

Đọc [quy tắc gán nhãn](ANNOTATION_GUIDE.md). Huy gán 25, app owner gán độc lập 6; chưa có nhãn người thật thì không đánh dấu công việc này hoàn tất. Sau khi gán: phân xử bất đồng, báo support từng lớp/split, chỉ dùng phần train/validation cho cải tiến, test giữ lại để đánh giá cuối. Nếu thiếu ca nguy hại có quyền sử dụng, bổ sung tập mới và giữ báo cáo v1 như audit vận hành; không tự biến tin giáo dục thành nhãn nguy hại cho đủ lớp.
