# Kết quả D1–D2 của Huy — 17/10/2026

**Đã hoàn thành chuẩn bị dữ liệu và tiền xử lý; chưa hoàn thành nhãn người thật.** Đây là kết quả chạy thực trên CPU, không phải đánh giá chất lượng moderation.

| Hạng mục | Kết quả thực tế |
|---|---|
| Nguồn | 8 video gốc Wikimedia Commons, CC BY 3.0; có link tác giả/giấy phép/lịch sử review/checksum |
| Pilot | 25 đoạn × 25 giây, 5 nhóm; không phải 25 nguồn độc lập |
| Split trước trích frame | 17 train / 4 validation / 4 test, tương ứng 3 / 1 / 1 nhóm |
| Media | 25 MP4 giải mã được, khoảng 61 MiB; 8 file gốc 313 MiB, hash khớp manifest |
| Frame | 625 JPG, khoảng 1 fps; lưu PTS/time_base và offset về video gốc |
| ASR | 25 clip chạy tiền xử lý; 15 có transcript gồm 58 segment; 10 chưa phát hiện lời nói |
| Nhận diện ngôn ngữ ASR | Trong 15 clip có transcript: 14 `vi`, 1 `en`; chưa được người xác nhận |
| Timestamp ASR | 7 segment vượt biên được giữ mốc thô và giới hạn mốc playback; 0 interval bị loại hoàn toàn |
| Thời gian frame + ASR | Tổng thời gian từng clip 140,935 giây; gồm nạp model lần đầu, không gồm tải/cắt video; không phải benchmark latency app |
| Kiểm tra dữ liệu | 0 lỗi hash/group/split/timestamp; 0 ứng viên gần trùng giao split theo dHash 1fps |
| Cache | Chạy lại preprocessing: 25/25 cache hit |
| Nhãn Huy / app | 0/25 và 0/6 hoàn tất; 6 clip chéo = 24%, phủ 5 nhóm |

Chi tiết máy đọc được: [pilot_data_audit.json](pilot_data_audit.json). Môi trường Python 3.12.10, PyAV 16.1.0, faster-whisper 1.2.1, Whisper base CPU int8 (4 threads), CPU AMD Ryzen 5 7535HS. Cấu hình và hash 4 file model nằm trong cache từng clip; cùng checkpoint baseline đã bàn giao, không fine-tune bằng pilot.

## Nghiệm thu phần kỹ thuật

- `python -m pytest -q`: **44 tests pass** (19,10 giây), bao gồm parser JS cho trang gán nhãn. `git diff --check` sạch.
- Tests mới chặn nguồn/nhóm giao split, nội dung trùng đổi ID, đoạn chồng lấn, đường dẫn/range sai, sửa manifest âm thầm, cache sai phiên bản, nhãn bỏ trống bị tính thành 0, thiếu evidence, và timestamp frame giả định theo FPS.
- Video test tổng hợp có PTS không đều xác minh frame lấy tại 0 / 1,1 / 2,3 giây và offset nguồn 45 / 46,1 / 47,3 giây. Đây chỉ là fixture kiểm thử mã, không nằm trong 25 clip pilot thật.
- Player annotation đã kiểm tra mở trên trình duyệt local, nhập ghi chú và khôi phục sau reload; ghi chú QA được xóa, không tạo nhãn thay người. Gói ZIP có 31 file, gồm đủ 25 video; kiểm tra CRC không lỗi. Chỉ lấy allowlist media, HTML, phiếu trắng, hướng dẫn và ghi công; không trộn nhãn nộp/ASR vào gói. Chưa kiểm tra trọn luồng tải/nạp JSON bằng thao tác người dùng trên máy thứ hai.

## Giới hạn và việc còn lại

Tập chọn thuận tiện, nhiều đoạn cùng nguồn và chủ yếu nội dung thông tin. Ngôn ngữ chưa được người nghe xác nhận; ASR có thể bỏ sót hoặc tạo lời sai. 10 trạng thái không phát hiện lời nói **không đồng nghĩa** video không có tiếng hoặc không độc hại. Không dùng ASR để điền nhãn.

Chưa biết support dương/âm theo Violence/Toxic, nên chưa có precision/recall/F1 trên pilot và chưa có căn cứ chốt ngưỡng ứng dụng. dHash không chứng minh không có mọi dạng gần trùng. Nhóm test chỉ có một nguồn chiến dịch, nên ngay cả sau có nhãn cũng phải báo hạn chế về khái quát.

Huy gán 25 clip và thành viên app gán độc lập 6 clip theo [hướng dẫn](../docs/ANNOTATION_GUIDE.md); giữ các bản gốc, phân xử bất đồng và xác nhận ngôn ngữ. Sau đó kiểm tra support từng split. Nếu không có dương tính trong test, không báo recall lớp dương như thể đã đo được: bổ sung dữ liệu nguy hại có quyền ở phiên bản mới, hoặc công bố pilot v1 chỉ kiểm tra vận hành/cảnh báo nhầm. D3–D7 vẫn cần đánh giá chung, tích hợp app, thử máy thứ hai và báo cáo cuối.
