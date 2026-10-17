# Tiến độ KidSafeReels

Cập nhật 17/10/2026. Nhánh đang làm: `feat/ml-pilot-data`, nối tiếp `cb64e05` (pipeline CPU). Đây là nhật ký trạng thái thực; chỉ đánh dấu xong khi có đầu ra và kiểm tra tương ứng.

## Công việc hiện tại: bù phần D1–D2

- [x] Chốt P0: Violence + Toxic tiếng Việt; API/schema v1 và baseline CPU đã có.
- [x] Người dùng xác nhận chưa có clip riêng; dùng nguồn công khai có quyền sử dụng.
- [x] Kiểm tra giấy phép: 8 nguồn CC BY 3.0 có lịch sử review; tạo 25 đoạn video thật, mỗi đoạn 25 giây.
- [x] Khóa manifest trước trích frame; 5 nhóm nguồn, train/validation/test = 17/4/4 (68/16/16).
- [x] Có quy tắc nhãn, player/phiếu cho Huy 25 clip và app 6 clip độc lập (24%).
- [x] Chạy thực 25/25 clip: 625 frame theo PTS, 58 đoạn ASR, cache theo checksum và cấu hình.
- [x] Audit hash/nguồn/nhóm/timestamp: 0 lỗi, 0 ứng viên gần trùng giao split theo dHash; **44 tests pass** (19,10 giây), gồm regression kiểm tra cú pháp JS.
- [ ] Hai người xem/nghe clip, gán nhãn độc lập và giải quyết bất đồng (chưa thực hiện).
- [x] Đã commit/push lên `origin/feat/ml-pilot-data`: `ddd0256` (D1–D2), `d380b64` (giữ checksum manifest khi clone Windows/Linux). Raw media, frame, ASR, ZIP và checkpoint không lên Git.

**Phần kỹ thuật của Huy trong D1–D2 đã có đầu ra; phần nhãn người thật chưa hoàn tất.** Không gộp hai trạng thái thành 100% D1–D2 của cả nhóm. 25 đoạn chỉ đến từ 8 video gốc; tập chủ yếu thông tin đời thường/giáo dục, chưa chứng minh có đủ lớp dương. Không dùng số lượng clip để suy ra độ chính xác.

Whisper trả transcript cho 15 clip (14 nhận diện `vi`, 1 `en`); 10 clip chưa phát hiện lời nói. Đây là kết quả ASR, chưa xác nhận ngôn ngữ/nội dung bằng người thật. Có 7 segment bị giới hạn mốc ở cuối clip, giữ cả mốc thô để đối chiếu. Không chạy classifier trên pilot và chưa dùng test chọn model/ngưỡng.

## Đầu ra và cách tiếp tục khi hết quota

- [Hướng dẫn chạy lại](PILOT_DATA.md), [quy tắc gán nhãn](ANNOTATION_GUIDE.md), [báo cáo D1–D2](../reports/PILOT_D1_D2.md).
- Manifest khóa: `data/pilot/manifest.json`; SHA-256 `6ef3f86b0ac931dd720a866231719d8ba2d01c72b102a55c911d08746c911e20`.
- Audit: `reports/pilot_data_audit.json`. Media/frame/ASR trong `data/processed/pilot-v1/`, không lên Git.
- Gói giao hai người: `data/private/kidsafe-pilot-v1-annotation.zip` (~61 MiB); giải nén, Huy mở `review-huy.html`, app mở `review-app.html`. Có ghi công nguồn; không chứa dự đoán/ASR/nhãn đã nộp.
- Nếu xử lý bị gián đoạn: chạy lại `python -m ml.pilot_data prepare`, `preprocess`, `audit`. Các clip hoàn thành có cache riêng. Không cần tải/huấn luyện model lại.
- Kiểm tra Git: byte manifest trong commit khớp checksum local; `git diff HEAD~2 --check` sạch sau cấu hình `cr-at-eol` riêng cho manifest khóa. Gói ZIP được kiểm tra CRC và đủ 25 media. Nhánh app không bị sửa/merge trong lượt này.
- Việc kế tiếp: hai người nộp JSON riêng → kiểm tra bằng `ml.pilot_annotation check` → phân xử bất đồng → báo support → bổ sung ca nguy hại có quyền nếu thiếu lớp dương → đánh giá theo split đã chốt. Giữ nhãn test khỏi vòng điều chỉnh.

## Mốc đã hoàn thành trước đó

- Toxic ViCTSD: train/evaluate thật, F1 Toxic 0,4492; `3bba5b5`.
- Visual RLVS + ASR + API: F1 Violence 0,9598 trên RLVS, 24 test và 9 ca API; `cb64e05`.
- Model artifacts và gói ZIP nằm local; tài liệu tại `docs/VIDEO_PIPELINE.md`.
- Có ca ASR gây cảnh báo Toxic nhầm; chưa có đánh giá moderation trên tập video tiếng Việt chung.

## Phân công đang chờ

Huy: dữ liệu/nhãn, đánh giá chung, báo cáo ML. Thành viên app: tích hợp model, UI/playback, máy thứ hai; có nhánh remote `feat/app` riêng, không bị sửa trong công việc dữ liệu này. D3–D7 chưa được coi là hoàn tất chỉ vì đã có baseline.
