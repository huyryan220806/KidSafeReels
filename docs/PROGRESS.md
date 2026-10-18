# Tiến độ KidSafeReels

Cập nhật **18/10/2026**, nhánh tích hợp **`feat/ml-release`**. Người dùng xác nhận chưa gán nhãn và yêu cầu hoàn thiện các phần khác. Không tự tạo nhãn hoặc metric pilot.

## Trạng thái phần việc của Huy

| Mốc | Đã thực hiện | Còn cần hoàn tất |
|---|---|---|
| D1–D2 | Phạm vi, 25 clip có giấy phép, manifest nhóm/split khóa, frame + ASR, player gán nhãn | Huy gán 25, app gán 6, phân xử |
| D3 | Huấn luyện và đánh giá hai baseline, giữ checkpoint và checksum | Không có việc kỹ thuật còn thiếu trong baseline đã chọn |
| D4 | Hai modality thật, ghép app d82798d, 5 clip train qua HTTP/model thật | Không có lỗi tích hợp đã biết trong 13 ca smoke |
| D5 | Metric benchmark, ca lỗi ASR, công cụ agreement/gold/evaluation/ablation, 117 tests | Metric và ablation trên pilot có nhãn |
| D6 | Model card, cài bundle có kiểm hash, phụ thuộc PyAV 16.1.0, lock môi trường, handoff | Nghiệm thu cài/chạy trên máy thứ hai |
| D7 | Báo cáo kỹ thuật, PDF, slide và speaker notes, kịch bản demo + Q&A đã kiểm tra | Bổ sung metric pilot/thông tin sinh viên, diễn tập và quay demo |

Không quy đổi bảng này thành 100% dự án: nhãn, đánh giá đích và nghiệm thu của người dùng vẫn còn.

## Kiểm chứng bản tích hợp

- Full suite: **117 passed**, 71,50 giây. `pip check` pass. 8 artifact khớp manifest.
- **13/13 ca HTTP thật pass**: 5 clip train có cả hai score, 2 clip tổng hợp không tiếng/có âm thử, 6 ca đầu vào lỗi. Đây là kiểm tra kỹ thuật, không phải số dự đoán nội dung đúng.
- Lần gọi đầu 23,34 giây; 4 lần sau median 5,20, p95 mô tả 11,95 giây. Cùng nhóm nguồn, mỗi clip chạy một lần, chưa là benchmark hiệu năng đại diện.
- Nhánh app giữ nguyên trên remote. Nhánh tích hợp ghép từ ML `32d1ad4` và app `d82798d`; schema v1 không đổi, không fallback mock.
- Giữ kết quả/ASR/nhãn/video/checkpoint ở thư mục private hoặc ignored. Chỉ 005–009 train đã chạy classifier cho smoke; test pilot chưa chạy classifier.
- Audit D1–D2 bên dưới là ảnh chụp trạng thái 17/10, giữ nguyên. Dòng `predictions_run=false` trong audit áp dụng lúc audit, trước smoke train ngày 18/10.

## Đầu ra để tiếp tục khi hết quota

- [Chạy bản tích hợp và bàn giao máy khác](RELEASE_HANDOFF.md).
- [Quy trình so phiếu, tạo gold và đánh giá](EVALUATION_GUIDE.md).
- [Báo cáo kỹ thuật](../reports/TECHNICAL_REPORT.md), [PDF](../output/pdf/KidSafeReels_BaoCaoKyThuat.pdf), [slide](../output/presentation/KidSafeReels_ThuyetTrinh.pptx).
- [Kịch bản demo và câu hỏi bảo vệ](DEFENSE_NOTES.md).
- Model ZIP local: `models/checkpoints/bundle/kidsafe-cpu-v1.zip`; không nằm trên Git.
- Phiếu/video local: `data/private/kidsafe-pilot-v1-annotation.zip`; không chứa dự đoán.
- Chưa có phiếu người thật. Khi nhận hai JSON, chạy hướng dẫn evaluation; không tải/train lại baseline, không thay split hoặc chỉnh ngưỡng bằng test.
- Đã commit và push bản tích hợp `fdec72b` lên `origin/feat/ml-release`. PDF 9 trang và slide 11 trang đã render/kiểm tra từng trang. Nhật ký này được lưu bằng commit tiếp theo; checkpoint/video vẫn chỉ ở local.

---

## Lịch sử đã lưu ngày 17/10


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
