# Tiến độ KidSafeReels

## Đang thực hiện cải thiện model — 20/10/2026

- Huy yêu cầu thực hiện kế hoạch và xác nhận chưa lưu các video lỗi; tiếp tục với dữ liệu hiện có.
- Đã thêm công cụ chọn cấu hình trên validation, khóa artifact rồi đánh giá test; xuất CSV/PNG/SVG, Accuracy/Loss và các metric theo lớp. Hướng dẫn: [MODEL_EXPERIMENTS.md](MODEL_EXPERIMENTS.md).
- Toxic: đã train 6 ứng viên. Ứng viên chọn trên validation giảm FP test 74 → 57 nhưng TP cũng giảm 53 → 48; F1 0,4492 → 0,4486. Chưa thay classifier đang dùng vì chưa cải thiện cân bằng precision/recall.
- ASR: đã so 3 cấu hình trên 16 câu dev, chốt small/beam5 rồi đánh giá 16 câu test FLEURS riêng. WER test 0,3836 → 0,1983; CER 0,2240 → 0,1070. Có CSV, hình và audit ảnh hưởng ASR lên đầu ra Toxic; chưa phải accuracy moderation.
- Violence: đang trích feature 24 frame, so mean/max với thêm std/biến thiên liền kề; giữ nguyên nhóm/split RLVS. Chưa có kết quả test ứng viên trong checkpoint tiến độ này.
- Đã kiểm 20 tests metric/pipeline/baseline và 9 tests cấu hình/đồng nhất feature train–inference. Chưa chạy full suite sau cùng hoặc đổi bundle hoạt động. Cache nằm local để tiếp tục nếu gián đoạn.

## Ưu tiên tiếp theo theo phản hồi Huy — 19/10/2026

**Đã ghi nhận, chưa chạy train/đánh giá mới; tiếp tục khi Huy quay lại sau khi quota hồi.** Kế hoạch chi tiết và checklist: [MODEL_IMPROVEMENT_PLAN.md](MODEL_IMPROVEMENT_PLAN.md).

- Toxic đang cảnh báo nhầm video nói chuyện bình thường; Huy thấy transcript sai. Cần tách lỗi ASR, classifier và cách gộp đoạn để cải thiện, chưa chốt nguyên nhân cho từng clip.
- Violence đang bỏ sót video bạo lực rõ theo phản hồi dùng app; cần kiểm tra lấy mẫu frame, recall và khác biệt dữ liệu thực tế so với RLVS.
- Chỉ tập trung Toxic + Violence; Sexual/Dangerous hoãn khỏi đợt này, tiếp tục `null`/chưa đánh giá.
- Yêu cầu đầu ra: kết quả train/validation/test, Accuracy, Loss, Precision/Recall/F1, confusion matrix, đồ thị và CSV; so sánh baseline/bản cải thiện và ghi lỗi điển hình.
- Baseline chưa lưu lịch sử loss theo epoch; phải đo/log thật khi bổ sung, không dựng đường học. Giữ test khỏi vòng chọn model/ngưỡng. Nhãn pilot vẫn cần người thực hiện trước khi báo metric toàn pipeline.
- Điểm bắt đầu lần tới: kiểm tra artifact/cache/split, bổ sung báo cáo metric/CSV/đồ thị, rồi làm các thí nghiệm cải thiện. Chưa có clip lỗi mới của Huy được lưu thành tập dữ liệu.

## Sửa mở lại app khi cổng đã được dùng — 19/10/2026

- Ảnh lỗi của Huy do phiên kiểm thử trước vẫn chạy ở 8000/8501; health xác nhận API backend `model` và web đều hoạt động.
- Launcher nhận diện OpenAPI KidSafeReels, backend yêu cầu và web health; nếu khớp thì dùng phiên hiện có và trả thành công. Không tạo server trùng, không tự dừng tiến trình chiếm cổng.
- `Chay-KidSafeReels.cmd` truyền `-OpenBrowser`: mở trình duyệt khi khởi động xong hoặc mở lại phiên đang chạy. Khi cổng thuộc dịch vụ khác/sai backend/chưa sẵn sàng, in hướng dẫn thay vì thêm stack trace PowerShell cho lỗi cổng.
- **12 tests launcher pass (8,44 giây)**: mở lại, sai backend/dịch vụ, health lỗi, trình duyệt không mở được, khởi động mới chờ readiness và dọn web. Không chạy lại toàn bộ suite 118 tests của lần merge vì phần API/model/UI không thay đổi.
- Chạy thật `scripts/run-model.ps1` cả có/không có `-OpenBrowser`: exit 0, 8 artifact đúng checksum; PID API/web giữ nguyên. Cập nhật README và APP_GUIDE. Các việc nhãn pilot, metric pilot và quay demo vẫn đang chờ.

## Review và merge app — 19/10/2026

- Đã nhận app `47326a6` và merge fast-forward vào `feat/ml-release`; giữ nhánh app remote nguyên trạng.
- Chạy lại trên máy Huy: **118 tests**, **13/13 HTTP/model**, **9/9 UI AppTest** đều qua; pip check và 8 artifact qua.
- Đã kiểm trình duyệt thật: upload, kết quả model, evidence tại giây 4, tải TXT, đổi file xóa kết quả cũ, video không tiếng. Không chạy pilot.
- Sửa launcher để dùng cả `.venv-model`/`.venv`; sửa script nghiệm thu để không ghi đè hoặc gắn nhầm tên máy Tiến.
- Hướng dẫn mở web và giới hạn review: [APP_MERGE_REVIEW.md](APP_MERGE_REVIEW.md). Số đo tóm tắt: `reports/app_merge_review_2026-10-19.json`.
- Còn nhãn của hai người, metric pilot và quay demo; không coi việc thành viên app đã bàn giao code là các mục này đã xong.


## Nghiệm thu máy app - 18/10/2026

- Thành viên app Lê Quyết Tiến nhận ZIP khớp SHA-256, kiểm 17 tệp bàn giao và 8 artifact model.
- Ghép `feat/ml-release` f5d9310 lên `feat/app` theo fast-forward, giữ giao diện d82798d và toàn bộ ML của Huy.
- Cài mới `.venv-model` trên Windows 11 / i5-12450H / Python 3.12.5. Đây là máy khác với Ryzen 5 7535HS của Huy.
- **118 tests pass**, `pip check` và verifier qua; **13/13 ca HTTP model thật**, **9/9 ca UI AppTest nối HTTP/model thật** qua. Xem `APP_ACCEPTANCE_TIEN.md` và ba JSON `reports/app_*` tương ứng.
- Thêm launcher model, biên bản riêng, thông tin sinh viên và sửa ánh xạ lỗi media từ model thành 422. Không sửa ngưỡng hoặc model.
- Chưa chạy classifier trên pilot; 6 nhãn độc lập của Tiến và 25 nhãn Huy vẫn cần người thực hiện. Không có metric pilot mới.
- Công cụ trình duyệt từ chối quyền truy cập nên không xác nhận kiểm thử tay/ảnh UI model hoặc quay video mới. Ghi rõ kiểm thử giao diện ở mức AppTest, không coi là thao tác trình duyệt thủ công.
- Báo cáo/slide bản dành cho Tiến cập nhật bằng chứng máy app và thông tin đã xác nhận. Phần ML gốc vẫn được giữ để đối chiếu.

Nhật ký dưới đây là lịch sử công việc của Huy; số đo của từng máy không trộn lẫn.

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

## Bàn giao cho app ngày 18/10

- Đã cập nhật `docs/HANDOFF_APP.md` thành hướng dẫn tiếp nhận bản tích hợp; thay yêu cầu làm app từ đầu bằng nghiệm thu máy thứ hai, nhãn chéo và demo.
- Có tin nhắn/prompt chuyển cho Codex app tại `docs/APP_HANDOFF_MESSAGE.md` và mẫu trống `docs/APP_ACCEPTANCE_TEMPLATE.md`. Không điền pass thay cho người kiểm tra máy khác.
- Gói gửi riêng nằm tại `data/private/handoff-app/kidsafe-app-handoff-2026-10-18.zip`, kèm SHA-256; chứa model ZIP, annotation ZIP, tài liệu và PDF/PPTX. Không chứa dự đoán/ASR private hoặc raw dataset huấn luyện. Code lấy từ GitHub.
- Gói để Huy gửi cho thành viên app; chưa tự gửi tin nhắn, upload dịch vụ ngoài hoặc xác nhận người nhận đã nhận.

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
