# Kế hoạch một tuần và phân công

**Toxic hybrid 29/10:** đã triển khai hướng Huy chốt: Whisper tiếng Việt → PhoBERT + luật từ/cụm từ, giải thích nguồn cảnh báo trên app/TXT. `022` được cảnh báo thêm, `019` vẫn review; ViCTSD test F1 giữ 53,39%. Có ablation CSV/đồ thị, không train mới. [Thiết kế và chạy](TOXIC_HYBRID.md), [báo cáo](../reports/TOXIC_HYBRID_2026-10-29.md). Tiếp theo vẫn là nhãn speech200/nhóm nguồn và test đích giữ riêng; không dùng các ca đã xem làm test độc lập.

**Dữ liệu mới 28/10:** đã tiếp nhận/kiểm tra 200 clip lời nói (100 Toxic + 100 bình thường theo mô tả Huy). Huy rà soát 200, app rà soát độc lập 40; có phiếu local/ZIP và audit 7 cặp gần trùng khác ID nguồn. [Hướng dẫn speech200](SPEECH200_ANNOTATION.md). Hoàn tất nhãn/nhóm nguồn trước khóa split, ASR và lượt train tiếp theo; chưa có Accuracy/Loss mới từ bộ này.

**Cập nhật 28/10/2026:** CPU v3 đã thay hai classifier bằng PhoBERT fine-tune và R3D-18 + LR sau kiểm thử. Có kết quả test/đồ thị/CSV, 152 tests, 13/13 HTTP và 8 UI AppTest pass (1 chưa chạy do fixture không có evidence). Nhãn nguồn AIRTLab được xác minh riêng; Accuracy 53% cho thấy khác biệt bối cảnh còn lớn. [Báo cáo mới](../reports/MODEL_IMPROVEMENT_2026-10-28.md), [mở app/bàn giao v3](MODEL_UPDATE_2026-10-28.md). Việc còn lại: nhãn pilot/ca lỗi đích, kiểm máy Tiến, cập nhật PDF/PPTX và demo. Các mốc dưới là lịch sử.

**Cập nhật 20/10/2026:** đã hoàn tất đợt thực nghiệm theo phản hồi Huy: 6 cấu hình Toxic, 9 cấu hình Violence, 3 cấu hình ASR; có train/validation/test, Accuracy/Loss/F1, CSV và đồ thị. CPU v2 nâng Whisper small beam5, giữ hai classifier v1 vì ứng viên không cải thiện đủ trên test. 146 tests, 13/13 HTTP/model, 8 UI AppTest pass và 1 ca chưa chạy vì không có evidence ở fixture. [Báo cáo](../reports/MODEL_IMPROVEMENT_2026-10-20.md), [checklist cải thiện](MODEL_IMPROVEMENT_PLAN.md), [cài/mở v2](MODEL_UPDATE_2026-10-20.md). Còn nhãn pilot, ca lỗi thực tế, kiểm v2 trên máy Tiến và hoàn thiện bài nộp/demo; Sexual/Dangerous tiếp tục hoãn.

**Cập nhật 19/10/2026:** đã nhận code/nghiệm thu tự động máy Tiến, merge vào `feat/ml-release` và kiểm lại trên máy Huy với 118 tests, 13 ca HTTP, 9 ca UI AppTest; có thêm kiểm tra trình duyệt thật. Còn nhãn pilot, metric chung, diễn tập và quay demo. [Biên bản merge và cách mở web](APP_MERGE_REVIEW.md).

**Cập nhật 18/10/2026:** Huy đã hoàn thành baseline D3, tích hợp hai nhánh D4 và công cụ đánh giá/bàn giao D5–D6. Đã có báo cáo PDF 9 trang, slide 11 trang và kịch bản D7. Còn nhãn người thật, đánh giá pilot, máy thứ hai và diễn tập/quay demo. Xem [progress hiện tại](PROGRESS.md); lịch bên dưới là lịch gốc.

**Bổ sung D1–D2 ngày 17/10/2026:** đã chuẩn bị 25 đoạn thật từ 8 nguồn có giấy phép, khóa 5 nhóm và split 17/4/4 trước trích frame; chạy 625 frame + 58 segment ASR. Có gói gán nhãn Huy 25/app 6, nhưng chưa ai hoàn tất nhãn và chưa đo chất lượng moderation trên pilot. Xem [tiến độ lưu để tiếp tục](PROGRESS.md) và [báo cáo D1–D2](../reports/PILOT_D1_D2.md). Đây là phần kỹ thuật D1–D2, không thay thế nghiệm thu chung của cả hai người.

**Tiến độ 16/10/2026:** lịch 08–14/10 bên dưới là kế hoạch ban đầu, chưa hoàn thành theo lịch. Đã có baseline Toxic trên ViCTSD, baseline visual trên RLVS, Whisper base và pipeline hai nhánh qua API; xem `docs/VIDEO_PIPELINE.md` và `reports/`. ViHarm4Kids không còn là dữ liệu bắt buộc. Việc tiếp theo: Huy chuẩn bị/gán nhãn tập video tiếng Việt và đánh giá chung; app owner hoàn thiện playback/evidence, kiểm thử UI và cài trên máy thứ hai. Chưa tự dời deadline 08/11 của giảng viên.

**Ưu tiên ngay sau bàn giao model:** Huy chọn 20–30 clip pilot có quyền sử dụng và lời nói tiếng Việt, ghi nguồn/group_id rồi gán nhãn trước khi xem dự đoán; thành viên app gán nhãn độc lập khoảng 20% để kiểm tra bất đồng. Giữ một phần chưa dùng để điều chỉnh model/ngưỡng cho đánh giá cuối. Đây là pilot nhỏ, báo support và ca lỗi, không công bố độ chính xác tổng quát. Tạm giữ baseline để hoàn thiện sản phẩm; chưa thêm OCR/PhoBERT/hai nhãn P1.

## Lịch và giới hạn

Giả định triển khai **08–14/10/2026** (7 ngày), nằm trong khoảng đề bài **08/10–08/11/2026**. Có thể dời cả lịch nếu nhóm chọn tuần khác. Ngày 05/10 là khởi tạo repo. Mỗi người dự kiến 3–4 giờ tập trung/ngày, tổng 42–56 giờ công; nếu không đủ thời gian hoặc chưa có dữ liệu phù hợp vào ngày 2, giảm phạm vi ngay. Khoảng thời gian còn lại trước hạn dành cho phản hồi giảng viên và nộp bài, không lên kế hoạch thêm tính năng.

Người dùng thực tế: phụ huynh/giáo viên tải video có sẵn lên để sàng lọc trước khi chia sẻ với trẻ. Sản phẩm hỗ trợ quyết định của người lớn; không cam kết an toàn tuyệt đối và chưa chặn nội dung trong app TikTok/Reels/Shorts.

## Phạm vi nghiệm thu

**P0 — cần hoàn thành:** upload video ngắn, ít nhất hai modality thật (hình ảnh và transcript), ít nhất một bộ phân loại được nhóm huấn luyện trên dữ liệu có nhãn, kết quả review/flagged/no_risk_detected, điểm theo nhãn được hỗ trợ, evidence có thời gian, tải JSON, xử lý file lỗi, báo cáo test giữ riêng và demo trên máy khác. Hai lớp rủi ro ưu tiên: violence và toxic_language.

**P1 — chỉ làm khi P0 ổn:** OCR chữ tiếng Việt, thêm sexual/dangerous_acts nếu có dữ liệu phù hợp và kết quả đánh giá đủ rõ, fine-tune PhoBERT/backbone, deploy online.

**Ngoài phạm vi:** extension chặn feed, crawler nền tảng, tài khoản/phân quyền, ứng dụng điện thoại native, dashboard nhiều người, huấn luyện mọi backbone từ đầu, nhận diện tuổi người xuất hiện.

Giữ thiết kế đa nhãn của hai ảnh: một video có thể vừa bạo lực vừa có lời nói độc hại. `Safe` không phải một lớp ngang hàng độc lập; UI dùng “chưa phát hiện rủi ro trong phạm vi đã đánh giá”. Hai lớp chưa hỗ trợ vẫn hiện “chưa đánh giá”. Nếu giảng viên bắt buộc đủ bốn nhóm rủi ro, cần chốt lại phạm vi ngay ngày 1.

## Trách nhiệm

| Hạng mục | Huy — ML lead / leader | Thành viên app | Đầu ra bàn giao |
|---|---|---|---|
| Phạm vi, nhãn, dữ liệu | Quyết định, nguồn dữ liệu, hướng dẫn nhãn | Góp ý tình huống dùng; gán nhãn chéo mẫu | Manifest và phạm vi đóng băng |
| Tiền xử lý / ASR / model | Xây dựng, huấn luyện, cache đặc trưng | Gọi pipeline theo hợp đồng | Hàm suy luận + artifact version |
| API / web | Cung cấp pipeline, review contract | Sở hữu API, upload, hiển thị, thông báo lỗi | Luồng upload → kết quả |
| Fusion và đánh giá | Chọn ngưỡng trên validation, test, ablation | Đo luồng ứng dụng và độ trễ | Bảng metric thật + ví dụ lỗi |
| Tích hợp | Duyệt PR, chốt phiên bản | Ghép app với model, thử trên máy khác | Demo chạy lại được |
| Báo cáo / thuyết trình | Phương pháp, dữ liệu, thực nghiệm | Kiến trúc sản phẩm, hình UI, video demo | Một báo cáo và slide thống nhất |

## Lịch cụ thể

| Ngày | Huy — ML | Thành viên app | Điểm chốt cuối ngày |
|---|---|---|---|
| D1 · 08/10 | Chốt 2 nhãn P0; kiểm tra nguồn/quyền dữ liệu; chuẩn bị 20–30 clip pilot, quy tắc nhãn, thử CPU/Colab | Clone và chạy mock; hoàn thiện luồng upload/kết quả; thống nhất schema v1 | Cả hai chạy được repo; có dữ liệu pilot và API thống nhất |
| D2 · 09/10 | Chốt manifest; chia theo video/nguồn 70/15/15 trước trích frame; lấy frame và transcript có timestamp | Kiểm tra kích thước/định dạng/thời lượng bằng giải mã thực; tiến trình/lỗi; gán nhãn chéo ~20% pilot | Chạy được tiền xử lý 10 clip, có train/val/test không trùng nguồn |
| D3 · 10/10 | Train visual frozen-feature classifier; TF-IDF + logistic regression cho transcript; lưu baseline | Bảng nhãn/điểm, vùng evidence, tải JSON; tiếp tục dùng mock khi cần | Có baseline ML thật và UI chờ pipeline |
| D4 · 11/10 | Ghép hai modality; chọn ngưỡng trên validation; nối analyze_video; thử PhoBERT feature nếu đủ giờ | Tích hợp model thật, xử lý thiếu audio và timeout; playback ở mốc evidence | Luồng đầu-cuối chạy với ít nhất 5 video thật |
| D5 · 12/10 | Đánh giá test một lần sau khi chốt; precision/recall/F1 từng lớp, macro-F1, lỗi điển hình; ablation | Kiểm thử video rỗng/hỏng/dài/không tiếng; đo độ trễ và dọn tệp | Báo cáo thực nghiệm và danh sách lỗi phải sửa |
| D6 · 13/10 | Sửa lỗi tích hợp; model card, mô tả dữ liệu/phương pháp; đóng băng checkpoint | Cài mới trên máy thứ hai, hoàn thiện README, chụp UI, quay demo dự phòng | Bản release candidate; không thêm tính năng |
| D7 · 14/10 | Ghép báo cáo, slide và phần giải thích ML; diễn tập | Ghép demo, hướng dẫn sử dụng và kịch bản lỗi; diễn tập | Bản nộp chạy được, có báo cáo và video demo |

## Điều kiện cắt giảm

- Cuối D2 chưa có dữ liệu/nhãn phù hợp: bỏ OCR, sexual, dangerous_acts; giữ đúng hai nhánh visual + transcript và ghi rõ độ bao phủ. Không tự thu gom nội dung nhạy cảm liên quan trẻ em để bù dữ liệu.
- Colab/GPU không khả dụng: giữ backbone đóng băng, phân loại tuyến tính, ASR cấu hình CPU nhẹ; không fine-tune toàn bộ.
- D4 chưa tích hợp được: dừng tối ưu model, ưu tiên pipeline thật và API trước mọi tính năng phụ.
- Model yếu: công bố kết quả thật, dùng trạng thái review cho bất định; không thay điểm bằng dữ liệu giả để demo.
- Deploy mất quá 2 giờ: demo local trên máy đã kiểm chứng, kèm video quay sẵn có ghi phiên bản.

## Checklist đóng dự án

- [ ] Hai máy cài mới chạy được theo README.
- [x] Có checkpoint/classifier nhóm huấn luyện, cấu hình và quy trình tái tạo.
- [x] Cả visual và transcript thực sự được sử dụng qua HTTP/model thật.
- [ ] So sánh visual-only/text-only/fusion trên pilot có nhãn (đã có công cụ, chưa có kết quả).
- [ ] Chia dữ liệu theo nguồn; không có frame/clip gần trùng giữa train và test.
- [x] Metric benchmark theo lớp, support và thời gian chạy CPU có giới hạn rõ.
- [ ] Metric trên tập video chung có nhãn và đo latency đại diện.
- [x] UI phân biệt mock/model, null/0, thiếu modality và lỗi; evidence dựa trên tín hiệu thực.
- [x] 13 ca HTTP/model thật tự động, gồm video không tiếng và file hỏng.
- [ ] Diễn tập kiểm thử thủ công luồng sử dụng cuối.
- [x] Không đưa video cá nhân, trọng số lớn, token hoặc dữ liệu nhạy cảm lên Git.
- [ ] Chốt báo cáo và slide sau khi thêm metric pilot/thông tin sinh viên.
- [ ] Diễn tập và quay video demo 3–5 phút.
