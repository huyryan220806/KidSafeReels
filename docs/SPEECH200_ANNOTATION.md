# Rà soát 200 clip lời nói — 28/10/2026

**Cập nhật29/10:** Huy yêu cầu train ngay với dữ liệu hiện có. Đã thực hiện một lượt **học từ nhãn yếu cấp clip**, giữ phiếu nghe duyệt và `human_label` nguyên trạng. Có split theo nhóm142/28/30, MILmaxlogit,5epoch và [báo cáo kết quả](../reports/LOCAL_ADAPTATION_2026-10-29.md). Không coi nhãn yếu/ASR là gold hoặc gán nhãn clip cho mọi câu; hướng dẫn nghe duyệt dưới đây vẫn cần cho bước cải thiện tiếp. Holdout của lượt này đã được đánh giá, không còn là test chưa xem cho các lần chỉnh model sau.

Huy đã bổ sung 100 clip nói chuyện bình thường trong `data/Video_Train/Toxic/KidSafeReels-100-Clip-LoiNoi/Non_Toxic`, bên cạnh 100 clip được mô tả là Toxic. Có 74 video nguồn ở nhóm đầu và 58 ở nhóm mới, tổng 132 ID YouTube khác nhau. Lời mô tả của Huy được lưu ở `user_reported_clip_label` (100 nhãn 1, 100 nhãn 0). Metadata cả hai bộ ghi nhãn gợi ý bằng ASR/từ khóa, `labels.toxic_language=null`; chưa có phiếu nghe duyệt từng clip. Không biến nhãn gợi ý thành nhãn chuẩn đã được người kiểm tra.

Đây là bộ mới `speech200-intake-v2`. Giữ nguyên manifest speech100 cũ; không ghi đè phiếu đã nộp. Kết quả kiểm tra kỹ thuật thực tế nằm ở [báo cáo intake](../reports/SPEECH200_INTAKE_2026-10-28.json).

## Mở và sử dụng phiếu

- Huy: mở `data/private/speech200-2026-10-28/review-huy.html` bằng Chrome/Edge, 200 clip.
- Thành viên app: giải nén `speech200-app-review.zip` trong cùng thư mục, mở `review-app.html`, 40 clip. Phải giải nén cả thư mục `media/` trước khi mở HTML.
- Các clip được trộn và đổi sang mã `review-001`…; tên file, đường dẫn phát video và thứ tự không tiết lộ nhóm Toxic/Non_Toxic. Phiếu không chứa ASR, dự đoán, tiêu đề nguồn hoặc nhãn gợi ý. Bảng ánh xạ riêng chỉ nằm ở manifest local.
- App nhận 20 clip từ mỗi nhóm mô tả ban đầu, mỗi clip từ một video nguồn khác nhau (40 nguồn), seed42. Đây là mẫu đối chiếu nhãn, chưa phải tập test. Hai người gán độc lập; chưa tự gửi file/tin nhắn cho thành viên app.

1. Nghe cả clip, xác nhận **Tôi đã xem/nghe toàn bộ clip** và ngôn ngữ.
2. Chọn nhãn 1, 0 hoặc chưa rõ. Chưa nghe thì để trống; trống không phải 0.
3. Nhãn 1: thêm khoảng bằng chứng, ghi câu/lý do thực sự nghe được. Mốc tính từ đầu clip; có nút lấy thời điểm đang phát.
4. Transcript là tùy chọn; chỉ tích đã đối chiếu toàn bộ khi đã nghe và sửa đầy đủ. Không đoán từ bị che hoặc không rõ.
5. Tải JSON sau vài clip và trước khi đóng. Nạp lại JSON để đổi máy/tiếp tục. Local storage chỉ là bản nháp; không thay bản sao JSON. Được tải phiếu dở nếu những nhãn đã điền hợp lệ.

Theo [quy tắc nhãn](ANNOTATION_GUIDE.md): chửi tục, lăng mạ, đe dọa gây hại, miệt thị nghe được → 1. Giọng lớn, phương ngữ, bất đồng hoặc chỉ nhắc đến “chửi bới” không tự là Toxic. Trích lời chửi vẫn nghe rõ: ghi bối cảnh. Không nghe rõ → `uncertain`. Clip có nhãn 0 vẫn có thể chứa rủi ro hình ảnh; đây chỉ là nhãn lời nói.

## Trước huấn luyện và đánh giá

Một clip Toxic có thể chứa nhiều câu bình thường: không gán 1 cho toàn bộ đoạn ASR. Dùng bằng chứng có mốc và transcript đã rà soát để tạo nhãn đoạn; đoạn không được đánh dấu cũng không tự thành âm tính nếu chưa rà hết.

132 ID YouTube chỉ là nhóm nguồn tạm. Audit SHA phát hiện tệp trùng chính xác, dHash ở 1 fps gợi ý hình gần trùng; cả hai không chứng minh video khác ID là khác buổi nói/stream. Danh sách cần rà soát nằm ở `duplicate_review.json` local. Kiểm reupload/nguồn gốc trước khi khóa split và chạy ASR/trích đặc trưng model; không chia ngẫu nhiên từng clip. Cùng video, buổi nói hoặc bản đăng lại đã xác định phải chung split.

**Ngoại lệ chẩn đoán đã ghi nhận 28/10:** Huy báo ba ca lỗi `vi-speech-048/019/022`; đã dùng riêng chúng để chẩn đoán/sửa code. `development-exposure-user-failures-2026-10-28.json` local ghi 3 clip đã chạy và 11 clip liên quan bị loại khỏi test độc lập tương lai. Khi chia nguồn phải đọc record này; không đổi manifest/phiếu trống thành nhãn thật. Huy đã xem đầu ra các ca này nên không gọi việc gán lại của Huy trên chúng là gán nhãn mù. ASR/bộ lọc và kết quả xem [báo cáo](../reports/ASR_FILTER_FIX_2026-10-28.md).

Lộ trình tiếp theo:

1. Huy nghe 200, app nghe độc lập 40; đối chiếu/phân xử bất đồng, loại clip không rõ ngôn ngữ hoặc nhãn. Giữ cả phiếu gốc.
2. Hoàn tất nhóm nguồn, khóa manifest/split theo nhóm (mục tiêu 70/15/15, số lượng thực tế tùy nhóm/lớp); test chỉ dùng sau khi đã chọn model/ngưỡng.
3. Chạy Whisper small beam5; sửa transcript/nhãn đoạn ở train và validation. Giữ ASR thô để phân tích lỗi, không coi ASR là transcript chuẩn.
4. So baseline v3 và ứng viên bằng cùng quy trình. Dùng validation để chọn; xuất Accuracy, Loss, Precision/Recall/F1, FP/FN và đồ thị/CSV test sau khi khóa model. So lỗi do ASR với lỗi classifier khi có transcript người sửa.

Tập cân bằng 100/100 do tuyển chọn không đại diện tỷ lệ Toxic thực tế. Metadata cả hai lớp đã lọc bằng từ khóa nên có thiên lệch; báo support, nhóm nguồn và giới hạn, không suy ra accuracy toàn TikTok. 50 clip Violence đã trùng SHA hoàn toàn với AIRTLab dùng đánh giá trước đó; không tính là dữ liệu mới hoặc đưa vào train rồi giữ tên test ngoài cũ.

## Lệnh tiếp tục

```powershell
.\.venv\Scripts\python.exe -m ml.speech_intake prepare
# Chỉ tạo lại HTML/ZIP từ manifest đã khóa, không giải mã lại:
.\.venv\Scripts\python.exe -m ml.speech_intake pack
# Sau khi tải phiếu (tạo submissions và chép file vào):
.\.venv\Scripts\python.exe -m ml.speech_intake check --annotator huy --input data/private/speech200-2026-10-28/submissions/annotations-huy-speech200.json
.\.venv\Scripts\python.exe -m ml.speech_intake check --annotator app --input data/private/speech200-2026-10-28/submissions/annotations-app-speech200.json
```

`check` chỉ kiểm cấu trúc, mức hoàn tất, ID và mốc; không tự tạo gold hoặc kết luận nhãn đúng. `prepare` từ chối thay manifest hiện có nếu đầu vào đổi; tạo phiên bản mới khi cần, giữ phiếu cũ. Không trộn bộ này với pilot25 đang chờ nhãn.

Raw video, manifest có nguồn và phiếu giữ local, đã gitignore. Metadata quyền sử dụng còn chưa xác minh; giữ provenance, không đăng lại raw video lên repo. Model app vẫn v3, chưa có train/test mới từ speech200.
