# Rà soát 100 clip lời nói mới — 28/10/2026

**Lưu lịch sử, đã thay bằng [bộ speech200](SPEECH200_ANNOTATION.md).** Huy đã bổ sung 100 clip bình thường trong ngày 28/10. Manifest speech100 cũ giữ nguyên; HTML/ZIP dùng hiện tại nằm ở `data/private/speech200-2026-10-28`. `ml.speech_intake prepare/check` hiện làm việc với speech200, không chạy các đường dẫn mẫu speech100 bên dưới. Phần còn lại ghi trạng thái tại lần tiếp nhận đầu, khi chưa có nhóm âm tính; chưa có phiếu speech100 hoàn tất.

Bộ local ở `data/Video_Train/Toxic/KidSafeReels-100-Clip-LoiNoi`: 100 clip từ 74 video nguồn, tổng khoảng 32 phút. Huy mô tả tập này đều có chửi tục/lăng mạ. Metadata nguồn cho biết nhãn được gợi ý bằng ASR/từ khóa, chưa có phiếu nghe xác nhận từng clip. Lưu hai loại thông tin riêng: `user_reported_clip_label=1` theo lời Huy; `human_label=null` cho đến khi có phiếu từng clip. Không tự đổi lời xác nhận ở cấp tập thành transcript/nhãn đoạn đã kiểm chứng.

## Mở phiếu

Phiếu Huy: `data/private/speech100-2026-10-28/review-huy.html` (100 clip). Phiếu thành viên app: `review-app.html` (20 clip, mỗi clip từ một video nguồn khác nhau, seed42). Mở bằng Chrome/Edge; thư mục `media/` phải nằm cạnh HTML. Mỗi lần chỉ mở một phiếu của một người.

1. Chọn clip, nghe đủ cả đoạn rồi xác nhận **Tôi đã xem/nghe toàn bộ clip** và ngôn ngữ.
2. Chọn nhãn 1/0/chưa rõ. Clip chưa xử lý để trống; không tự coi trống là 0.
3. Với nhãn 1, thêm ít nhất một khoảng bằng chứng và câu/lý do nghe được. Có nút lấy mốc đang xem. Mốc tính từ đầu clip, không phải video YouTube gốc.
4. Có thể chép/sửa transcript. Chỉ tích xác nhận transcript khi đã đối chiếu toàn bộ với audio. Không đoán từ bị che hoặc không nghe rõ.
5. **Tải JSON nhãn hiện tại** sau vài clip và trước khi đóng. Phiếu chưa xong vẫn tải được nếu những nhãn đã điền có đủ thông tin. Local storage là bản nháp tiện dụng; file JSON là bản sao có thể nạp lại/đổi máy.

Thành viên app gán độc lập, không xem phiếu Huy trước. Chưa gửi tin nhắn hoặc phiếu cho thành viên app tự động. Hai phiếu không chứa transcript ASR, điểm dự đoán hay nhãn gợi ý nguồn để tránh dẫn dắt người nghe. Muốn gán 0 hoặc chưa rõ khác với mô tả ban đầu vẫn được; giữ bằng chứng và ghi chú để phân xử.

## Tiêu chí và nhãn đoạn

Theo [quy tắc Toxic hiện có](ANNOTATION_GUIDE.md): chửi tục, lăng mạ, đe dọa gây hại, miệt thị nghe được → 1. Giọng nói lớn, phương ngữ, bất đồng quan điểm hoặc chỉ nhắc tới “chửi bới” không tự là Toxic. Nếu trích lời chửi vẫn nghe rõ thì ghi bối cảnh trích dẫn. Không nghe rõ/thiếu ngữ cảnh → `uncertain`.

Một clip có Toxic không có nghĩa mọi câu trong clip đều Toxic. Chưa gán nhãn 1 cho toàn bộ các đoạn ASR để fine-tune PhoBERT. Các khoảng bằng chứng hỗ trợ tách phần độc hại và phần nói chuyện bình thường sau khi đã kiểm transcript. Đây là nguyên nhân kỹ thuật cần rà soát trước train, không phải yêu cầu bạn điền lại nhãn chỉ để có hình thức.

## Chia nguồn và đánh giá

74 ID YouTube mới là nhóm nguồn tạm thời; video reupload cùng buổi nói/stream có thể có ID khác. Không chia ngẫu nhiên 100 clip thành train/test. Giữ các đoạn cùng video và cùng nguồn gốc đã xác định trong một split; kiểm reupload trước khi khóa split, trích đặc trưng/ASR và train.

Huy xác nhận chưa có video bình thường và sẽ thêm sau. Bộ hiện tại thiên về lớp dương: chưa thể đánh giá false-positive rate, specificity hoặc macro-F1 hai lớp đáng tin cậy. Tỷ lệ phát hiện trên một tập toàn Toxic không thay thế chất lượng kiểm duyệt chung. Không ghép bình luận ViCTSD bình thường vào test video rồi coi đó là cùng miền dữ liệu.

50 file `data/Video_Train/Violence` cần đối chiếu với AIRTLab đã đánh giá: bản trùng không phải dữ liệu mới và không đưa vào train khi vẫn dùng kết quả AIRTLab cũ làm test ngoài.

## Kiểm tra phiếu và dữ liệu

```powershell
# Tạo/kiểm lại bộ local, giữ phiếu nộp riêng.
.\.venv\Scripts\python.exe -m ml.speech_intake prepare
# Sau khi tải JSON, đặt vào thư mục submissions tự tạo dưới data/private/speech100-2026-10-28.
.\.venv\Scripts\python.exe -m ml.speech_intake check --annotator huy --input data/private/speech100-2026-10-28/submissions/annotations-huy-speech100.json
.\.venv\Scripts\python.exe -m ml.speech_intake check --annotator app --input data/private/speech100-2026-10-28/submissions/annotations-app-speech100.json
```

Lệnh `check` chỉ kiểm cấu trúc, ID, nhãn, mốc và mức hoàn tất; không chứng minh nhãn đúng hoặc tạo gold. Sau đó mới đối chiếu hai phiếu/phân xử, giữ nguyên phiếu gốc. Hướng dẫn/tập mới không thay thế pilot25 đang chờ nhãn.

Raw video, manifest có URL/tên nguồn và phiếu nộp giữ local; `data/Video_Train/` được Git ignore toàn bộ. Giữ metadata quyền sử dụng như nguồn cung cấp, không tự suy quyền tái phân phối. Bản model đang chạy vẫn v3; chưa có lượt train hoặc accuracy mới từ bộ này.
