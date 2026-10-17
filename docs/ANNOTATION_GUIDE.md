# Gán nhãn pilot tiếng Việt — phiên bản 1

Mục đích: tạo nhãn người thật cho hai nhánh P0, kiểm tra pipeline và các cảnh báo nhầm. Đây là tập nhỏ chọn theo khả năng sử dụng nguồn, không đại diện cho nội dung TikTok/Reels/Shorts. Không gán nhãn từ tiêu đề, ASR hay dự đoán model.

## Phân công và cách mở

- Huy xem/nghe toàn bộ **25 clip**, dùng `review-huy.html` trong gói annotation.
- Thành viên app xem/nghe độc lập **6 clip (24%)**, dùng `review-app.html`: `001, 005, 011, 016, 022, 023`. Mẫu chéo phủ cả 5 nhóm và cả ba split; đây là mẫu kiểm tra chất lượng, không phải mẫu ngẫu nhiên để ước lượng độ đồng thuận toàn tập.
- Giải nén ZIP trước khi mở HTML bằng Chrome/Edge. Thư mục `media/` phải nằm cạnh HTML. Không cần server, model hoặc tài khoản. Trang không chứa ASR, điểm model hay nhãn của người khác.
- Tải JSON sau vài clip và trước khi đóng. Lưu tạm trong trình duyệt chỉ là tiện ích; dùng nút **Nạp lại JSON** khi đổi máy/trình duyệt. Không mở hai cửa sổ gán nhãn cùng người đồng thời.
- Huy giữ `annotations-huy.json`; thành viên app giữ `annotations-app.json`. Chỉ trao đổi sau khi cả hai hoàn tất nhãn độc lập. Không commit các file này; lưu ở `data/private/pilot-v1/submissions/`.

## Đơn vị và giá trị nhãn

Đơn vị là **toàn bộ đoạn 25 giây có sẵn**, không phải video gốc. Xem đủ hình và bật tiếng, có thể xem lại. Mốc evidence là giây tính từ đầu clip, trong `[0,25]`; xem thời gian hiện bên dưới player.

| Giá trị | Ý nghĩa |
|---|---|
| Bỏ trống / `null` | Chưa gán; không được đổi thành 0 khi tính metric |
| `0` | Đã xem/nghe đủ và không thấy rủi ro theo định nghĩa của nhãn |
| `1` | Có bằng chứng; bắt buộc ghi khoảng thời gian và mô tả ngắn |
| `uncertain` | Mờ/khó nghe/thiếu ngữ cảnh/bất đồng định nghĩa; chờ phân xử, không tự làm tròn |

Xác nhận ngôn ngữ riêng: có lời tiếng Việt / ngôn ngữ khác / không có lời / không rõ. Không suy từ tên nguồn. Không nghe rõ thì Toxic = `uncertain`, không phải 0. Video rõ ràng không có lời nói có thể nhận Toxic = 0 sau khi nghe toàn bộ; khi đánh giá hệ thống vẫn báo riêng những clip thiếu modality, không đổi điểm model `null` thành 0.

## Violence — bằng chứng hình ảnh

Gán 1 khi nhìn thấy đánh/đá/tấn công người, đánh nhau, hành động dùng vũ khí gây hại, bắn/đâm hoặc thương tích nghiêm trọng thể hiện trực tiếp. Cảnh diễn, hoạt hình hoặc tái hiện vẫn là 1 nếu có hành vi đó; ghi bối cảnh vào ghi chú. Ghi ít nhất một khoảng evidence; thêm các khoảng khác vào ghi chú nếu cần.

Gán 0 cho người đứng nói về chiến tranh/bạo lực nhưng hình ảnh không thể hiện hành vi đó, đi lại thường ngày, tranh luận không động thủ, vận động thể thao bình thường. Mang đồng phục, cầm vật dụng, có từ “đánh” trong lời nói không tự tạo nhãn Violence. Va chạm hoặc thể thao đối kháng khó phân biệt với hành hung: `uncertain`, ghi rõ lý do để hai người thống nhất.

## Toxic language — lời nói nghe được

Gán 1 khi nghe được chửi tục, nhục mạ trực tiếp, đe dọa gây hại, miệt thị/hạ nhục nhóm người hoặc kích động thù ghét. Chửi trực tiếp trong cảnh phim vẫn tính. Không dùng giọng địa phương, ý kiến chính trị, danh tính hoặc bất đồng quan điểm làm tín hiệu độc hại.

Phân biệt **nói về** một hành vi với **thực hiện** hành vi bằng lời: bản tin nói “bạo lực/chửi bới” hoặc lời khuyên phòng tránh không tự là Toxic. Khi trích nguyên lời chửi/đe dọa vẫn nghe rõ nội dung độc hại, gán 1 và ghi `quoted/educational` để phân tích ngữ cảnh riêng. Hài hước/mỉa mai mà không xác định được ý nghĩa: `uncertain`.

Chỉ xét lời nói thật trong audio cho P0. Chữ trên màn hình chưa được OCR hỗ trợ: ghi chú để nêu giới hạn, không trộn vào nhãn Toxic của nhánh audio. Khi ASR sau này sai, giữ nhãn người nghe; không sửa gold cho khớp ASR.

## Chưa hỗ trợ và phân xử

`sexual` và `dangerous_acts` không gán trong pilot P0, giữ null. Cả hai nhãn P0 bằng 0 không có nghĩa clip phù hợp cho mọi độ tuổi hoặc an toàn tuyệt đối.

Sau khi hai người nộp nhãn độc lập: Huy so sánh 6 clip chéo, ghi từng bất đồng theo nhãn, cùng xem lại và lưu quyết định + lý do + hai giá trị gốc. Không ghi đè phiếu ban đầu. Nhãn chưa thống nhất phải giữ `uncertain` và báo số lượng bị loại khỏi metric. Chỉ tính tỷ lệ đồng thuận trên các mục mà cả hai đã gán; báo support. Cohen's kappa có thể không xác định nếu toàn bộ nhãn là một lớp, không thay bằng 1.

Huy có thể gán nhãn test trước khi chạy model nhưng không dùng nhãn/ASR/ảnh test để điều chỉnh ngưỡng, chọn model hay chọn clip theo độ khó. Ngưỡng chỉ dùng train/validation; chốt xong mới đánh giá test. Nếu cả test không có dương tính thì recall/F1 lớp dương không đủ thông tin, cần báo “không ước lượng được” thay vì khẳng định model tốt. Pilot này có ít nguồn; còn cần bổ sung clip nguy hại có quyền sử dụng để đánh giá độ bao phủ.

## Kiểm tra file nộp

Tại repo, sau khi đặt JSON tải về vào thư mục riêng:

```powershell
.\.venv\Scripts\python.exe -m ml.pilot_annotation check --annotator huy --input data/private/pilot-v1/submissions/annotations-huy.json
.\.venv\Scripts\python.exe -m ml.pilot_annotation check --annotator app --input data/private/pilot-v1/submissions/annotations-app.json
```

Lệnh kiểm tra ID, phiên bản manifest, loại nhãn, xác nhận đã xem, mốc evidence và số clip hoàn thành. Đây là kiểm tra cấu trúc; không chứng minh người gán đã xem hoặc nhãn đúng. Không tự tạo gold hoặc tự sửa bất đồng.
