# Huy gán nhãn toàn bộ 200 video — 05/11/2026

Phiếu gồm 200 clip lời nói: hai nhóm thu thập ban đầu 100 Toxic / 100 bình thường. Đây là mô tả nguồn, không phải đáp án cần điền. Nghe từng clip để xác nhận nhãn 0 / 1 / chưa rõ.

## Mở phiếu trên máy Huy

1. Mở **Google Chrome hoặc Microsoft Edge bên ngoài chat Codex**.
2. Dán đường dẫn dưới đây vào thanh địa chỉ rồi Enter:

```text
D:\VLU\261\ML\BTL\KidSafeReels\data\private\huy-toxic-full200-2026-11-05\annotation\review-huy.html
```

Cách khác: mở thư mục `annotation` bằng File Explorer, nhấp phải `review-huy.html` → **Open with → Google Chrome**. Giữ thư mục `media` cạnh HTML. Không cần khởi động app kiểm duyệt, model hoặc server. Lệnh khởi động server ở bước trước bị bộ duyệt tự động chặn; hướng dẫn hiện tại dùng HTML trực tiếp.

Trang hiện **Người rà soát: Huy**, ban đầu **11 / 200 clip đã điền nhãn**, còn 189. 11 nhãn được giữ gồm 10 nhãn Huy đã xác nhận và một nhãn hợp lệ từ JSON speech200 trong Downloads. Không lấy đề xuất máy hoặc phiếu chưa hoàn tất của app để điền sẵn nhãn. Nhãn cũ, nguồn và bản tải ban đầu được giữ nguyên riêng.

## Làm theo từng lượt

- Có **10 lượt × 20 clip**. Chọn ở ô **Chọn lượt gán nhãn**, hoặc bấm **Đến clip chưa gán** để tới mục còn trống. Chuyển lượt vẫn giữ dữ liệu đã nhập.
- Nghe hết clip, tích đã nghe toàn bộ và chọn ngôn ngữ. Chọn nhãn 0 / 1 / chưa rõ theo audio.
- Nhãn **1**: thêm ít nhất một khoảng bằng chứng, ghi câu thực sự nghe được và mốc bắt đầu/kết thúc. Không dùng mốc máy ước lượng chưa đối chiếu.
- Nhãn **0**: đã nghe đủ và không có lời độc hại; giải thích bối cảnh ở Ghi chú nếu cần. Không thêm bằng chứng Toxic vào nhãn 0.
- Không nghe rõ hoặc phân vân: chọn **chưa rõ**, ghi lý do. Không ép số nhãn 0 và 1 bằng nhau.
- Transcript đầy đủ là tùy chọn; ưu tiên các clip khó. Chỉ tích đã đối chiếu toàn bộ transcript khi thực sự chép và kiểm đủ. Không đoán từ bị che.

Tổng thời lượng video khoảng **65 phút 41 giây**; thời gian nghe lại, ghi mốc và điền phiếu sẽ lâu hơn. Có thể chia nhiều buổi.

## Lưu để tiếp tục

Sau mỗi 5–10 clip hoặc cuối mỗi lượt, bấm **Tải JSON nhãn hiện tại**. JSON luôn chứa cả 200 clip và toàn bộ tiến độ, không chỉ lượt đang xem.

Tên file dự kiến:

```text
annotations-huy-toxic-full200-2026-11-05.json
```

Nhấn **Ctrl+J** trong Chrome/Edge để tìm bản tải mới nhất. Khi làm tiếp trên cùng trình duyệt, trang có thể khôi phục bản nháp. Nếu đổi máy/trình duyệt, dùng **Nạp lại JSON** từ phiếu full200 này. Không xóa dữ liệu trình duyệt khi chưa có JSON dự phòng. Nếu xuất bị từ chối, đọc dòng lỗi dưới nút tải và sửa nhãn/mốc được báo.

Nếu đã nhập thêm nhãn trên phiếu 20 clip hoặc phiếu cũ khác, tải JSON ở đúng phiếu đó và giữ riêng để Codex ghép có kiểm tra nguồn/bất đồng. Không nạp trực tiếp JSON của bộ khác vào full200. Không cần bỏ công việc đã làm.

Khi xong, gửi đường dẫn file JSON mới nhất cho Codex. Có thể gửi bản dở để lưu tiến độ; không gửi `annotations-huy.initial.json` thay kết quả mới.

## Giữ cách chia dữ liệu

Manifest gốc giữ 142 train / 28 validation / 30 test. Gán nhãn toàn bộ không làm test trở thành train; khi huấn luyện chỉ dùng train, chọn model/ngưỡng bằng validation. Test cũ đã được đánh giá trước đây nên không công bố như test mới độc lập. Chưa chạy model hoặc tạo metric mới trong lúc chuẩn bị phiếu.

File local `preparation-audit.json`, `preserved-label-provenance.json`, `source-splits-private.json` trong thư mục `huy-toxic-full200-2026-11-05` lưu trạng thái và nguồn nhãn. Đã kiểm checksum 200 video, cấu trúc 11 nhãn giữ lại và cú pháp JavaScript; 37 tests liên quan qua, gồm chuyển lượt và xuất/nạp JSON. Chưa kiểm trình duyệt thủ công trong đợt này.
