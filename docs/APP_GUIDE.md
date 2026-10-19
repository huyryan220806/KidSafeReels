# Hướng dẫn sử dụng và demo app

**Giao diện cập nhật 17/10/2026:** bố cục dành cho phụ huynh với ba bước Chọn video → Kiểm tra nội dung → Xem kết quả. Nút thao tác chính là **Kiểm tra video**. **Lưu kết quả về máy** tải bản tóm tắt tiếng Việt dạng TXT; JSON và chỉ số chi tiết nằm trong mục **Chi tiết phân tích**. Hướng dẫn lần đầu và thông tin kết nối nằm cuối trang. Mô phỏng luôn được ghi rõ và không hiển thị điểm đánh giá giả.

## Khởi động trên Windows

Bản đã tích hợp model: mở `Chay-KidSafeReels.cmd` hoặc chạy `powershell -ExecutionPolicy Bypass -File scripts/run-model.ps1`. Dùng `.venv-model` hoặc `.venv` đã cài đủ phụ thuộc và checkpoint; xem `README.md` để cài trên máy mới. Hướng dẫn dưới đây vẫn dùng cho bản app/mock nhẹ.

Mở PowerShell tại repo (máy hiện tại: `D:\KidSafeReels`):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
powershell -ExecutionPolicy Bypass -File scripts/run.ps1
```

Chỉ cần setup ở lần đầu hoặc khi dependency thay đổi. Mở http://127.0.0.1:8501. Script từ chối chạy nếu cổng 8000/8501 đang bị dùng; không tự dừng chương trình khác. Nhấn Ctrl+C trong terminal để đóng app; đợi request đang xử lý kết thúc hoặc chạm timeout.

## Dùng app

1. Kiểm tra nhãn trạng thái ở đầu trang và thông báo bên dưới phần giới thiệu. Nhãn “Bản dùng thử” kèm thông báo mô phỏng nghĩa là chưa phân tích rủi ro nội dung. Mục hướng dẫn cuối trang có nút thử kết nối lại.
2. Chọn video MP4/MOV/WebM, tối đa 50 MiB, 60 giây. Có thể xem trước bằng trình phát. MOV/codec không phát được trên trình duyệt nên xuất lại MP4/H.264.
3. Bấm **Kiểm tra video**. Trạng thái chờ không hiển thị phần trăm giả.
4. Đọc kết luận, điểm, nhóm chưa đánh giá và cảnh báo. `Chưa đánh giá` khác `0.000`.
5. Khi model trả evidence, chọn mốc để trình phát bắt đầu từ giây đó; UI làm tròn xuống giây để không bỏ đầu đoạn. Xem nguồn hình ảnh/lời nói/OCR và mô tả.
6. Bấm **Lưu kết quả về máy** để tải bản tóm tắt dễ đọc. Khi cần JSON, mở **Chi tiết phân tích** rồi bấm **Tải dữ liệu JSON**. Kết quả còn trong phiên khi tương tác; đổi/xóa file sẽ xóa kết quả cũ. API luôn dọn file tạm sau request.

## Xử lý sự cố

| Tình huống | Cách xử lý |
|---|---|
| API chưa kết nối | Kiểm tra terminal, chạy lại script; bấm Kiểm tra kết nối |
| Model chưa sẵn sàng | Nhờ Huy kiểm tra dependencies/checkpoint; dùng mock để làm UI và nói rõ đang mô phỏng |
| File hỏng/không phải video | Xuất lại video hợp lệ; đổi đuôi không sửa được nội dung |
| Quá 60 giây/50 MiB | Cắt ngắn hoặc xuất nhẹ hơn |
| Máy bận | Chờ request trước kết thúc rồi bấm lại; app không tự gửi lại |
| Quá thời gian | Thử clip ngắn/độ phân giải thấp; kiểm tra hiệu năng model |
| Không có âm thanh | Đọc cảnh báo và giới hạn đánh giá; không suy ra ngôn ngữ an toàn |
| Kết quả không đúng contract | Huy và app owner đối chiếu API_CONTRACT.md; không sửa điểm để che lỗi |

## Bộ video tổng hợp

```powershell
.\.venv\Scripts\python.exe -m scripts.make_demo_videos
```

Tám file ở `data/private/app-demo/`: MP4 không tiếng, MP4 có âm thử, MOV, WebM, video đúng 60 giây, video 61 giây, file hỏng và file rỗng. Video chỉ có hình khối/màu và âm sine; không có nhãn nội dung thật, không dùng đánh giá độ chính xác model. Thư mục đã gitignore.

## Kịch bản demo 3–5 phút

- 0:00–0:30: giới thiệu phụ huynh/giáo viên sàng lọc video trước khi chia sẻ.
- 0:30–1:30: upload clip hợp lệ, xem trước, phân tích. Khi chưa có model, nói rõ đây là kiểm tra app với mock và mọi điểm chưa đánh giá.
- 1:30–2:30: sau khi Huy bàn giao model, trình bày kết luận/nhãn/evidence thật và phát tại mốc. Trước bàn giao, không dựng điểm giả để thay thế phần này.
- 2:30–3:15: tải JSON; đổi video để thấy kết quả cũ được xóa; thử file hỏng hoặc 61 giây.
- 3:15–4:00: thử video không tiếng và giải thích giới hạn.
- 4:00–5:00: giải thích kiến trúc, thời gian đo thật và trách nhiệm hai thành viên. Chất lượng ML do Huy báo cáo bằng metric thật.

Ảnh UI được lưu tại `docs/screenshots/`. Quay demo cuối có model và chạy trên máy Huy sau khi ghép bản tích hợp; không coi video mock là demo chất lượng model.

## Cấu hình thời gian

PowerShell đặt biến môi trường trước khi chạy:

```powershell
$env:KIDSAFE_MEDIA_TIMEOUT = '30'
$env:KIDSAFE_MODEL_TIMEOUT = '180'
$env:KIDSAFE_HTTP_TIMEOUT = '225'
```

Web timeout phải lớn hơn tổng hai timeout server cộng thời gian gửi file. Nếu dùng Compose, sửa cả environment và timeout nginx tương ứng; `.env.example` không được tự nạp bởi app local.
