# Nghiệm thu trên máy thành viên app

**Mẫu trống — chưa nghiệm thu.** Copy thành `docs/APP_ACCEPTANCE_<ten-may-quy-uoc>.md`, điền từ lần chạy thật. Log, JSON dự đoán, phiếu nhãn và video gốc để `data/private/`; chỉ commit kết quả tóm tắt và ảnh đã kiểm tra thông tin riêng.

## Môi trường

| Mục | Giá trị thực tế |
|---|---|
| Người kiểm tra / ngày | Chưa ghi |
| Máy thứ hai hay venv mới trên máy Huy | Chưa xác nhận |
| OS / Python / CPU / RAM | Chưa ghi |
| Nhánh / commit | Chưa ghi |
| Model version / số artifact kiểm hash thành công | Chưa chạy |
| PyAV / faster-whisper / torch | Chưa ghi |
| Backend thực tế của kết quả upload | Chưa chạy |

## Kết quả kiểm thử

| Hạng mục | Kết quả | Bằng chứng / ghi chú |
|---|---|---|
| Cài mới và `pip check` | Chưa chạy | |
| `ml.release_verify` | Chưa chạy | |
| `pytest -q` | Chưa chạy | Ghi passed/failed/skipped và lỗi, không sao chép 117 từ máy Huy |
| Upload / preview / kết quả model thật | Chưa chạy | |
| Train 005–009 sau khi hoàn tất nhãn độc lập | Chưa chạy | Không dùng kết quả làm nhãn |
| Thiếu audio / Toxic null | Chưa chạy | |
| Sexual/Dangerous chưa đánh giá | Chưa chạy | |
| Evidence / playback khi có | Chưa chạy | Nếu không có evidence, ghi chưa kiểm tra thay vì pass |
| Tải TXT/JSON, đổi file reset kết quả | Chưa chạy | |
| File hỏng / rỗng / quá dài | Chưa chạy | |
| API dừng / thiếu checkpoint | Chưa chạy | |
| Worker bận / timeout | Chưa chạy | Nêu tự động hay thủ công |
| Gán 6 clip độc lập, xuất/kiểm tra JSON | Chưa thực hiện | Chỉ ghi trạng thái, không chép nhãn vào tài liệu công khai |
| Quay demo 3–5 phút | Chưa thực hiện | Commit, ngày quay, nơi giữ video |

## Độ trễ

Ghi thời gian từ gửi HTTP đến nhận kết quả, tách lần đầu/lần sau. Đo này không thay thế đánh giá chất lượng nội dung.

| Clip train hoặc fixture | Thời lượng video | Lần đầu / lần sau | Số lần đo | Thời gian từng lần | Kết quả/lỗi |
|---|---|---|---|---|---|
| Chưa đo | | | | | |

## Lỗi và bàn giao lại

Với mỗi lỗi: bước tái hiện, kết quả mong đợi/thực tế, mã HTTP nếu có, log đã bỏ thông tin riêng, commit sửa và kiểm tra lại. Nếu chưa xử lý được, ghi rõ nguyên nhân và người phụ trách.

Trạng thái kết luận: **Chưa nghiệm thu**. Chỉ đổi sau khi đã chạy và có bằng chứng; nêu rõ phần chưa kiểm tra. Huy nhận JSON nhãn riêng sau khi cả hai hoàn tất độc lập để đối chiếu và đánh giá ML.
