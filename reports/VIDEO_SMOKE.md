# Kiểm thử tích hợp video — bàn giao 17/10/2026

Kết quả đã lưu tại [video_smoke_results.json](video_smoke_results.json). Chạy `python -m ml.smoke_video --prepare` để tạo fixture và kiểm tra lại. Đây là FastAPI TestClient trong tiến trình, không phải kiểm thử trình duyệt/mạng hay benchmark độ chính xác.

## Kết quả thực

| Ca | Kết quả API | Thời gian request |
|---|---|---:|
| 3 clip RLVS NonViolence giữ riêng | 200; có điểm visual, `review` | 10,13 / 0,32 / 0,32 giây |
| 2 clip RLVS Violence giữ riêng | 200; `flagged`, evidence toàn clip | 1,40 / 1,44 giây |
| Video nền màu không tiếng | 200; Toxic=null, có cảnh báo thiếu audio | 0,29 giây |
| Video nền màu + giọng nói tiếng Việt | 200; cả visual và Toxic có điểm; có evidence ASR | 2,35 giây |
| Video 61 giây | 422; vượt giới hạn thời lượng | 0,02 giây |
| Byte hỏng mang đuôi .mp4 | 422; media không hợp lệ | Không đo riêng |

**9/9 ca đáp ứng kiểm tra tích hợp.** Điều này không có nghĩa 9/9 dự đoán nội dung đúng. Request đầu có thời gian nạp model; các request sau dùng cache. Những video này rất ít và không đại diện cho độ trễ thực tế; không suy ra p95 hay tốc độ xử lý mọi video dài 60 giây.

Kiểm tra lại bằng CLI ở một tiến trình mới khi tiếp tục công việc ngày 17/10 cho cùng mẫu tiếng Việt: điểm/evidence không đổi, `processing_seconds` = 39,01 giây (gồm nạp model). Sự khác biệt so với lượt dùng cache cho thấy cần đo nhiều lượt và ghi trạng thái nạp model; chưa thể hứa một mức thời gian cố định.

## Ca lỗi Toxic cần trình bày trong báo cáo

Mẫu tiếng Việt dùng audio FLEURS `vi_vn/dev`, file `10056093347220194289.wav`, ghép lên nền màu. Bản chép lời chuẩn nói về quan sát động vật, không có lời độc hại. Whisper base chép sai một đoạn từ 4,66–9,26 giây, và model Toxic chấm đoạn đó **0,6333**, vượt ngưỡng **0,59**. Vì vậy pipeline trả `flagged`: đây là một cảnh báo nhầm thực, được giữ nguyên trong JSON.

Khi đưa toàn bộ văn bản chuẩn vào cùng classifier, điểm là **0,1525**, dưới ngưỡng. So sánh này minh họa tác động của ASR và cách chia/gom đoạn, không đủ để tách riêng đóng góp của từng nguyên nhân hoặc tính chất lượng ASR nói chung. Chưa đo WER/CER trên một tập đại diện. Không điều chỉnh ngưỡng theo mẫu smoke này rồi gọi đó là cải thiện trên test độc lập.

Nguồn audio: [Google FLEURS](https://huggingface.co/datasets/google/fleurs), CC-BY-4.0, revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`; SHA-256 và bản chép lời chuẩn ở JSON. Mẫu ghép chỉ kiểm tra đường đi audio → ASR → Toxic, không phải dữ liệu moderation để huấn luyện/đánh giá.

## Kiểm tra tự động và giới hạn

24 test hiện có bao phủ API mock/lỗi upload/dọn file, baseline text, giữ null khi thiếu modality, timestamp, media/thời lượng, split nhóm, nhãn trùng mâu thuẫn, lỗi ASR và tương thích giải mã audio. `pip check` không phát hiện phụ thuộc hỏng. Ghim PyAV 16.1.0 để giải quyết lỗi faster-whisper 1.2.1 gọi tham số đã bị bỏ ở PyAV 19.

Thành viên app cần kiểm thử UI/playback và chạy gói model trên máy thứ hai. ML lead cần tập video ngắn tiếng Việt có nhãn chung để đánh giá từng nhánh và kết quả kết hợp; chưa có bằng chứng đạt chất lượng lọc nội dung cho trẻ ngoài thực tế.
