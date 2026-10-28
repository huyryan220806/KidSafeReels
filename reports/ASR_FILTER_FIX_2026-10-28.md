# Sửa mất transcript trước bộ phân loại Toxic — 28/10/2026

Huy báo ba clip Toxic trong bộ speech200 bị trả kết quả chưa rõ. Đã chạy lại đúng ba clip qua code local và API đang chạy; API dùng PhoBERT/R3D-18/Whisper small beam5 thật, không phải mock. Đây là chẩn đoán ca lỗi người dùng, không phải test ngẫu nhiên hay lượt huấn luyện mới.

## Nguyên nhân xác minh được

Ở hai clip, Whisper nhận diện `vi` (điểm ngôn ngữ lần lượt khoảng 0,866 và 0,755), trả 6 và 4 đoạn. Tuy nhiên `ml/inference.py` lọc lần hai bằng điều kiện đơn `no_speech_prob < 0.6`, làm mất cả 10 đoạn và trả Toxic `null`.

Mã thư viện faster-whisper đã cài trong môi trường có bước lọc im lặng trước khi yield segment: dùng cả `no_speech_prob` và `avg_logprob`. Với hai clip này, `no_speech_prob` khoảng 0,892/0,773 nhưng `avg_logprob` khoảng -0,476/-0,762, cao hơn ngưỡng mặc định -1,0 nên decoder vẫn giữ lời nói. Việc lọc lại chỉ bằng xác suất không lời nói đã ghi đè quyết định đó. Đây là lỗi tích hợp, không phải bằng chứng clip không có tiếng Việt.

Clip thứ ba đã có 7 đoạn tới classifier từ đầu; điểm Toxic khoảng 0,022752, thấp hơn ngưỡng 0,78. Có từ tục trong ASR nhưng score đoạn vẫn thấp; cần kiểm audio/transcript tham chiếu và bổ sung dữ liệu đúng định nghĩa nhãn. Không quy toàn bộ ba lỗi cho ASR hoặc hạ ngưỡng để ép ba clip qua.

## Thay đổi và kết quả

Giữ VAD, kiểm ngôn ngữ tiếng Việt và bộ lọc gốc của decoder. Chuyển các đoạn decoder đã chấp nhận, có text không rỗng sang classifier; bỏ bước lọc `no_speech_prob` đơn lẻ. Giữ nguyên trọng số, ngưỡng Toxic 0,78/Violence 0,41 và schema API. `model_version` thêm hậu tố `+asr-native-filter-v2` để phân biệt hành vi suy luận mới.

| Ca người dùng | Trước sửa | Sau sửa qua HTTP |
|---|---|---|
| vi-speech-048 | Toxic null, review | Toxic 0,901374, flagged; 6 đoạn được phân loại |
| vi-speech-019 | Toxic null, review | Toxic 0,002520, review; 4 đoạn được phân loại |
| vi-speech-022 | Toxic 0,022752, review | Toxic 0,022752, review; 7 đoạn được phân loại |

Một ca đã được cảnh báo; hai ca vẫn bỏ sót theo xác nhận của Huy. Chưa có transcript người sửa, nhãn từng đoạn hoặc bộ test đủ hai lớp để đo Accuracy/Loss/FPR mới. Score không phải xác suất đã hiệu chỉnh. Không lấy 1/3 làm accuracy tổng quát.

## Kiểm chứng và tái hiện

- Regression mới tái hiện lỗi trước sửa: 1 failed / 4 passed, đúng ca decoder giữ lời nói nhưng pipeline loại bỏ.
- Sau sửa: **63 tests pass**, gồm `test_video_pipeline.py`, `test_inference_config.py`, `test_api.py`, `test_web.py`, `test_web_design.py`. Có ca giữ lời nói đã được decoder chấp nhận; transcript rỗng, không tiếng Việt và nhận diện ngôn ngữ yếu vẫn không tạo điểm Toxic.
- Ba ca người dùng trả HTTP 200, local/API cùng score và có hậu tố phiên bản mới. Đã nạp lại riêng worker model của API hiện hành, giữ web và API cha đang chạy.
- **4/4 kiểm HTTP bổ sung pass:** im lặng và âm đơn tần vẫn Toxic null/không có evidence lời nói; hai clip pilot thuộc train vẫn có điểm Toxic. Đây chỉ là kiểm kỹ thuật, không coi hai clip pilot là nhãn âm tính đã xác minh hoặc đo false-positive rate. 17 artifact trọng số/cấu hình vẫn khớp checksum. [Biên bản máy đọc](ASR_FILTER_FIX_2026-10-28.json).
- Raw ASR/score từng đoạn, JSON trước/sau, script chẩn đoán và đối chứng nằm local ở `runs/speech-toxic-diagnostic-2026-10-28/`; không đưa transcript/video lên Git.
- Trước suy luận đã ghi `development-exposure-user-failures-2026-10-28.json` bên cạnh manifest speech200: 3 clip đã xem, **11 clip liên quan** qua cùng nguồn hoặc ứng viên dHash được giữ khỏi test độc lập tương lai. Đây là ghi nhận sử dụng dữ liệu, không phải split train/validation/test đã khóa hoặc nhãn gold.

Huy cũng nói đã thử nhiều video khác nhưng chưa có đủ tên file. 11 clip là danh sách loại khỏi test **tối thiểu đã xác định**, chưa chứng minh toàn bộ phần còn lại chưa từng được dùng để chọn/sửa model. Khi khóa test cần rà lịch sử đã thử; nếu thiếu dữ liệu chưa xem thì báo đánh giá development hoặc bổ sung nguồn test mới.

Các kết quả classifier ViCTSD/RLVS đã báo trước đó không thay đổi. Số đo ASR/ablation lịch sử dùng bộ lọc cũ vẫn là lịch sử; `ml/asr_experiments.py` và `ml/asr_toxic_audit.py` chưa đổi protocol để tái tạo run cũ. Khi làm thí nghiệm cho pipeline mới phải dùng run ID/protocol mới và khớp bộ lọc `asr-native-filter-v2`, không áp số đo cũ thành kết quả mới.

## Việc cần làm tiếp

Ưu tiên Huy nghe/sửa transcript và đánh dấu câu độc hại của ba clip này để tách lỗi nhận dạng khỏi lỗi classifier. Với speech200, hoàn tất nhãn cả hai nhóm và nguồn/reupload; đưa nguồn đã xem vào development, giữ test chưa dùng để sửa model. Chạy ASR theo protocol mới, tạo nhãn đoạn đã duyệt, fine-tune và chọn ngưỡng trên validation. Chỉ công bố Accuracy/Loss/Precision/Recall/F1, false positives và đồ thị/CSV sau đánh giá tập test giữ riêng.

Để thử lại trên máy Huy: web `http://127.0.0.1:8501`, chọn clip rồi **bấm Kiểm tra video lại**; kết quả cũ trong phiên không tự cập nhật. Chi tiết phiên bản cần kết thúc bằng `+asr-native-filter-v2`. Máy khác lấy code mới, dừng/mở lại app; không cần tải lại ZIP trọng số v3.
