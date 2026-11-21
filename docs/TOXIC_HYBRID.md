# Toxic kết hợp PhoBERT và bộ luật — 29/10/2026

Huy chốt hướng Whisper → transcript tiếng Việt → PhoBERT + đối chiếu từ/cụm từ. Bản này bổ sung quyết định từ bộ luật, không huấn luyện lại hoặc thay trọng số CPU v3. [Kết quả đo](../reports/TOXIC_HYBRID_2026-10-29.md).

## Luồng xử lý

1. Whisper small beam5/VAD nhận dạng, giữ kiểm tiếng Việt và bản sửa `asr-native-filter-v2`. Không dịch sang tiếng Anh, không sửa đoán transcript.
2. Mỗi đoạn được PhoBERT chấm điểm như trước; ngưỡng 0,78, lấy max điểm theo clip.
3. Cùng transcript được đối chiếu `ml/resources/toxic_lexicon_v1.json`: 14 từ/cụm rõ ràng có thể cảnh báo, 4 viết tắt chỉ gợi ý nghe lại. Đây là bộ từ khởi đầu do dự án soạn, không phải từ điển đầy đủ hoặc bộ luật đã được chứng minh tổng quát.
4. Toxic được cảnh báo nếu model vượt ngưỡng **hoặc** có luật `flag` khớp và tín hiệu ASR đạt điều kiện dưới đây. Gợi ý `review` không tự cảnh báo. Violence giữ ngưỡng 0,41; một nhánh cảnh báo thì toàn clip `flagged`, còn lại `review`.
5. Evidence ghi nguồn model/bộ luật, cụm khớp và transcript có thể sai; thời gian bao phủ segment ASR, không phải vị trí chính xác từng từ.

Chuẩn hóa bộ luật: Unicode NFC, chữ thường, gộp khoảng trắng; giữ dấu tiếng Việt và ranh giới từ Unicode. Không khớp một phần của từ, không bỏ dấu, không fuzzy-match lỗi ASR. Không dùng các từ đơn chung “chó”, “mẹ”, “chết”, “ngu” làm luật cảnh báo. Cụm nhiều từ cần liền nhau sau gộp khoảng trắng; dấu câu chen giữa hoặc hai segment tách rời có thể làm bỏ sót.

Điều kiện **chỉ cho cảnh báo mới từ bộ luật**: `avg_logprob` hữu hạn trong [-1,0], `compression_ratio` trong (0,2.4]. Thiếu/chất lượng ngoài khoảng → chỉ gợi ý nghe lại. Không dùng riêng `no_speech_prob` để loại lại lời nói. Đây là kiểm tra heuristic, không bảo đảm transcript đúng; không làm thay đổi quyết định nhánh model cũ.

Theo phạm vi nhãn lời nói của dự án, video có từ tục nghe được vẫn có thể cần lưu ý dù đang trích dẫn hoặc khuyên không nói tục. Bộ luật không suy ra ý định xúc phạm, không đủ để phân biệt mọi bối cảnh giáo dục. Cần thống nhất tiêu chí này khi duyệt nhãn; không xem lời nói về sức khỏe/giới tính tự động là Toxic.

## Điểm và phiên bản

`scores.toxic_language` vẫn là max **điểm PhoBERT thật**; `evidence.score` là điểm PhoBERT của segment tương ứng. Không đẩy điểm lên 0,9/1 khi khớp từ khóa và không gọi điểm bộ luật là xác suất. Vì vậy clip có thể `flagged` với score thấp: app/TXT giải thích nguồn cảnh báo. Schema 1.0 không đổi, Sexual/Dangerous vẫn null.

Hậu tố phiên bản: `+toxic-hybrid-v1-56b3a177c7b2`; 12 ký tự cuối lấy từ SHA-256 lexicon. Code Git xác định logic, lexicon là file UTF-8/LF trong repo. Thiếu/sai lexicon hoặc cấu hình mode → báo model chưa sẵn sàng, không âm thầm bỏ qua.

## Chạy và đối chiếu

Tại thư mục repo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run-model.ps1 -OpenBrowser
```

Mặc định `hybrid`. Mở http://127.0.0.1:8501, chọn clip và bấm **Kiểm tra video**. Đọc phần mô tả từng đoạn để biết nguồn cảnh báo; **Chi tiết phân tích** hiển thị phiên bản/điểm/JSON, TXT cũng chứa giải thích.

Để chạy baseline, **dừng API/web hiện tại bằng Ctrl+C trong terminal đã mở app**, rồi:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run-model.ps1 -ToxicMode model_only -OpenBrowser
```

Launcher tái dùng server đang chạy đúng backend; mở launcher lần nữa không đổi cấu hình worker cũ. Sau đổi code/mode luôn dừng và mở lại. Cách dùng trực tiếp: `VideoService(directory, toxic_mode="model_only")`; biến môi trường `KIDSAFE_TOXIC_MODE=model_only` cho pipeline/API khi tự chạy, còn launcher dùng tham số `-ToxicMode`. Không sửa các ngưỡng để so sánh.

## Tái tạo đánh giá và việc còn lại

`ml.hybrid_evaluation` khóa protocol/lexicon/hash trước khi tính metric, đối chiếu cache PhoBERT bằng ID/nhãn/nhóm/ngưỡng và checksum dữ liệu ViCTSD. Tái dùng dự đoán thật của lượt 28/10; không giả lập lượt train mới. Xuất metrics CSV/JSON, biểu đồ PNG/SVG; kết quả từng hàng chỉ lưu local. Chạy với run-id mới nếu cần tái tạo, không ghi đè lượt đã báo cáo:

```powershell
.\.venv\Scripts\python.exe -m ml.hybrid_evaluation --run-id toxic-hybrid-reproduction
```

Cần dữ liệu ViCTSD và cache local của lượt 28/10. Đây là ablation **văn bản viết** trên benchmark đã báo cáo, không đo ASR hoặc cổng chất lượng audio. Quyết định rule/hybrid không có phân phối xác suất nên Loss/AUC không được tạo cho hai nhánh đó; Loss PhoBERT vẫn giữ giá trị thật. Không mở rộng từ điển hoặc chỉnh ngưỡng theo lỗi test vừa thấy.

Ba video lỗi đã được dùng phát triển; 11 clip liên quan vẫn được giữ khỏi test độc lập tương lai. Speech200 chưa có bộ nhãn người duyệt hoàn chỉnh hoặc train/test mới. Bước tiếp theo: nghe và sửa transcript chuẩn trên tập dev, hoàn tất nhãn clip/đoạn và nhóm nguồn, khóa test giữ riêng, rồi đo Precision/Recall/F1/FP cho cả pipeline. Đánh giá WER/CER cần transcript người chép; không dùng ASR làm đáp án chuẩn.
