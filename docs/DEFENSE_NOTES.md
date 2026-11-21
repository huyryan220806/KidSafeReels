# Kịch bản demo và giải thích ML

Phiên bản 18/10/2026. Đây là kịch bản chuẩn bị, chưa phải biên bản diễn tập hoặc video đã quay. Slide có speaker notes; báo cáo và slide vẫn ghi rõ pilot chưa có nhãn.

## Demo sản phẩm 3–5 phút

Chuẩn bị: khởi động theo `RELEASE_HANDOFF.md`, kiểm tra backend model và checksum. Dùng clip train `vi-pilot-005.mp4` có quyền sử dụng cho kiểm tra kỹ thuật. Không dùng test để chọn clip biểu diễn. Kết quả của clip không phải nhãn chuẩn và không cho người gán nhãn xem trước khi họ hoàn tất phiếu. Có thể quay phần giao diện upload trước, rồi quay phần kết quả sau khi nộp nhãn.

| Thời gian | Người nói | Nội dung và thao tác |
|---|---|---|
| 0:00–0:30 | Huy | Bài toán người lớn kiểm tra video có sẵn. Hai nhãn P0: Violence và Toxic tiếng Việt. |
| 0:30–1:20 | App owner | Upload, preview, bấm kiểm tra. Chỉ rõ mode model. Giải thích giới hạn 60 giây / 50 MiB khi chờ xử lý. |
| 1:20–2:10 | Huy | Hai nhánh frame và transcript. Điểm chưa calibration; thiếu nhãn hiện chưa đánh giá. Nếu có evidence, đọc mô tả và phạm vi thời gian thật. |
| 2:10–2:45 | App owner | Phát lại evidence nếu có, tải TXT và JSON. Thử video không tiếng: Toxic null vẫn khác điểm 0. |
| 2:45–3:20 | App owner | Thử file hỏng hoặc quá 60 giây, xem thông báo lỗi; không bật mock để che lỗi model. |
| 3:20–4:15 | Huy | Nêu F1 benchmark theo từng tập; Toxic còn yếu. Pilot và thử máy thứ hai đang chờ, hoặc cập nhật đúng kết quả đã làm ở thời điểm quay. |

Nếu clip không tạo evidence, giải thích đúng trạng thái và không dựng evidence giả. Không cần mỗi ca phải flagged. Nếu model không chạy, xử lý lỗi hoặc trình chiếu video quay từ phiên bản đã kiểm chứng; ghi rõ đây là bản ghi.

## Câu hỏi bảo vệ thường gặp

**Nhóm tự huấn luyện phần nào?** Hai classifier Logistic Regression: một trên vector từ ResNet18, một trên TF-IDF của ViCTSD. ResNet18 và Whisper là pretrained. Nhóm không tuyên bố huấn luyện backbone hoặc ASR từ đầu.

**Vì sao chọn model đơn giản?** Thời gian một tuần và CPU. Backbone đóng băng giảm chi phí, classifier tuyến tính dễ tái tạo và phân tích lỗi. Cần benchmark rõ trước khi cân nhắc fine-tune.

**Tại sao đây là đa phương thức?** Hình ảnh cung cấp tín hiệu Violence; audio được chuyển thành transcript để nhận Toxic. Cả hai được gọi thật trong pipeline. OR là kết hợp quyết định giữa hai nhãn, không phải learned fusion.

**Ngưỡng lấy từ đâu?** Violence 0,43 và Toxic 0,59 được chọn theo macro-F1 trên validation của benchmark tương ứng. Không chọn bằng test pilot. Hai ngưỡng chưa được chứng minh phù hợp mọi video tiếng Việt.

**Tại sao không nói “an toàn”?** Hai nhãn P0 chỉ bao phủ một phần rủi ro; model có sai số và pilot chưa được đánh giá. Bản v1 trả flagged hoặc review, Sexual/Dangerous vẫn null.

**F1 Violence cao có nghĩa sản phẩm chính xác 96% không?** Không. Đó là F1 dương trên RLVS, với 198 dương tính trong 386 test. RLVS khác video đích và thiếu ID nguồn gốc đầy đủ. Cần đánh giá pipeline trên video chung có nhãn người thật.

**Vì sao Toxic có accuracy thấp hơn luôn dự đoán âm?** Lớp Toxic hiếm: 109/988 test. Baseline âm có accuracy 0,8897 nhưng không phát hiện Toxic. Model có accuracy 0,8684, macro-F1 0,6872 và F1 Toxic 0,4492; recall 0,4862 vẫn là điểm yếu.

**Có rò rỉ dữ liệu không?** ViCTSD loại trùng chuẩn hóa/giao split; RLVS loại trùng hash và nhóm dHash trước split; pilot chia theo nhóm nguồn trước frame. Heuristic RLVS không chứng minh độc lập nguồn hoàn toàn. Công bố giới hạn này.

**Không có audio thì sao?** Toxic null; Violence vẫn có thể vượt ngưỡng và flagged. Dưới ngưỡng không đồng nghĩa an toàn. Audio track có thể chỉ chứa nhạc, cũng chưa đủ để chấm Toxic.

**Ablation đo gì?** Bỏ một nhánh rồi xét đích “có ít nhất một rủi ro P0” trên cùng những clip đủ nhãn và score. Báo support/coverage, không so F1 hai nhánh từ hai bộ dữ liệu khác nhau để suy ra fusion tốt hơn.

**Tại sao không dùng ViHarm4Kids hoặc đủ bốn nhãn?** Nhóm chưa có dữ liệu ViHarm4Kids dùng được và chọn giảm phạm vi để hoàn thành sản phẩm trong một tuần. Hai nhãn còn lại hiển thị chưa hỗ trợ, không mượn score của nhãn khác.

**25 clip có đủ để kết luận không?** Chỉ là pilot tiện lợi, từ 8 nguồn và 5 nhóm, chưa biết support dương. Nếu test không có dương tính thì không ước lượng recall/F1 dương. Cần bổ sung tập mới có quyền và giữ test mới độc lập để nghiên cứu tiếp.

## Danh sách nghiệm thu do người thực hiện

- Huy nộp 25 nhãn, thành viên app nộp 6 độc lập, cùng phân xử bất đồng.
- Chạy `EVALUATION_GUIDE.md`, bổ sung số liệu và ca lỗi thật vào báo cáo/slide.
- App owner chạy bản tích hợp cùng 8 artifact trên máy thứ hai, ghi OS/Python/model version, ảnh hoặc log và kết quả.
- Bổ sung tên đầy đủ, MSSV, lớp, giảng viên vào bản nộp theo mẫu học phần.
- Diễn tập, quay demo 3–5 phút, ghi commit và ngày quay. Không đánh dấu các mục này chỉ vì đã có kịch bản.
