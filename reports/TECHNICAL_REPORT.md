# KidSafeReels - Báo cáo kỹ thuật

Học phần Học máy và ứng dụng. Nhóm hai người: Huy phụ trách ML và điều phối, thành viên còn lại phụ trách app. Phiên bản tích hợp CPU ngày 18/10/2026. Đây là báo cáo từ kết quả đã chạy; đánh giá pilot có nhãn và nghiệm thu máy thứ hai đang chờ.

## 1. Bài toán và phạm vi sản phẩm

Video ngắn kết hợp hình ảnh, lời nói và bối cảnh. Một bộ phân loại chỉ nhìn hình ảnh không xử lý được lời nói độc hại; một bộ phân loại văn bản không nhận biết hành động bạo lực khi không có lời. KidSafeReels thử nghiệm hai nhánh xử lý để hỗ trợ người lớn sàng lọc video trước khi cho trẻ xem. Người dùng mục tiêu là phụ huynh hoặc giáo viên có file video cần kiểm tra.

Luồng sử dụng: chọn video có sẵn, xem trước, bấm kiểm tra, đọc các nhãn được hỗ trợ và phát lại evidence, tải báo cáo TXT hoặc JSON. App nhận MP4/MOV/WebM tối đa 60 giây, 50 MiB. Phần P0 gồm Violence từ hình ảnh và Toxic tiếng Việt từ audio. Sexual/Dangerous Acts giữ null, chưa có model. OCR, nhận diện tuổi, ứng dụng mobile, tài khoản và chặn feed TikTok/Reels/Shorts nằm ngoài phạm vi một tuần.

Sản phẩm là công cụ hỗ trợ kiểm tra, không đưa ra chứng nhận “an toàn cho trẻ”. Bản hiện tại trả flagged khi có tín hiệu vượt ngưỡng và review cho các trường hợp còn lại. Không chạy mô phỏng khi model thật lỗi.

## 2. Dữ liệu, nhãn và cách chia tập

| Nguồn | Train | Validation | Test | Đơn vị và mục đích |
|---|---:|---:|---:|---|
| RLVS sau lọc | 1.157 | 386 | 386 | Video, học Violence |
| ViCTSD sau lọc | 6.965 | 1.975 | 988 | Bình luận, học Toxic |
| Pilot công khai v1 | 17 | 4 | 4 | Đoạn 25 giây, kiểm tra pipeline và chuẩn bị nhãn |

RLVS có 3.902 file trong archive tải về. Sau loại trùng SHA-256, nhãn mâu thuẫn, media quá dài hoặc lỗi, còn 1.929 video. Nhóm dùng dHash ba frame để gom các clip tương tự rồi chia tập theo nhóm với seed 42. Tập test có 198 Violence và 188 NonViolence. Do thiếu metadata video nguồn, cách nhóm gần trùng này không đảm bảo độc lập nguồn hoàn toàn; kết quả chỉ mang tính khám phá trên RLVS.

ViCTSD giữ revision cố định, loại bình luận trùng sau chuẩn hóa và giao nhau giữa các split. Tập train có 759 Toxic, validation 232 và test 109. Nhãn Toxic mất cân bằng rõ rệt. ViCTSD là bình luận viết, không phải transcript video. Nguồn chưa có LICENSE tái phân phối rõ khi kiểm tra; RLVS ghi “Data files © Original Authors”. Nhóm lưu dữ liệu thô local và không đưa lên Git.

Pilot gồm 25 đoạn từ 8 video gốc, 5 nhóm trên Wikimedia Commons. Mỗi nguồn có CC BY 3.0 và lịch sử kiểm tra tag giấy phép; manifest lưu tác giả, giấy phép, phiên bản trang, URL và hash. Các đoạn cùng nguồn hoặc bối cảnh liên quan nằm chung split trước trích frame. Tỷ lệ 17/4/4 tương đương 68/16/16. Tập thiên về tin tức/dịch vụ/giáo dục, chưa đại diện video dọc hoặc đủ ca nguy hại.

Huy được phân công gán 25 clip, thành viên app gán độc lập 6 clip (24%). Giá trị bỏ trống khác nhãn 0; nhãn chưa rõ cần phân xử, không tự biến thành âm tính. Hai người chưa thực hiện bước này tại ngày lập báo cáo. Không gán gold từ ASR, tiêu đề hoặc dự đoán model. Hướng dẫn và player độc lập đã có trong repo.

## 3. Phương pháp học máy

Nhánh hình ảnh lấy tối đa 8 frame phân bố đều trong video. ResNet18 tiền huấn luyện ImageNet đóng băng trích vector 512 chiều mỗi frame. Mean pooling và max pooling theo thời gian ghép thành 1.024 chiều. StandardScaler và Logistic Regression học trên train, C=0,1 và class_weight=balanced. Scaler chỉ fit train. Ngưỡng 0,43 được chọn theo macro-F1 validation, không dùng test để chọn. Cách này phù hợp CPU nhưng không học chuyển động và có thể bỏ sót cảnh ngắn.

Nhánh văn bản dùng TF-IDF char_wb 3-5 gram và Logistic Regression với C=1, class_weight=balanced. N-gram ký tự giúp xử lý biến thể viết nhưng không hiểu đầy đủ ngữ cảnh, mỉa mai và nội dung trích dẫn. Ngưỡng Toxic 0,59 được chọn trên validation ViCTSD. Cả hai classifier được nhóm huấn luyện thật; backbone ResNet18 và ASR là mô hình tiền huấn luyện.

Audio đi qua faster-whisper base đa ngôn ngữ, CPU int8, bốn luồng, beam_size=1, VAD, không nối previous_text. Nhánh Toxic chỉ sử dụng tiếng Việt khi language_probability đạt 0,5 và đoạn có no_speech_prob dưới 0,6. Lấy điểm Toxic cao nhất giữa các đoạn. Có audio track không chứng minh có lời nói; lỗi/thiếu audio hoặc ngôn ngữ chưa xác nhận trả Toxic=null.

Kết hợp là phép OR ở mức quyết định: nếu Violence vượt 0,43 hoặc ít nhất một đoạn Toxic vượt 0,59 thì flagged. Hai nhánh dự đoán hai nhãn khác nhau; đây không phải trung bình có trọng số của cùng lớp hoặc mô hình fusion học được. Điểm chưa hiệu chỉnh thành xác suất an toàn.

Evidence visual chỉ bao phủ toàn clip, chưa định vị cảnh bạo lực. Evidence lời nói lấy mốc ASR, giới hạn trong thời lượng clip. Trong tiền xử lý pilot, mốc thô vẫn được giữ khi phải giới hạn mốc playback. Nhóm không tạo timestamp giả từ điểm phân loại toàn clip.

## 4. Kết quả thực nghiệm trên benchmark

| Nhãn dương / benchmark | Precision | Recall | F1 dương | Macro-F1 hai lớp | Support dương |
|---|---:|---:|---:|---:|---:|
| Violence / RLVS | 0,9550 | 0,9646 | 0,9598 | 0,9585 | 198 |
| Toxic / ViCTSD | 0,4173 | 0,4862 | 0,4492 | 0,6872 | 109 |

RLVS test: TN=179, FP=9, FN=7, TP=191 trên 386 video. Visual tốt trên tập đang dùng nhưng không chứng minh mức độ phù hợp cho trẻ hoặc khái quát sang Shorts. Lấy mẫu thưa và nhóm nguồn bằng heuristic là hai hạn chế chính.

ViCTSD test: TN=805, FP=74, FN=56, TP=53 trên 988 bình luận. Accuracy model là 0,8684, thấp hơn baseline luôn dự đoán NonToxic (0,8897). Tuy vậy macro-F1 của model là 0,6872 so với 0,4708 của baseline. Điều này cho thấy accuracy dễ gây hiểu nhầm khi lớp Toxic hiếm. Recall Toxic 0,4862 vẫn bỏ sót nhiều ca; cần trình bày trung thực thay vì mô tả đây là bộ lọc đã đủ tin cậy.

Hai hàng kết quả sử dụng hai benchmark khác nhau. Không lấy trung bình các F1 này để gọi là chất lượng đa phương thức. Chưa có kết quả precision/recall/F1 của toàn pipeline trên pilot tiếng Việt chung.

Một ca smoke đã ghi nhận audio FLEURS có nội dung trung tính nhưng ASR chép sai. Điểm Toxic của đoạn ASR là 0,6333, vượt 0,59; văn bản chuẩn toàn câu nhận 0,1525. Đây là ví dụ cảnh báo nhầm liên quan ASR và cách chia đoạn, không đủ để tách riêng từng nguyên nhân hay suy ra WER chung. Không chỉnh ngưỡng theo ca này rồi gọi đó là cải thiện trên test độc lập.

## 5. Sản phẩm tích hợp và kiểm thử

Streamlit gọi FastAPI qua HTTP. API lưu upload ở thư mục tạm, worker PyAV kiểm tra container/đuôi/thời lượng và giải mã video, sau đó gọi pipeline model. Worker spawn được dùng lại để cache model. Chỉ một lượt phân tích tại một thời điểm; request khi bận trả 429. Timeout media 30 giây, model 180 giây; worker bị dừng trước khi dọn file khi timeout.

App kiểm tra mode đúng backend, evaluated_labels khớp score khác null, evidence thuộc nhãn được đánh giá và không vượt clip. Model chưa sẵn sàng trả 503, kết quả sai contract trả 502, quá thời gian trả 504. Giao diện phân biệt mock/model, thiếu nhãn và điểm 0; có preview, mốc evidence, bản tóm tắt TXT và JSON chi tiết.

Bản tích hợp giữ app commit d82798d và pipeline ML đã huấn luyện. Xung đột PyAV được giải quyết bằng 16.1.0 cho app và faster-whisper; 8 artifact khớp checksum. Hai test giả định chưa có model được đổi sang thư mục checkpoint trống riêng, vì máy Huy đã có model thật. Bộ kiểm thử hiện tại có 117 test pass, pip check không phát hiện phụ thuộc hỏng.

Kiểm thử HTTP thật qua worker/model đạt 13/13 ca. Trong đó có 5 clip train thật dài 25 giây, video tổng hợp không tiếng và có âm thử, file quá dài, file hỏng, sai đuôi/container, file rỗng và thiếu trường upload. Đây là kiểm thử kỹ thuật, không phải 13 dự đoán nội dung đúng. Kết quả score/transcript được giữ riêng để không làm lệch quá trình gán nhãn.

Lần gọi đầu mất 23,34 giây, gồm nạp worker/model. Bốn clip sau có median 5,20 giây và p95 mô tả 11,95 giây. Các clip cùng một nhóm nguồn và mỗi clip chỉ chạy một lần, nên chưa đủ làm benchmark hiệu năng. Máy chạy Windows, Python 3.12.10, AMD Ryzen 5 7535HS; model dùng CPU. Chưa nghiệm thu cài mới trên máy thứ hai hoặc Docker có model.

## 6. Pilot và quy trình đánh giá cuối

Tiền xử lý pilot đã chạy 25/25 clip, tạo 625 frame theo PTS thật và 58 đoạn ASR. Trong 15 clip có transcript, ASR nhận diện 14 là tiếng Việt và 1 là tiếng Anh; 10 clip chưa phát hiện lời nói. Đây là đầu ra tự động, không phải xác nhận nội dung bằng người thật. Bảy segment có mốc vượt biên được lưu mốc thô và giới hạn mốc dùng phát lại. Audit hiện không có lỗi hash/nhóm/timestamp và không có ứng viên gần trùng giao split theo dHash 1fps.

Công cụ pilot_evaluation đã có các bước so phiếu độc lập, tính đồng thuận, tạo quyết định phân xử, đóng gold có provenance và đánh giá theo split. Phiếu trống không tạo gold. Bất đồng hoặc nhãn chưa rõ phải có người xét và lý do; null cuối cùng được xem là loại trừ, không chuyển thành nhãn 0. Kappa không xác định khi chỉ một lớp được giữ null.

Đánh giá báo support, support dương, coverage, abstention và confusion matrix trên phần có score. Khi không có dương tính, recall/F1 dương chưa ước lượng được. Ablation bỏ nhánh visual hoặc text dùng cùng cohort đầy đủ nhãn/score, với đích “có ít nhất một rủi ro P0”. Chưa chạy thí nghiệm này vì chưa có gold; bảng kết quả sẽ bổ sung sau gán nhãn.

Test khóa theo manifest, gold, artifact, code và media. Chỉ dùng train/validation cho lựa chọn mô hình/ngưỡng. Lần đánh giá test hoàn tất không được ghi đè; cache cho phép tiếp tục đúng protocol khi gián đoạn. Nếu test thiếu ca nguy hại, cần thêm dữ liệu có quyền ở phiên bản mới, không sửa nhãn cho đủ lớp.

## 7. Đóng góp và giới hạn

Huy phụ trách phạm vi, nguồn dữ liệu, chia tập, hai baseline, ASR, fusion, manifest/artifact, công cụ nhãn/đánh giá, review và tích hợp hai nhánh, nội dung phương pháp/thực nghiệm. Thành viên app phụ trách upload, preview, trình bày kết quả, evidence playback, worker/timeout, validation, kiểm thử app, hướng dẫn chạy và giao diện dành cho phụ huynh. Hai người cùng thực hiện gán nhãn chéo, kiểm thử máy thứ hai và demo cuối.

Các giới hạn còn ảnh hưởng kết luận: chưa có nhãn pilot; Toxic còn yếu trên benchmark viết; ASR chưa đo WER/CER đại diện; thiếu chuyển động và localization ở visual; chưa đánh giá công bằng theo phương ngữ hoặc nhóm người; chưa có Sexual/Dangerous/OCR; chưa tích hợp trực tiếp vào nền tảng. Độ chính xác RLVS không được hiểu là mức bảo vệ trẻ em ngoài thực tế.

Với giới hạn một tuần, nhóm giữ hai nhãn P0 và demo local. Hạng mục phải hoàn tất trước nộp là gán nhãn và phân xử, đánh giá chung với support rõ, thử máy thứ hai, bổ sung thông tin sinh viên theo mẫu giảng viên, diễn tập và quay demo 3-5 phút. Bản báo cáo này đã chứa kết quả đã đo và phương pháp tái tạo, không đánh dấu các mục trên là đã thực hiện.

## 8. Tài liệu và dữ liệu tái tạo

Nguồn mã: https://github.com/huyryan220806/KidSafeReels, nhánh feat/ml-release. Mã huấn luyện nằm ở ml/train_visual.py và ml/toxic_baseline.py; inference ở ml/inference.py. Phiên bản/phụ thuộc: requirements-video.txt, requirements-release.lock.txt. Checksum: models/artifacts.json.

Kết quả gốc: reports/visual_baseline_metrics.json, reports/toxic_baseline_metrics.json, reports/video_smoke_results.json, reports/pilot_data_audit.json và reports/release_smoke_results.json. Dùng các tệp máy đọc được để tái tạo bảng số liệu, không nhập số từ hình ảnh.

ViCTSD: https://github.com/tarudesu/ViCTSD (revision e1ffb854ef7b7b7ae45a56d132c1f125e02e55e2). RLVS: https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset (v1). Whisper: https://github.com/SYSTRAN/faster-whisper, checkpoint Systran/faster-whisper-base revision ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66. Torchvision ResNet18: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html.

Nguồn pilot và ghi công từng đoạn: data/pilot/ATTRIBUTION.md và manifest.json. FLEURS: https://huggingface.co/datasets/google/fleurs, CC-BY-4.0, revision và tên mẫu ghi trong VIDEO_SMOKE.md. Hướng dẫn chạy bản tích hợp: docs/RELEASE_HANDOFF.md; hướng dẫn đánh giá: docs/EVALUATION_GUIDE.md. Các đường dẫn tương đối ở trên tính từ gốc repo.
