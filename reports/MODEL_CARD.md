# Model card — KidSafeReels CPU baseline v1

**Bổ sung 18/10/2026:** bản tích hợp `feat/ml-release` đã chạy 13/13 ca HTTP với worker/model thật, 117 tests pass. Các kết quả benchmark bên dưới không đổi. Chưa có nhãn/metric pilot hoặc nghiệm thu máy thứ hai. Xem `docs/RELEASE_HANDOFF.md`.

Cập nhật 16/10/2026. Pipeline local kết hợp `violence-resnet18-v1`, `toxic-tfidf-v1` và Whisper base. [Cách chạy](../docs/VIDEO_PIPELINE.md), [artifact/checksum](../models/artifacts.json), [visual metrics](visual_baseline_metrics.json), [text metrics](toxic_baseline_metrics.json), [kiểm thử tích hợp](video_smoke_results.json).

## Mục đích và phạm vi

Hỗ trợ người lớn sàng lọc video ≤60 giây trong demo học phần. Hình ảnh cung cấp điểm Violence, lời nói tiếng Việt cung cấp điểm Toxic. Không có model Sexual/Dangerous Acts, OCR, nhận diện tuổi hoặc chặn feed TikTok/Reels/Shorts. Không được diễn giải kết quả thành chứng nhận video an toàn cho trẻ.

## Dữ liệu và mô hình

- **Visual:** RLVS Kaggle v1; 3.902 file trong archive, còn 1.929 video sau loại bản trùng, một video có nhãn mâu thuẫn, video quá dài/lỗi giải mã. Train/validation/test = 1.157/386/386; SHA-256 dedup và nhóm dHash trước chia tập, seed 42. Thiếu ID nguồn gốc, nên phép nhóm này chưa đảm bảo độc lập nguồn hoàn toàn. [Manifest chia tập](../data/splits/rlvs-v1.json).
- **Backbone:** ResNet18 ImageNet đóng băng, chuẩn hóa theo Torchvision; tối đa 8 frame lấy đều. Mean/max pooling cho vector 1.024 chiều; StandardScaler + Logistic Regression C=0,1, balanced weights. Scaler/classifier fit train, ngưỡng chọn bằng macro-F1 validation, giữ test riêng.
- **Text:** ViCTSD từ commit ghim; 6.965/1.975/988 bình luận sau làm sạch. TF-IDF char_wb 3–5 gram + Logistic Regression C=1, balanced weights; ngưỡng 0,59. Test F1 Toxic=0,4492, recall=0,4862; bỏ sót 56/109 ca Toxic. Đây là benchmark bình luận, khác lời nói ASR.
- **ASR:** faster-whisper base đa ngôn ngữ, CPU int8, 4 luồng, beam_size=1, VAD, không dùng previous_text. Chỉ dùng tiếng Việt với language_probability ≥0,5 và đoạn no_speech_prob <0,6. Chưa có WER/CER đại diện.
- **Kết hợp:** max Toxic theo đoạn; Violence cấp clip. Chỉ cần một nhánh đạt/vượt ngưỡng để `flagged`; còn lại `review`. Hai nhánh không dự đoán cùng một nhãn, không có weighted fusion được học.

## Đánh giá và giới hạn

[Báo cáo Visual](VISUAL_BASELINE.md) và [báo cáo Toxic](TOXIC_BASELINE.md) ghi metric thật trên hai benchmark riêng. Smoke test gọi FastAPI qua TestClient và kiểm tra đường đi model/lỗi; chưa phải kiểm thử trình duyệt hay đánh giá moderation trên tập video tiếng Việt chung. Chưa có số liệu chứng minh hệ thống bảo vệ trẻ em trong thực tế.

- Lấy frame thưa có thể bỏ sót hành động nhanh; không học chuyển động. Evidence visual `[0, duration]` không định vị đoạn bạo lực.
- Sai ASR, nói nhỏ/nhạc nền/phương ngữ và chia đoạn có thể đổi điểm Toxic. Đã gặp cảnh báo nhầm trên một mẫu FLEURS có lời nói trung tính; xem báo cáo smoke.
- Score chưa calibration. Ngưỡng Toxic chọn trên bình luận, chưa được xác nhận cho max nhiều đoạn của một video. Thiếu audio/không nhận ra tiếng Việt/ASR lỗi → Toxic=null, không suy ra an toàn.
- RLVS có khác biệt về bối cảnh/camera so với video dọc và thiếu metadata nguồn; vẫn có nguy cơ gần trùng chưa phát hiện.
- Chưa đánh giá công bằng theo nhóm/ngôn ngữ/phương ngữ, chưa đo latency đại diện hay kiểm thử máy thứ hai. App tích hợp có timeout media 30 giây/model 180 giây; chưa nghiệm thu triển khai public.

## Artifact và dữ liệu

Checkpoint/raw media ở local, không được commit. Gói inference tạo bởi `ml.package_models` không chứa dữ liệu huấn luyện. Chỉ nạp joblib từ nguồn tin cậy. Nguồn ViCTSD chưa có LICENSE rõ ràng khi kiểm tra; RLVS ghi “Data files © Original Authors”. Không suy từ khả năng tải về thành quyền tái phân phối dữ liệu. Mẫu giọng nói FLEURS có CC-BY-4.0, nguồn và revision được ghi trong report smoke.

Máy thí nghiệm: Windows, Python 3.12.10, AMD Ryzen 5 7535HS (6 core/12 thread); pipeline dùng CPU. GPU RTX 4060 Laptop có trên máy nhưng không dùng trong lượt này. Chi tiết phiên bản phụ thuộc: `requirements-ml.txt`, `requirements-video.txt` và `docs/VIDEO_PIPELINE.md`.
