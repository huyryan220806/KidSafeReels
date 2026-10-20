# Model card — KidSafeReels CPU v2

**Cập nhật 20/10/2026:** bundle v2 nâng ASR lên Whisper small beam5; giữ hai classifier v1 sau khi thử 6 cấu hình Toxic và 9 cấu hình Violence. [Kết quả train/validation/test, CSV và hình](MODEL_IMPROVEMENT_2026-10-20.md), [kiểm thử tích hợp](model-improvement-2026-10-20/integration_checks.json), [cài v2](../docs/MODEL_UPDATE_2026-10-20.md).

Pipeline local kết hợp `violence-resnet18-v1`, `toxic-tfidf-v1` và `whisper-small-beam5`. [Artifact/checksum](../models/artifacts.json). Baseline/report ngày 16–19/10 được giữ làm lịch sử; nghiệm thu v1 trên máy Tiến không thay thế kiểm tra v2 trên máy đó.

## Mục đích và phạm vi

Hỗ trợ người lớn sàng lọc video ≤60 giây trong demo học phần. Hình ảnh cung cấp điểm Violence, lời nói tiếng Việt cung cấp điểm Toxic. Không có model Sexual/Dangerous Acts, OCR, nhận diện tuổi hoặc chặn feed TikTok/Reels/Shorts. Không được diễn giải kết quả thành chứng nhận video an toàn cho trẻ.

## Dữ liệu và mô hình

- **Visual:** RLVS Kaggle v1; 3.902 file trong archive, còn 1.929 video sau loại bản trùng, một video có nhãn mâu thuẫn, video quá dài/lỗi giải mã. Train/validation/test = 1.157/386/386; SHA-256 dedup và nhóm dHash trước chia tập, seed 42. Thiếu ID nguồn gốc, nên phép nhóm này chưa đảm bảo độc lập nguồn hoàn toàn. [Manifest chia tập](../data/splits/rlvs-v1.json).
- **Backbone:** ResNet18 ImageNet đóng băng, chuẩn hóa theo Torchvision; tối đa 8 frame lấy đều. Mean/max pooling cho vector 1.024 chiều; StandardScaler + Logistic Regression C=0,1, balanced weights. Scaler/classifier fit train, ngưỡng chọn bằng macro-F1 validation, giữ test riêng.
- **Text:** ViCTSD từ commit ghim; 6.965/1.975/988 bình luận sau làm sạch. TF-IDF char_wb 3–5 gram + Logistic Regression C=1, balanced weights; ngưỡng 0,59. Test F1 Toxic=0,4492, recall=0,4862; bỏ sót 56/109 ca Toxic. Đây là benchmark bình luận, khác lời nói ASR.
- **ASR:** faster-whisper small đa ngôn ngữ, CPU int8, 4 luồng, beam_size=5, temperature=0, VAD, không nối previous_text, task=transcribe. Tự nhận dạng ngôn ngữ; chỉ dùng tiếng Việt với language_probability ≥0,5 và đoạn no_speech_prob <0,6. Revision `536b0662742c02347bc0e980a01041f333bce120`. Chọn trên 16 câu dev FLEURS, đo trên 16 câu test khác prompt ID: WER 19,83%, CER 10,70% (base: 38,36%/22,40%). Đây là pretrained ASR, không fine-tune; mẫu nhỏ đọc thành tiếng chưa đại diện video mạng xã hội.
- **Kết hợp:** max Toxic theo đoạn; Violence cấp clip. Chỉ cần một nhánh đạt/vượt ngưỡng để `flagged`; còn lại `review`. Hai nhánh không dự đoán cùng một nhãn, không có weighted fusion được học.

## Đánh giá và giới hạn

[Báo cáo mới](MODEL_IMPROVEMENT_2026-10-20.md) đo cả baseline và ứng viên chọn bằng validation trên các split cũ. Toxic ứng viên giảm FP nhưng recall test giảm 48,62% → 44,04%; Violence 24 frame giảm F1 test 95,98% → 94,74%. Cả hai bị loại khỏi release, không chọn lại dựa trên test. Full regression 146 tests pass; kiểm thử HTTP và Streamlit AppTest với v2 ghi riêng trong báo cáo tích hợp. Chưa có nhãn/metric moderation trên tập video tiếng Việt chung, chưa có số liệu chứng minh hệ thống bảo vệ trẻ em trong thực tế.

- Lấy frame thưa có thể bỏ sót hành động nhanh; không học chuyển động. Evidence visual `[0, duration]` không định vị đoạn bạo lực.
- Sai ASR, nói nhỏ/nhạc nền/phương ngữ và chia đoạn có thể đổi điểm Toxic. Small giảm WER/CER trong tập FLEURS nhỏ và đổi kết quả mẫu smoke cũ, nhưng chưa kiểm chứng trên các clip Huy báo lỗi vì chưa lưu clip. Classifier Toxic vẫn yếu trên benchmark bình luận.
- Score chưa calibration. Ngưỡng Toxic chọn trên bình luận, chưa được xác nhận cho max nhiều đoạn của một video. Thiếu audio/không nhận ra tiếng Việt/ASR lỗi → Toxic=null, không suy ra an toàn.
- RLVS có khác biệt về bối cảnh/camera so với video dọc và thiếu metadata nguồn; vẫn có nguy cơ gần trùng chưa phát hiện.
- Chưa đánh giá công bằng theo nhóm/ngôn ngữ/phương ngữ hoặc đo latency đại diện. V1 đã kiểm trên máy thứ hai; v2 còn cần kiểm lại. App tích hợp có timeout media 30 giây/model 180 giây; chưa nghiệm thu triển khai public.

## Artifact và dữ liệu

Checkpoint/raw media ở local, không được commit. Gói v2 tạo bởi `ml.package_improved` không chứa dữ liệu huấn luyện; giữ ZIP/manifest v1 để khôi phục. Chỉ nạp joblib từ nguồn tin cậy. Nguồn ViCTSD chưa có LICENSE rõ ràng khi kiểm tra; RLVS ghi “Data files © Original Authors”. Không suy từ khả năng tải về thành quyền tái phân phối dữ liệu. FLEURS có CC-BY-4.0, revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`; protocol, provenance và nguồn ghi trong thư mục thực nghiệm.

Máy thí nghiệm: Windows, Python 3.12.10, AMD Ryzen 5 7535HS (6 core/12 thread); pipeline dùng CPU. GPU RTX 4060 Laptop có trên máy nhưng không dùng trong lượt này. Chi tiết phiên bản phụ thuộc: `requirements-ml.txt`, `requirements-video.txt` và `docs/VIDEO_PIPELINE.md`.
