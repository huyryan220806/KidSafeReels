# Model card — KidSafeReels CPU v3

**Ứng viên train bổ sung29/10 đã lưu riêng:** PhoBERT MIL từ speech200 nhãn yếu và Violence LR thêm100AIRTLab. Không thay phiên bản phục vụ này: Toxic CPU hybrid tăng báo nhầm, Violence giảm RecallRLVS. [Báo cáo ứng viên](LOCAL_ADAPTATION_2026-10-29.md). AIRTLab vẫn là đánh giá ngoài lịch sử của v3, nhưng là dữ liệu train đối với ứng viên mới; không trộn hai vai trò.

**Inference cập nhật 29/10:** thêm bộ luật Toxic song song PhoBERT, không train lại; mặc định hybrid, có `model_only` để đối chiếu. [Thiết kế](../docs/TOXIC_HYBRID.md), [ablation](TOXIC_HYBRID_2026-10-29.md). ViCTSD test của hybrid bằng PhoBERT (F1 53,39%), rule-only chỉ bắt 1/109 mẫu Toxic. `vi-speech-022` được cảnh báo thêm trong ba ca phát triển đã xem; chưa có metric toàn pipeline trên test video đích.

Cập nhật 28/10/2026: PhoBERT fine-tune + R3D-18/LR + Whisper small beam5, inference hoàn toàn CPU. [Artifact/checksum](../models/artifacts.json), [báo cáo số đo và hình](MODEL_IMPROVEMENT_2026-10-28.md), [cài/mở/rollback](../docs/MODEL_UPDATE_2026-10-28.md). V1/v2 được giữ để đối chiếu, không trộn nghiệm thu máy Tiến cũ với v3.

## Phạm vi

Hỗ trợ người lớn sàng lọc video ≤60 giây trong demo học phần. Hình ảnh cung cấp điểm Violence, lời nói tiếng Việt cung cấp điểm Toxic. Không có Sexual/Dangerous Acts, OCR, nhận diện tuổi hoặc chặn feed TikTok/Reels/Shorts. Kết quả không phải chứng nhận video an toàn cho trẻ.

## Dữ liệu, huấn luyện và suy luận

- **Toxic:** ViCTSD sau làm sạch 6.965/1.975/988. PhoBERT-base từ VinAI revision `01daacda68afe13d83023d16ec647239e344a1e6`, phân đoạn PyVi trên NFC/casefold/whitespace. Fine-tune toàn bộ 6 epoch, seed 42, 192 token, batch 8 × tích lũy 2, lr 2e-5, AdamW/warmup 10%, weighted CE. Chọn epoch 4/ngưỡng 0,78 bằng validation; dùng safetensors offline trên CPU. PyVi khác RDR dùng trong pretraining.
- **Violence:** RLVS Kaggle v1, 1.929 clip sau dedup/lọc lỗi; train/validation/test 1.157/386/386. Gom SHA/dHash trước split nhưng không có đủ metadata để bảo đảm độc lập nguồn. R3D-18 Kinetics400 đóng băng; 5 cửa sổ × 16 frame ở 15 fps, resize171×128/crop112, normalization Kinetics. Mean/max thành 1.024 chiều, StandardScaler + Logistic Regression balanced C=0,01. Nhóm huấn luyện LR, không fine-tune backbone. Chọn C/biểu diễn/ngưỡng 0,41 trên validation trong 6 ứng viên.
- **AIRTLab ngoài train:** 100 clip cam1, 50/50, được xác minh Git blob nguồn; không phát hiện trùng RLVS qua SHA/dHash. Cả bối cảnh giữ ngoài fitting/threshold. Dùng nhãn tác giả, nhãn dự án vẫn null. Hành động dàn dựng cùng bối cảnh, không phải mẫu Shorts tự nhiên độc lập rộng.
- **ASR:** giữ faster-whisper small, CPU int8/4 luồng, beam5/temperature0/VAD, task=transcribe, không condition previous text. Chỉ dùng tiếng Việt khi language_probability≥0,5, giữ segment không rỗng decoder chấp nhận; từ bản sửa 28/10 không lọc lại riêng `no_speech_prob`. Revision `536b0662742c02347bc0e980a01041f333bce120`. Lượt 20/10 đo 16 câu test FLEURS: WER 19,83%, CER 10,70%, chỉ là mẫu đọc nhỏ với pipeline lọc cũ; không gọi là đánh giá mới cho hành vi sau sửa.
- **Kết hợp:** điểm Toxic lấy max PhoBERT theo segment ASR, Violence cấp clip. Model đạt ngưỡng hoặc bộ luật Toxic đủ điều kiện tạo cảnh báo → flagged; còn lại → review. Rule/hybrid không tạo xác suất mới. Hai modality phục vụ hai nhãn khác nhau, không có trọng số fusion được học. Thiếu audio/ASR không đủ điều kiện → Toxic=null; hai nhãn chưa hỗ trợ luôn null.

## Chất lượng đo được

| Benchmark | Accuracy v2 → v3 | F1 lớp rủi ro v2 → v3 | FP/FN v3 |
|---|---:|---:|---:|
| ViCTSD test n=988 | 86,84% → 89,57% | 44,92% → 53,39% | 53/50 |
| RLVS test n=386 | 95,85% → 97,93% | 95,98% → 97,97% | 3/5 |
| AIRTLab nhãn nguồn n=100 | 49,00% → 53,00% | 62,22% → 64,12% | 39/8 |

ViCTSD/RLVS test đã được báo cáo ở lượt trước: đây là benchmark cũ, không phải test đích mới chưa từng xem. Không chỉnh ngưỡng theo test hoặc AIRTLab. Có Accuracy/Loss/Precision/Recall/F1 theo train/validation/test, confusion matrix, PR/ROC và lịch sử epoch thật. Toxic loss tăng 0,4193→0,4641, có quá khớp; score chưa calibration. Không dùng train accuracy hoặc RLVS 97,93% làm độ chính xác toàn app.

## Giới hạn

- Toxic vẫn bỏ sót 50/109 và cảnh báo nhầm 53/879 trên bình luận test; khác lời nói nhận dạng từ ASR. Đã kiểm ba ca Huy chỉ ra: 048/022 flagged, 019 review; đây là dữ liệu phát triển, không phải test độc lập. Max nhiều segment và phép OR với luật có thể tăng false positive ở cấp video; chưa đo mức ảnh hưởng trên speech200.
- AIRTLab vẫn báo nhầm 39/50 clip bình thường. RLVS dễ hơn không chứng minh khả năng chuyển sang video dọc/camera/bối cảnh khác. Chưa đủ dữ liệu đích để tuyên bố hiệu quả bảo vệ trẻ em.
- R3D học biểu diễn chuyển động pretrained, nhưng chỉ xem5 cửa sổ có thể bỏ sót sự kiện giữa chúng. Evidence visual [0,duration] chỉ là phạm vi clip, không định vị mọi cảnh trong khoảng đó.
- Nhạc nền/phương ngữ/ASR sai, cắt 192 token và lỗi chia segment có thể đổi Toxic. Transcript evidence có thể sai; người dùng cần xem lại.
- Chưa đánh giá công bằng theo nhóm/ngôn ngữ/phương ngữ; chưa có metric pilot/ablation trên hai nhãn cùng video, hoặc latency đại diện.

## Kiểm thử và phát hành

152 tests pass, pip check; 13/13 HTTP model thật, 8 UI AppTest pass + 1 chưa chạy do fixture không có evidence. Cài ZIP vào thư mục riêng: 17 artifact đúng hash. Không phải nghiệm thu trình duyệt thủ công hoặc máy Tiến. [Biên bản](model-improvement-2026-10-28/integration_checks.json).

Đã kiểm CPU với checkpoint GPU: Toxic delta tối đa 2,06e-6 ở 32 bình luận train. R3D có sai khác nhỏ do cuDNN TF32; audit FP32 trên 386 test không đổi nhãn, CPU 20 mẫu gần ngưỡng khớp FP32 delta≤8,23e-7. Giữ kết quả kiểm đầu failed và biên bản xử lý, không đổi ngưỡng. CPU median lượt sau ở mẫu nhỏ: Toxic 0,059 s/bình luận, Violence 2,56 s/clip, chưa gồm ASR/API.

Windows/Python 3.12.10, Ryzen 5 7535HS, inference CPU 4 luồng. Train dùng RTX 4060 Laptop 8 GiB trong môi trường riêng. `requirements-training.lock.txt` ghi CUDA; `requirements-release.lock.txt` ghi CPU. ZIP v3≈1,01GiB không chứa video; giữ v2 để rollback.

## Nguồn và quyền sử dụng

PhoBERT/PyVi có MIT; xem nguồn trong protocol/báo cáo. ViCTSD chưa có LICENSE rõ ràng khi kiểm tra; RLVS ghi Data files © Original Authors. AIRTLab phát hành cho nghiên cứu/giáo dục và yêu cầu trích dẫn. Không suy từ tải được thành quyền tái phân phối raw dataset. FLEURS CC-BY-4.0. Checkpoint/raw media/private predictions không commit; chỉ nạp joblib do nhóm tạo hoặc nguồn tin cậy. Đây là sản phẩm học phần, chưa nghiệm thu triển khai public.
