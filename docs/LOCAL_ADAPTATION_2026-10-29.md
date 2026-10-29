# Huấn luyện với speech200 và AIRTLab100 — 29/10/2026

Huy yêu cầu train 200 clip cho nhánh Toxic và 100 clip cho Violence. Run mới: `local-adaptation-2026-10-29`, độc lập với bản hybrid trước đó. Giữ checkpoint app hiện tại trong khi chạy/đánh giá ứng viên.

**Đã hoàn tất:**5 epoch Toxic,4 cấu hình Violence, đánh giá holdout GPU/CPU, lưu checkpoint và CSV/đồ thị. [Kết quả](../reports/LOCAL_ADAPTATION_2026-10-29.md). Không nâng app: Toxic CPU hybrid FP6→8 dù FN6→2; Violence RLVS FN5→14. Không chạy lại lệnh train/evaluate vào run này để sửa kết quả.

## Nhãn và chia dữ liệu

- Speech200 thực tế là **100 Toxic + 100 bình thường** theo mô tả người cung cấp. Dùng nhãn đó như **nhãn yếu cấp clip**, không ghi thành nhãn người nghe duyệt từng đoạn. Không tự điền phiếu `human_label` hoặc tạo transcript chuẩn.
- Gom cùng video nguồn, SHA trùng và thành phần liên thông của các cặp dHash nghi gần trùng. Khóa **142 train (71/71), 28 validation (14/14), 30 test (15/15)** trước ASR/huấn luyện. 11 clip liên quan ba ca lỗi đã xem bắt buộc vào train. Chọn split bằng metadata/nhãn để cân bằng, không dùng điểm model.
- Nguồn vẫn có thể là reupload khác ID; cùng người nói có thể ở nhiều split. Huy đã thử các video khác chưa được liệt kê. Vì vậy chỉ gọi test là holdout theo nhóm nguồn tạm, **không phải test mù/gold độc lập đã rà soát**.
- AIRTLab100 đã xác minh nguồn: **50 bạo lực + 50 không bạo lực**, dùng nhãn tác giả. Theo yêu cầu train mới, cả bối cảnh này chuyển vào train như một nhóm, kết hợp RLVS train1.157. Không chia cùng bối cảnh vào train/test. Validation/test RLVS giữ 386/386 như cũ.
- AIRTLab không còn là test ngoài cho ứng viên mới. Báo cáo external ngày28/10 vẫn đúng cho model cũ ở thời điểm đó; mọi kết quả của model mới trên AIRTLab phải ghi **training only**. Không sửa manifest/báo cáo lịch sử để che việc đổi vai trò dữ liệu.

## Phương pháp

**Toxic:** Whisper small beam5/VAD trên GPU FP16, giữ kiểm tiếng Việt và segment không rỗng. ASR thô không phải transcript chuẩn. Transcript GPU có thể khác CPU INT8 đang dùng trong app; kết quả lượt này chưa đại diện pipeline CPU triển khai.

PhoBERT khởi tạo từ v3, mở học hai encoder block cuối và classifier, 5 epoch, seed42, encoder lr5e-6/classifier lr2e-5, weight decay0,01, tích lũy8 clip, max192 token/segment. Mỗi clip là một túi segment: lấy max logit Toxic giữa các segment rồi tính BCE với nhãn clip. Cách này giữ max-score tại inference và không gán tất cả câu trong video Toxic thành Toxic. Với nhãn yếu, model vẫn có thể học nhầm một câu; cần nhãn thật để kiểm chứng.

Clip không có ASR đủ điều kiện bị loại khỏi fitting/loss xác suất và báo coverage riêng; không đặt score null thành0. Chọn epoch/ngưỡng chỉ bằng validation, ưu tiên Precision/Recall không giảm so với v3 nếu có ứng viên đủ; sau đó macro-F1/F1/loss. Lưu optimizer, scaler, RNG và checkpoint cuối mỗi epoch để resume. Bộ từ khóa hybrid giữ nguyên, không dùng để tạo nhãn.

**Violence:** R3D-18 pretrained đóng băng, 5×16 frame/15fps như v3. Huấn luyện StandardScaler + Logistic Regression balanced trên **1.257 clip** (1.157 RLVS +100 AIRTLab); C∈{0,001;0,01;0,1;1}. Chọn C/ngưỡng bằng validation RLVS; sau đóng băng mới đánh giá test RLVS cũ. Không gọi đây là fine-tune backbone R3D.

## Chạy lại hoặc tiếp tục

Môi trường app `.venv` CPU giữ nguyên; `.venv-train` dùng CUDA và phụ thuộc ghim trong `requirements-training.lock.txt`. GPU ASR thêm faster-whisper/ctranslate2/onnxruntime; Windows dùng DLL CUDA/cuDNN có sẵn trong torch/lib của môi trường train, không đổi PATH hệ thống.

```powershell
# Chỉ cho run mới; prepare từ chối ghi đè manifest/protocol:
.\.venv\Scripts\python.exe -m ml.local_adaptation prepare --run-id local-adaptation-2026-10-29
# Resume từ các cache/epoch đã hoàn tất:
.\.venv-train\Scripts\python.exe -m ml.local_adaptation asr --run-id local-adaptation-2026-10-29
.\.venv-train\Scripts\python.exe -m ml.speech_mil train --run-id local-adaptation-2026-10-29
.\.venv-train\Scripts\python.exe -m ml.local_adaptation asr-test --run-id local-adaptation-2026-10-29
.\.venv-train\Scripts\python.exe -m ml.speech_mil evaluate --run-id local-adaptation-2026-10-29
.\.venv-train\Scripts\python.exe -m ml.local_violence select --run-id local-adaptation-2026-10-29
.\.venv-train\Scripts\python.exe -m ml.local_violence evaluate --run-id local-adaptation-2026-10-29
# Kiểm CPU sau selection; không dùng kết quả để chỉnh ngưỡng:
.\.venv\Scripts\python.exe -m ml.speech_cpu_check --run-id local-adaptation-2026-10-29
.\.venv\Scripts\python.exe -m ml.adaptation_audit --run-id local-adaptation-2026-10-29
.\.venv\Scripts\python.exe -m ml.adaptation_report --run-id local-adaptation-2026-10-29
```

Không chạy đồng thời hai lệnh vào cùng run/task. `train/select` từ chối sau selection; `evaluate` từ chối sau evaluation. Khi muốn thí nghiệm mới phải dùng run-id mới, không xóa kết quả để tìm model bằng test.

Manifest/ASR/checkpoint/predictions từng clip giữ local trong `runs/local-adaptation-2026-10-29`. Protocol và chỉ số tổng hợp được lưu ở `reports/local-adaptation-2026-10-29`. App chỉ được đổi model sau khi đánh giá phù hợp và kiểm inference CPU; lượt này tạo ứng viên riêng theo yêu cầu train.
