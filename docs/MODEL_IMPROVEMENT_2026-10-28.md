# Đợt cải tiến model — 28/10/2026

Huy yêu cầu tiếp tục cải thiện chất lượng. Bundle CPU v2 đang dùng là mốc đối chiếu; không ghi đè trước kiểm thử. Hai classifier tuyến tính thử ngày 20/10 không cải thiện test, nên lần này thay đổi biểu diễn thay vì lặp lại tìm ngưỡng.

## Hướng thực hiện

1. Toxic: fine-tune PhoBERT-base trên train ViCTSD, phân đoạn từ tiếng Việt bằng PyVi nhất quán train/inference. Theo dõi loss/accuracy/F1 từng epoch thật; chọn checkpoint/ngưỡng trên validation. ASR small giữ nguyên để tách ảnh hưởng classifier.
2. Violence: dùng ResNet3D-18 pretrained Kinetics400 để trích đặc trưng chuỗi frame, so với và kết hợp đặc trưng ResNet18 cũ. Chọn classifier/ngưỡng trên validation RLVS; giữ nhóm/split cũ.
3. Khóa protocol trước train, checkpoint/seed/cache để tiếp tục khi gián đoạn. Test benchmark đã xem trong các đợt trước chỉ là so sánh hồi quy, không gọi là test đích mới. Không dò model/ngưỡng bằng test.
4. Nếu ứng viên tốt hơn đủ theo tiêu chí đã chốt, đo CPU inference và tích hợp app với bundle mới có rollback; chỉ dùng GPU để huấn luyện/trích feature. Nếu không đạt, công bố kết quả và giữ bundle v2.

## Tài nguyên và giới hạn

Máy Huy có RTX 4060 Laptop 8 GiB, khoảng 5,6 GiB GPU còn trống lúc kiểm tra. Dùng môi trường `.venv-train` riêng để giữ môi trường app CPU. Dữ liệu hiện có: ViCTSD 6.965/1.975/988; RLVS 1.157/386/386. Nhãn pilot và video lỗi thực tế chưa được xác nhận mới; không tự gán nhãn.

Nguồn: [PhoBERT/VinAI](https://huggingface.co/vinai/phobert-base) (MIT), [PyVi](https://github.com/trungtv/pyvi) (MIT), [R3D-18/Torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html), [PyTorch CUDA packages](https://pytorch.org/get-started/previous-versions/). Ghim revision/hash thật vào protocol khi tải. PyVi khác bộ phân đoạn RDR dùng trong pretraining PhoBERT; ghi đây là lựa chọn triển khai và đánh giá bằng validation.

## Tái tạo thực nghiệm

Môi trường CPU của app là `.venv` hoặc `.venv-model`; môi trường GPU riêng là `.venv-train`. Python 3.12, Torch 2.10.0/torchvision 0.25.0 CUDA 12.8 cho train; Torch 2.14.1/torchvision 0.29.1 CPU cho app. Khóa phiên bản tương ứng trong `requirements-training.lock.txt` và `requirements-release.lock.txt`. Không cài bản CUDA vào môi trường app.

```powershell
python -m venv .venv-train
.\.venv-train\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv-train\Scripts\python.exe -m pip install -r requirements-training.txt -c requirements-training.lock.txt
.\.venv-train\Scripts\python.exe -m ml.prepare_phobert
```

Cần có ViCTSD/RLVS và split/cache baseline theo `MODEL_EXPERIMENTS.md`. Với lần tái tạo mới, chọn một run-id mới; không ghi đè lượt đã công bố. Các lệnh dưới đây dùng run-id của lượt thực tế để chỉ vị trí resume:

```powershell
.\.venv-train\Scripts\python.exe -m ml.deep_toxic train --run-id model-improvement-2026-10-28
.\.venv-train\Scripts\python.exe -m ml.temporal_violence prepare --run-id model-improvement-2026-10-28
.\.venv-train\Scripts\python.exe -m ml.temporal_violence select --run-id model-improvement-2026-10-28
# Chỉ sau selection.json; mỗi evaluate từ chối chạy lại nếu đã có evaluation.json.
.\.venv-train\Scripts\python.exe -m ml.deep_toxic evaluate --run-id model-improvement-2026-10-28
.\.venv-train\Scripts\python.exe -m ml.temporal_violence evaluate --run-id model-improvement-2026-10-28
```

PhoBERT: batch 8, tích lũy gradient 2, learning rate 2e-5, AdamW decay 0,01, warmup 10%, tối đa 192 token, seed 42, 6 epoch, AMP FP16. Cross-entropy lúc tối ưu có trọng số lớp từ train. Checkpoint model/optimizer/scheduler/scaler/RNG lưu sau mỗi epoch; tiếp tục từ epoch hoàn tất, không chạy hai tiến trình train cùng run-id. Đường loss đánh giá là cross-entropy **không trọng số** để so sánh; không trộn với optimizer weighted loss. Đồ thị epoch dùng ngưỡng cố định 0,5, còn metric cuối dùng ngưỡng đã chọn trên validation.

R3D-18: backbone Kinetics400 đóng băng, không fine-tune backbone. Mỗi video 5 cửa sổ, mỗi cửa sổ 16 frame ở 15 fps; resize 171×128, crop 112×112, normalize Kinetics bằng cùng code train/inference. OpenCV INTER_LINEAR không bảo đảm bit-identical với resize của Torchvision. Mean/max 5 embedding tạo vector 1.024 chiều; StandardScaler + LR được fit train. Sáu cấu hình: temporal hoặc ghép static-temporal × C={0,01;0,1;1}. Chọn trên validation theo recall, giới hạn false positive và macro-F1 đã ghi trong protocol.

## AIRTLab từ thư mục Drive

100 video khớp Git blob của kho tác giả tại commit `1f7747e104301ccaa82ef5a2f6804b51ced1c398`, gồm 50 violent/50 non-violent, camera 1. Kiểm SHA-256 và dHash không phát hiện trùng RLVS, nhưng heuristic không chứng minh độc lập nguồn tuyệt đối. Toàn bộ bối cảnh này được giữ ngoài train/validation. Nhãn nguồn lấy từ thư mục tác giả; nhãn do nhóm dự án xác nhận vẫn **null**, không gộp hai loại nhãn.

Metadata và video nằm local trong `data/private/violence-drive-2026-10-28`. `ml.airtlab_external prepare` cần manifest/inventory đã tải từ Drive, đối chiếu Git tree chính thức, có fallback tải đúng blob tác giả khi Drive lỗi. `ml.airtlab_external evaluate` chỉ chạy khi Violence đã khóa; không chỉnh ngưỡng theo kết quả này. Không commit video, tên file cá nhân hoặc Drive ID.

Nguồn: [kho AIRTLab](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos), [bài mô tả](https://pmc.ncbi.nlm.nih.gov/articles/PMC7725718/). Bộ này là hành động dàn dựng trong một bối cảnh; không đại diện Shorts tiếng Việt. README nguồn cho nghiên cứu/giáo dục và yêu cầu trích dẫn; không tự diễn giải thành giấy phép tái phân phối không hạn chế.

## Quyết định phát hành

`reports/model-improvement-2026-10-28/release_gate.json` được ghi trước test: Toxic phải cải thiện macro-F1 và không giảm F1/precision/recall; Violence phải cải thiện macro-F1, không giảm F1/recall RLVS và không giảm macro-F1/recall AIRTLab. Đây là điều kiện nâng bản thử nghiệm tương đối với baseline, không phải chứng nhận chất lượng đủ cho triển khai thực tế. Sau đó cần parity/latency CPU, full suite, HTTP/UI và cài ZIP thật. `ml.package_deep` đóng v3, giữ v2 để khôi phục, chỉ activate khi kiểm thử đúng checksum bundle đã qua.

Kết quả và giới hạn được ghi tại [báo cáo 28/10](../reports/MODEL_IMPROVEMENT_2026-10-28.md). Nhãn pilot và ca lỗi tiếng Việt thực tế vẫn cần người xác nhận; không suy accuracy toàn app từ các benchmark riêng.
