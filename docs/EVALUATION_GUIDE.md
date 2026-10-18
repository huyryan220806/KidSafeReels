# Đánh giá sau khi nhận nhãn người thật

Ngày 18/10/2026, người dùng xác nhận **chưa gán nhãn** và yêu cầu hoàn thiện phần khác. Công cụ dưới đây đã có kiểm thử nhưng chưa chạy đánh giá chất lượng pilot. Không điền số liệu vào bảng pilot trước khi có gold.

## 1. Thu phiếu độc lập

Huy gán 25, app owner gán 6 theo `ANNOTATION_GUIDE.md`. Lưu hai JSON ở `data/private/pilot-v1/submissions/`. Không đưa điểm model hoặc ASR cho người gán nhãn. Năm clip train 005-009 đã dùng kiểm thử kỹ thuật API; kết quả dự đoán để riêng trong `data/private/release-smoke/`, không dùng làm gợi ý nhãn. Test pilot chưa chạy classifier.

```powershell
.\.venv\Scripts\python.exe -m ml.pilot_evaluation compare --huy data/private/pilot-v1/submissions/annotations-huy.json --app data/private/pilot-v1/submissions/annotations-app.json --output data/private/pilot-v1/adjudication.json
```

Lệnh từ chối phiếu chưa hoàn tất. Đầu ra có tỷ lệ đồng thuận, Cohen's kappa theo nhãn và các bất đồng/nhãn chưa rõ. Kappa trả null nếu không xác định được vì một lớp duy nhất. Mẫu chéo có chủ đích chỉ 6 clip, không đại diện toàn tập.

## 2. Phân xử và tạo gold

Cùng xem lại từng mục trong `decisions` của `adjudication.json`. Điền `value` = 0/1, hoặc null nếu quyết định loại nhãn đó khỏi metric; điền `reviewer`, `reason`, `reviewed=true`. Giá trị null cần lý do loại trừ. Giữ nguyên hai phiếu nộp gốc. Với mục không bất đồng, dùng nhãn của Huy; báo cáo vẫn nêu chỉ 24% clip được gán chéo.

```powershell
.\.venv\Scripts\python.exe -m ml.pilot_evaluation gold --huy data/private/pilot-v1/submissions/annotations-huy.json --app data/private/pilot-v1/submissions/annotations-app.json --adjudication data/private/pilot-v1/adjudication.json --output data/private/pilot-v1/gold.json
```

Gold lưu hash các phiếu, quyết định phân xử, manifest, nhóm và split. Lệnh không sửa phiếu ban đầu và không ghi đè đầu ra đã có. Phần mềm kiểm tra cấu trúc, không xác thực thay cho việc người thật xem video.

## 3. Chốt cấu hình rồi đánh giá

Bản hiện tại đóng băng Violence 0,43 và Toxic 0,59 từ validation của hai benchmark riêng; chưa tối ưu lại theo pilot. Kiểm tra checksum 8 artifact trước chạy. Không dùng test để chọn model/ngưỡng. Nếu sau này chọn ngưỡng trên pilot validation, cần lưu protocol/version mới cùng lập luận và chưa mở test; phiên bản công cụ hiện tại chỉ đánh giá ngưỡng đã đóng băng.

```powershell
.\.venv\Scripts\python.exe -m ml.pilot_evaluation evaluate --gold data/private/pilot-v1/gold.json --split validation --output data/private/pilot-v1/evaluation-validation.json
.\.venv\Scripts\python.exe -m ml.pilot_evaluation evaluate --gold data/private/pilot-v1/gold.json --split test --output data/private/pilot-v1/evaluation-test.json
```

Commit code/tài liệu trước chạy; công cụ từ chối working tree còn thay đổi và media khác checksum audit đã công bố.

Test có khóa lần chạy: cache theo gold/manifest/artifact/commit/media để tiếp tục khi gián đoạn; không ghi đè lần test đã hoàn tất hoặc chạy lại với protocol khác. Không xóa khóa để thử ngưỡng theo test. Khi cần một nghiên cứu mới, giữ lịch sử và thu tập test mới.

Mỗi nhãn báo support, support dương, số có score, coverage, số abstain, dương tính bị abstain và confusion matrix trên phần được chấm. Score null không thành 0. Recall/F1 lớp dương trả null khi không có dương tính trên phần được chấm; macro hai nhãn chỉ có khi cả hai xác định được. Nhãn chưa rõ bị loại có đếm số lượng.

Ablation dùng cùng những clip có cả hai nhãn rõ và cả hai score, với đích chung “có ít nhất một rủi ro P0”. Visual-only, text-only và fusion so cùng ID và support. Đây là thí nghiệm bỏ một modality, không phải ba mô hình đều dự đoán cùng từng nhãn. Hai nhánh xử lý hai loại rủi ro khác nhau; không tuyên bố learned fusion hoặc cộng hai F1 benchmark để suy ra F1 hệ thống.

Nếu test chỉ có âm tính, công bố giới hạn và số cảnh báo nhầm. Muốn đo recall phát hiện nguy hại cần tập có dương tính và quyền sử dụng phù hợp. Không gán giả nhãn dương vào tin giáo dục để đủ lớp.
