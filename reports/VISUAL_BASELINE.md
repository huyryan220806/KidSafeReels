# Baseline Violence — 16/10/2026

Đã huấn luyện và đánh giá thực trên CPU. Nguồn số liệu: [visual_baseline_metrics.json](visual_baseline_metrics.json); danh sách mẫu/split: [rlvs-v1.json](../data/splits/rlvs-v1.json). Đây là kết quả thăm dò trên RLVS, chưa phải độ chính xác của hệ thống bảo vệ trẻ em trên video mạng xã hội.

## Dữ liệu và kiểm tra chất lượng

[RLVS do tác giả cung cấp trên Kaggle](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset), version 1. Archive tải thực tế 3.847.870.446 byte (3,58 GiB), SHA-256 trong report JSON. Archive chứa hai cây thư mục, tổng 3.902 MP4, khác số lượng mô tả dataset. Không coi mỗi đường dẫn là một video độc lập.

Kiểm tra SHA-256 tìm được 1.937 nội dung file duy nhất. Trong đó `NV_226.mp4` và `V_504.mp4` có cùng nội dung nhưng nhãn trái nhau; loại cả 4 đường dẫn của nội dung này. Lọc thêm 3 video dài hơn 60 giây và 4 video không giải mã đủ frame lấy mẫu. Tổng 1.973 đường dẫn bị loại: 1.962 bản trùng chính xác, 4 bản có nhãn mâu thuẫn, 7 lỗi thời lượng/giải mã. **Giữ 1.929 video.** Xử lý nhãn mâu thuẫn hoàn tất trước khi fit và xem test.

Lấy ba dHash 64 bit từ các frame đầu/giữa/cuối của bộ lấy mẫu để gom clip tương tự (cả ba khoảng cách Hamming ≤10), rồi chia nhóm bằng StratifiedGroupKFold với seed 42. Có 1.832 nhóm; không nhóm nào xuất hiện ở nhiều split. Feature phục vụ huấn luyện chỉ được trích sau khi chốt split; frame dùng để băm là bước kiểm tra trùng lặp.

RLVS không cung cấp đủ ID video nguồn để xác nhận mọi clip cùng nguồn. dHash là heuristic, có thể bỏ sót bản cắt/reupload hoặc gom nhầm cảnh giống nhau. Không tuyên bố đã loại hết rò rỉ nguồn.

| Tập | Video | NonViolence | Violence | Nhóm |
|---|---:|---:|---:|---:|
| Train | 1.157 | 564 | 593 | 1.124 |
| Validation | 386 | 188 | 198 | 375 |
| Test | 386 | 188 | 198 | 333 |

## Phương pháp

ResNet18 tiền huấn luyện ImageNet, đóng băng toàn bộ backbone. Tối đa 8 frame lấy đều theo chỉ số frame, resize/crop/normalize theo transform của Torchvision IMAGENET1K_V1. Gom mean và max embedding thành 1.024 chiều, StandardScaler và Logistic Regression C=0,1, class_weight=balanced, solver=liblinear, seed 42. Chỉ fit scaler/classifier trên train.

Một cấu hình baseline cố định; chọn ngưỡng theo macro-F1 validation trên lưới 0,05–0,95: **0,43**. Không refit trên validation và không điều chỉnh sau khi xem test. Ngưỡng trong JSON có biểu diễn float `0.42999999999999994`.

## Kết quả

| Chỉ số | Validation | Test |
|---|---:|---:|
| Precision Violence | 0,9458 | 0,9550 |
| Recall Violence | 0,9697 | 0,9646 |
| F1 Violence | 0,9576 | 0,9598 |
| Macro-F1 | 0,9559 | 0,9585 |
| Accuracy | 0,9560 | 0,9585 |

Confusion test (hàng = nhãn thật, cột = dự đoán):

| | Dự đoán NonViolence | Dự đoán Violence |
|---|---:|---:|
| Thật NonViolence | 179 | 9 |
| Thật Violence | 7 | 191 |

Bỏ sót 7/198 ca Violence và cảnh báo nhầm 9/188 ca NonViolence. Không có nhãn thời gian để đánh giá khả năng định vị cảnh.

Fit classifier mất khoảng 0,33 giây sau khi có feature. Vòng đọc cache/trích feature của lượt cuối mất 654,28 giây; **lượt này đã dùng lại một phần cache**, không phải thời gian toàn bộ từ tải dữ liệu đến model, cũng không phải benchmark trích feature lần đầu. Chưa đo peak RAM. CPU AMD Ryzen 5 7535HS, 4 torch threads, Windows/Python 3.12.10, torch 2.14.1+cpu, torchvision 0.29.1+cpu, scikit-learn 1.9.1. Không dùng RTX 4060 có sẵn.

## Giới hạn và bước tiếp

Mô hình có thể học bối cảnh/camera của RLVS, chưa mô hình hóa chuyển động và có thể bỏ sót bạo lực thoáng qua. Evidence visual là điểm toàn clip, không phải timestamp cảnh bạo lực. Score chưa calibration. Kết quả cao trên benchmark này không thay thế đánh giá video dọc tiếng Việt có nhạc, chữ, chuyển cảnh và nội dung nhiều nhãn.

Giữ baseline và ngưỡng đã chọn để tích hợp app. Tiếp theo cần tập video đích độc lập có nhãn, phân tích lỗi theo bối cảnh, đánh giá cùng nhánh Toxic/ASR. Không thử nhiều cấu hình dựa trên 386 video test đã xem rồi báo đó là kết quả giữ riêng.

RLVS ghi quyền dữ liệu “Data files © Original Authors”; repo chỉ chứa mã, hash/split và kết quả, không tái phân phối video. Trích dẫn nguồn: Soliman et al., *Violence Recognition from Videos using Deep Learning Techniques*, ICICIS 2019.
