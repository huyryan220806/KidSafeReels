# Bàn giao cập nhật Toxic hybrid cho app — 29/10/2026

Nhánh tích hợp: `feat/ml-release`. Đã ghép pipeline và phần giải thích trên web, không cần app viết lại bộ luật. Không có tin nhắn hoặc file nào được tự gửi cho thành viên app.

## Thay đổi cần nhận

- `ml/toxic_rules.py`, `ml/resources/toxic_lexicon_v1.json`, `ml/inference.py`: rules song song PhoBERT, nguồn cảnh báo/mốc ASR và phiên bản lexicon.
- `apps/web/design.py`, `apps/web/app.py`: giải thích điểm model khác quyết định từ luật; báo cáo TXT có cùng thông tin. UI giữ luồng upload/phát đoạn/tải JSON.
- `scripts/run-model.ps1`: mặc định hybrid, tham số `-ToxicMode model_only` cho baseline. Không đổi dependency, tên trường hoặc schema 1.0; [contract đã cập nhật ngữ nghĩa](API_CONTRACT.md).
- Tests, [báo cáo/CSV/đồ thị](../reports/TOXIC_HYBRID_2026-10-29.md), [thiết kế và thao tác](TOXIC_HYBRID.md).

## Cài và chạy

Nhận code mới bằng fast-forward/merge nhánh tích hợp phù hợp checkout đang dùng; giữ thay đổi local của app, không reset hoặc force-push. Máy đã có ZIP CPU v3/17 tệp thì dùng lại. Nếu còn v1/v2, làm theo [cài v3](MODEL_UPDATE_2026-10-28.md) trước; bộ luật trong Git, không cần ZIP trọng số mới cho lượt 29/10.

Dừng API/web đang chạy bằng Ctrl+C trong terminal khởi động, sau đó tại repo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run-model.ps1 -OpenBrowser
```

Mở http://127.0.0.1:8501. Phiên bản kết quả cần có `+asr-native-filter-v2+toxic-hybrid-v1-56b3a177c7b2`. Mở launcher lại trong lúc server cũ sống chỉ tái dùng phiên, không reload model/config. `GET /health` chỉ xác nhận backend model, không xác nhận version đã nạp.

`flagged` với điểm Toxic thấp là hợp lệ nếu evidence nói rõ nguồn luật; app không tự suy lại nhãn từ điểm hoặc thay điểm lên cao. Hai nhãn chưa hỗ trợ vẫn null. Review do thiếu audio/không có bằng chứng đủ rõ không đồng nghĩa an toàn.

## Kiểm chứng đã làm và phần còn lại

Máy Huy: 238 tests, 7/7 HTTP/model thật, 17 checksum; trình duyệt thật đã kiểm clip022, nguồn cảnh báo, mốc phát, điểm gốc, TXT/JSON. Chưa nghiệm thu thay cho máy Tiến.

Máy app cần kiểm chạy model, upload một clip đã được phép dùng, đọc evidence/score/version, tải TXT/JSON và đổi file để xóa kết quả cũ. Không chia sẻ raw clip riêng tư qua Git. Nhãn speech200 của hai người, test video đích, cải thiện ca019 và demo/báo cáo nộp cuối vẫn chưa hoàn tất. Hybrid chưa tăng F1 ViCTSD so với PhoBERT; không trình bày như model mới đã train hoặc đạt độ chính xác cao hơn trên video thực tế.
