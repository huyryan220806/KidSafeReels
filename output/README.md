# Báo cáo và slide

Bản ngày 18/10/2026 gồm PDF kỹ thuật 9 trang và PPTX 11 slide có speaker notes, bảng/biểu đồ chỉnh sửa được. Nội dung bám theo `reports/TECHNICAL_REPORT.md` và JSON thực nghiệm. Chưa có metric pilot, nghiệm thu máy thứ hai hoặc thông tin đầy đủ sinh viên; bổ sung trước bản nộp cuối.

- `pdf/KidSafeReels_BaoCaoKyThuat.pdf`: bản đọc/in.
- `presentation/KidSafeReels_ThuyetTrinh.pptx`: bản thuyết trình chỉnh sửa được.
- `docs/DEFENSE_NOTES.md`: kịch bản demo và trả lời câu hỏi (đường dẫn từ gốc repo).

Nguồn PDF ở `scripts/build_report.py`, cần Python có ReportLab và Arial tại Windows Fonts. Sửa nội dung Markdown trước khi render lại. Các thư viện tạo tài liệu không phải phụ thuộc app.

Nguồn slide ở `scripts/build_slides.mjs`, dùng runtime `@oai/artifact-tool` của Codex. Khi tạo lại: copy vào `runs/presentation-build/`, liên kết `node_modules` ở đó tới gói runtime; đặt `RUNTIME_NODE_MODULES`, `KIDSAFE_PRESENTATION_SKILL`, `KIDSAFE_ARTIFACT_PYTHON` theo đường dẫn runtime trên máy. Chạy Node của runtime với CWD là gốc repo. Finalizer yêu cầu tên output mới nếu đã tồn tại. Sau thay đổi nội dung cần render và xem lại từng trang/slide.

Bản hiện tại đã render và kiểm tra hình ảnh từng trang/slide bằng công cụ tạo tài liệu. Chưa mở nghiệm thu trong PowerPoint hoặc thử trên máy thứ hai.
