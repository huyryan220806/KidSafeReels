# Báo cáo và slide

## Bản cập nhật máy app - 18/10/2026

- `pdf/KidSafeReels_BaoCaoKyThuat_Tien.pdf`: 9 trang, có Lê Quyết Tiến / 2474802010386 / lớp 03_02 / giảng viên Nguyễn Thái Anh và kết quả nghiệm thu tự động trên máy i5.
- `presentation/KidSafeReels_ThuyetTrinh_Tien.pptx`: 11 slide, cập nhật thông tin thành viên và số đo máy app; giữ nguyên dữ liệu benchmark, bảng và hai chart/workbook của Huy.
- Các tệp không có hậu tố `_Tien` là bản Huy bàn giao, giữ để đối chiếu. Nhật ký máy app ở `docs/APP_ACCEPTANCE_TIEN.md`.
- Bản cập nhật đã render và xem từng trang/slide; deck qua kiểm tra cấu trúc, layout, font, native table/chart và import. Chưa xác nhận mở trong PowerPoint. Đây không phải video quay demo hoặc bằng chứng nhãn pilot đã hoàn thành.

Tạo lại PDF bằng runtime có ReportLab:

```powershell
python scripts/build_report.py --profile reports/app_profile.json --output output/pdf/KidSafeReels_BaoCaoKyThuat_Tien.pdf
```

Nguồn cập nhật slide: `scripts/update_app_slides.mjs`. Chạy theo cấu hình runtime bên dưới. Helper `restore_source_chart_workbooks.py` giữ nguyên byte chart XML/workbook nguồn sau roundtrip import, không tạo số liệu mới hoặc thay bảng tính gốc. Finalizer yêu cầu output chưa tồn tại khi chạy lại.

## Bản gốc Huy bàn giao

Bản ngày 18/10/2026 gồm PDF kỹ thuật 9 trang và PPTX 11 slide có speaker notes, bảng/biểu đồ chỉnh sửa được. Nội dung bám theo `reports/TECHNICAL_REPORT.md` và JSON thực nghiệm. Chưa có metric pilot, nghiệm thu máy thứ hai hoặc thông tin đầy đủ sinh viên; bổ sung trước bản nộp cuối.

- `pdf/KidSafeReels_BaoCaoKyThuat.pdf`: bản đọc/in.
- `presentation/KidSafeReels_ThuyetTrinh.pptx`: bản thuyết trình chỉnh sửa được.
- `docs/DEFENSE_NOTES.md`: kịch bản demo và trả lời câu hỏi (đường dẫn từ gốc repo).

Nguồn PDF ở `scripts/build_report.py`, cần Python có ReportLab và Arial tại Windows Fonts. Sửa nội dung Markdown trước khi render lại. Các thư viện tạo tài liệu không phải phụ thuộc app.

Nguồn slide ở `scripts/build_slides.mjs`, dùng runtime `@oai/artifact-tool` của Codex. Khi tạo lại: copy vào `runs/presentation-build/`, liên kết `node_modules` ở đó tới gói runtime; đặt `RUNTIME_NODE_MODULES`, `KIDSAFE_PRESENTATION_SKILL`, `KIDSAFE_ARTIFACT_PYTHON` theo đường dẫn runtime trên máy. Chạy Node của runtime với CWD là gốc repo. Finalizer yêu cầu tên output mới nếu đã tồn tại. Sau thay đổi nội dung cần render và xem lại từng trang/slide.

Bản hiện tại đã render và kiểm tra hình ảnh từng trang/slide bằng công cụ tạo tài liệu. Chưa mở nghiệm thu trong PowerPoint hoặc thử trên máy thứ hai.
