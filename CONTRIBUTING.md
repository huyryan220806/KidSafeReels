# Làm việc với hai thành viên

Huy kiểm tra thành viên app có quyền Write trong GitHub → Settings → Collaborators. Clone repo public không cần quyền Write; push nhánh riêng cần quyền. Việc mời collaborator chưa thực hiện vì chưa có tài khoản GitHub của thành viên.

Mỗi người clone trên máy mình, mở thư mục repo trong Codex và đọc `AGENTS.md`. Dùng nhánh riêng:

```powershell
git switch main
git pull --ff-only origin main
git switch -c feat/app-upload
# ML lead dùng nhánh ví dụ: feat/ml-baseline
# Sau khi sửa và chạy test:
git add <cac-file-da-kiem-tra>
git commit -m "feat: describe the working increment"
git push -u origin HEAD
```

Tạo Pull Request về `main` trên GitHub. Một PR = một đầu ra nhỏ; ghi cách chạy và bằng chứng kiểm tra. Huy review và merge; khi đổi API, cả hai đọc phần thay đổi. Không cùng sửa một file khi chưa thống nhất. Không commit toàn bộ thư mục một cách mù quáng; xem `git diff --cached` trước.

Sau khi PR được merge, quay về main, pull rồi tạo nhánh mới. Nếu nhánh đang làm cần cập nhật, commit công việc hiện tại, chạy `git fetch origin` và `git merge origin/main`; giải quyết xung đột theo hợp đồng API, không chọn toàn bộ phía mình.

Cuối mỗi ngày dành 10–15 phút ghép kết quả: commit/PR nào, chạy được gì, đang vướng gì, đầu ra ngày mai. Thành viên app có thể làm ngay với mock; ML lead bàn giao qua `kidsafe.pipeline.analyze_video(Path)`, không yêu cầu bạn app hiểu notebook.

Tệp phụ thuộc hiện là khoảng phiên bản để khởi tạo. Trước buổi demo, ghi phiên bản thực đã chạy vào môi trường khóa riêng, kiểm tra cài mới trên máy thành viên còn lại. Không ép thành viên app cài torch/ASR khi chỉ làm mock.
