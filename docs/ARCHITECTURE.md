# Kiến trúc và quyết định kỹ thuật

Pipeline v1 đã có web/API, visual Violence và ASR → Toxic trên CPU. OCR/PhoBERT vẫn là mở rộng, chưa triển khai. Hai baseline được học trên hai benchmark riêng; chưa có đánh giá đa nhãn trên tập video đích chung.

```mermaid
flowchart TD
    U[Người lớn upload video] --> W[Streamlit]
    W --> A[FastAPI]
    A --> P[Kiểm tra media và tiền xử lý]
    P --> F[Tối đa 8 frame lấy đều]
    P --> S[Audio]
    F --> V[ResNet18 đóng băng + classifier học trên dữ liệu nhóm]
    S --> T[ASR tiếng Việt + transcript có timestamp]
    T --> C[TF-IDF baseline / PhoBERT features + classifier]
    F -. mở rộng .-> O[OCR]
    O -.-> C
    V --> M[Gộp kết quả hai nhãn và ngưỡng riêng]
    C --> M
    M --> R[Kết quả + evidence + giới hạn]
    R --> W
```

- **Streamlit + FastAPI:** cùng Python để giảm thời gian tích hợp; app owner gọi API mà không cần nạp torch. Web gọi API từ server, chưa cần bật CORS rộng.
- **Visual:** bắt đầu ResNet18 tiền huấn luyện làm bộ trích đặc trưng, rồi học classifier từ nhãn của dự án. Trọng số ImageNet không tự nhận biết “an toàn cho trẻ”. Một frame riêng lẻ cũng không đủ suy luận mọi hành động nguy hiểm.
- **Speech/text:** faster-whisper `base` đa ngôn ngữ chạy CPU int8; dùng transcript tiếng Việt có timestamp qua TF-IDF + logistic regression đã học trên ViCTSD. Chưa đo WER trên tập lời nói tiếng Việt. PhoBERT là thử nghiệm tiếp theo nếu dữ liệu/thời gian cho phép.
- **Fusion:** visual có thể cung cấp điểm violence, text cung cấp toxic_language. Đây đã là kết quả đa phương thức nhưng không được gọi là “weighted fusion cùng lớp” khi hai nhánh không có nhãn chung. Nếu hai nhánh cùng dự đoán một lớp, chọn trọng số/ngưỡng trên validation; không lấy trung bình các nhãn khác nghĩa.
- **Giới hạn hiện tại:** tối đa 60 giây, lấy đều tối đa 8 frame rồi mean/max pooling; có thể bỏ sót cảnh ngắn, chưa học chuyển động. Visual chỉ có evidence cấp clip; audio có timestamp ASR. Bản mock chỉ kiểm tra phần mở rộng và kích thước, chưa giải mã.
- **Lưu trữ:** API dùng thư mục tạm riêng mỗi request và xóa sau xử lý; chưa có database. Cache dữ liệu thí nghiệm local, không commit. Trước khi deploy công khai cần giới hạn request ở proxy, concurrency và thời gian decode; starter chỉ chạy local.
- **Phần cứng:** CPU cho app và baseline; Colab cho trích đặc trưng theo batch/fine-tune tùy chọn. Cache kết quả ASR/features một lần, không chạy lại trong mỗi notebook thử nghiệm.

Nguồn kỹ thuật đã tham khảo khi khởi tạo (05/10/2026): [Torchvision models](https://docs.pytorch.org/vision/stable/models.html), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [PhoBERT model card](https://huggingface.co/vinai/phobert-base-v2), [FastAPI UploadFile](https://fastapi.tiangolo.com/tutorial/request-files/), [Streamlit upload](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader). Cần kiểm tra giấy phép của đúng checkpoint/dataset được chọn; chưa tải hay lựa chọn bộ dữ liệu thực ở bước khởi tạo.
