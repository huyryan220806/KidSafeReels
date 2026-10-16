import hashlib
import json
import math
import os
from time import perf_counter

import requests
import streamlit as st

from apps.validation import validate_result
from apps.web.client import DECISIONS, LABELS, MODALITIES, analyze_upload, check_health, error_message, timestamp

st.set_page_config(page_title="KidSafeReels · Kiểm tra video", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1250px; padding-top: 2.5rem;}
[data-testid="stSidebar"] {background: #edf4f1;}
h1 {letter-spacing: -0.04em;}
.eyebrow {color: #34745b; font-size: .8rem; font-weight: 700; letter-spacing: .16em;}
[data-testid="stMetricValue"] {font-size: 1.45rem;}
</style>
""", unsafe_allow_html=True)


def clear_result():
    for key in ("analysis_result", "elapsed_seconds", "evidence_index", "analysis_error"):
        st.session_state.pop(key, None)


@st.cache_data(ttl=10, show_spinner=False)
def health_status(url):
    return check_health(url)


api_url = os.getenv("KIDSAFE_API_URL", "http://127.0.0.1:8000").rstrip("/")
try:
    http_timeout = float(os.getenv("KIDSAFE_HTTP_TIMEOUT", "225"))
    if not math.isfinite(http_timeout) or http_timeout <= 0:
        raise ValueError
except ValueError:
    st.error("Cấu hình KIDSAFE_HTTP_TIMEOUT phải là số giây dương.")
    st.stop()

with st.sidebar:
    st.markdown("### 🛡️ KidSafeReels")
    st.caption("Cùng người lớn chọn nội dung phù hợp cho trẻ.")
    st.divider()
    if st.button("Kiểm tra kết nối", use_container_width=True):
        health_status.clear()
    health = health_status(api_url)
    if health is None:
        st.error("Chưa kết nối được API")
        st.caption("Khởi động app bằng scripts/run.ps1 rồi kiểm tra lại.")
    else:
        st.success("API đang hoạt động")
        if health.get("backend") == "mock":
            st.warning("CHẾ ĐỘ MÔ PHỎNG")
            st.caption("Chỉ kiểm tra luồng ứng dụng. Chưa phân tích nội dung video.")
        elif health.get("backend") == "model":
            st.info("Đã chọn chế độ model")
            st.caption("Trạng thái API chưa xác nhận model đã nạp thành công.")
        else:
            st.error("Cấu hình backend không hợp lệ")
    st.divider()
    st.markdown("**Giới hạn video**")
    st.caption("MP4 / MOV / WebM · tối đa 50 MiB · 60 giây · tối đa 8,3 triệu điểm ảnh mỗi khung.")
    st.markdown("**Dữ liệu của bạn**")
    st.caption("API xóa file tạm sau mỗi lượt xử lý. Web giữ video và kết quả trong phiên đang mở; chọn video khác để xóa kết quả cũ.")
    st.caption("Công cụ hỗ trợ sàng lọc; người lớn vẫn cần xem xét ngữ cảnh.")

st.markdown('<div class="eyebrow">XEM TRƯỚC · HIỂU RÕ · CÂN NHẮC</div>', unsafe_allow_html=True)
st.title("Một bước kiểm tra trước khi trẻ xem")
st.write("Tải video ngắn lên để xem các dấu hiệu rủi ro và kiểm tra lại từng đoạn được chỉ ra.")
st.caption("01 Chọn video   →   02 Phân tích   →   03 Xem kết quả và bằng chứng")
st.divider()

left, right = st.columns([1, 1.2], gap="large")
with left:
    st.subheader("Video của bạn")
    upload = st.file_uploader(
        "Chọn video", type=["mp4", "mov", "webm"],
        on_change=clear_result, help="Video tối đa 60 giây và 50 MiB. API kiểm tra file thật trước khi phân tích.",
    )
    identity = None
    if upload is not None:
        identity = (upload.name, upload.size, hashlib.sha256(upload.getvalue()).hexdigest())
    if st.session_state.get("upload_identity") != identity:
        clear_result()
        st.session_state["upload_identity"] = identity
    if upload is not None:
        st.caption(f"{upload.name} · {upload.size / (1024 * 1024):.2f} MiB")
    preview = st.empty()
    if upload is None:
        preview.info("Chọn một video để bắt đầu. Kết quả sẽ xuất hiện ở bên phải.")
    else:
        st.caption("Nếu trình duyệt không phát được MOV/codec này, hãy xuất lại MP4/H.264 để xem trước.")
    submitted = st.button("Phân tích video", type="primary", disabled=upload is None, use_container_width=True)
    if submitted:
        clear_result()
        if upload.size == 0:
            st.session_state["analysis_error"] = "File video rỗng. Hãy chọn file khác."
        elif upload.size > 50 * 1024 * 1024:
            st.session_state["analysis_error"] = "Video vượt giới hạn 50 MiB."
        else:
            started = perf_counter()
            with st.status("Đang kiểm tra và phân tích video…", expanded=True) as status:
                st.write("Đang gửi video. API sẽ kiểm tra file, giải mã và gọi pipeline đã cấu hình.")
                try:
                    result = analyze_upload(api_url, upload, http_timeout)
                    st.session_state["analysis_result"] = result
                    st.session_state["elapsed_seconds"] = perf_counter() - started
                    status.update(label="Đã nhận kết quả", state="complete", expanded=False)
                except requests.Timeout:
                    st.session_state["analysis_error"] = "Hết thời gian chờ API. Hãy đợi lượt xử lý kết thúc rồi thử video ngắn hơn."
                except requests.ConnectionError:
                    st.session_state["analysis_error"] = "Chưa kết nối được API. Hãy chạy scripts/run.ps1 và kiểm tra kết nối."
                except requests.HTTPError as exc:
                    st.session_state["analysis_error"] = error_message(exc.response)
                except requests.RequestException:
                    st.session_state["analysis_error"] = "Không gửi được yêu cầu. Kiểm tra địa chỉ và kết nối API."
                except (ValueError, KeyError, TypeError):
                    st.session_state["analysis_error"] = "API trả dữ liệu không đúng hợp đồng. Kiểm tra phiên bản API."
                if "analysis_error" in st.session_state:
                    status.update(label="Chưa thể hoàn tất phân tích", state="error", expanded=False)

start_time = 0
with right:
    st.subheader("Kết quả sàng lọc")
    if "analysis_error" in st.session_state:
        st.error(st.session_state["analysis_error"])
    if "analysis_result" not in st.session_state:
        st.info("Chưa có kết quả phân tích cho video này.")
        st.markdown("**Bạn sẽ xem được**")
        st.write("Kết luận trong phạm vi model, điểm từng nhóm rủi ro, cảnh báo và các đoạn cần kiểm tra lại.")
        st.caption("Nhóm chưa được model đánh giá sẽ hiện “Chưa đánh giá”, không quy thành điểm 0.")
    else:
        try:
            result = validate_result(st.session_state["analysis_result"]).model_dump(mode="json")
        except (ValueError, TypeError):
            clear_result()
            st.error("API trả dữ liệu không đúng hợp đồng. Kiểm tra phiên bản API.")
        else:
            if result["mode"] == "mock":
                st.warning("KẾT QUẢ MÔ PHỎNG — chưa đánh giá nội dung video.")
            else:
                st.caption("KẾT QUẢ TỪ MODEL")
            decision = result["decision"]
            st.subheader(DECISIONS[decision])
            if decision == "flagged":
                st.error("Nên xem lại các đoạn được chỉ ra trước khi chia sẻ với trẻ.")
            elif decision == "no_risk_detected":
                st.success("Chưa ghi nhận rủi ro ở các nhóm đã đánh giá; kết quả không bảo đảm an toàn tuyệt đối.")
            else:
                st.info("Cần người lớn xem lại video và các giới hạn bên dưới.")
            st.table([
                {"Nhóm nội dung": LABELS[label], "Điểm model": "Chưa đánh giá" if score is None else f"{score:.3f}"}
                for label, score in result["scores"].items()
            ])
            st.caption("Điểm nằm trong khoảng 0–1; không phải xác suất đã hiệu chỉnh. Kết luận do pipeline quyết định theo ngưỡng của model.")
            evaluated = [LABELS[label] for label in result["evaluated_labels"]]
            st.caption("Đã đánh giá: " + (", ".join(evaluated) if evaluated else "Chưa có nhóm nào"))
            for warning in dict.fromkeys(result["warnings"]):
                st.warning(warning)
            st.markdown("#### Các đoạn cần xem lại")
            evidence = result["evidence"]
            if evidence:
                options = list(range(len(evidence)))
                index = st.selectbox(
                    "Chọn mốc để phát video", options, key="evidence_index",
                    format_func=lambda i: f"{timestamp(evidence[i]['start_seconds'])}–{timestamp(evidence[i]['end_seconds'])} · {LABELS[evidence[i]['label']]}",
                )
                item = evidence[index]
                start_time = math.floor(item["start_seconds"])
                st.write(item["description"])
                st.caption(f"Nguồn: {MODALITIES[item['modality']]} · Điểm: {item['score']:.3f} · Video bắt đầu từ {timestamp(start_time)}")
                with st.expander("Xem toàn bộ bằng chứng"):
                    st.dataframe([
                        {"Từ": timestamp(e["start_seconds"]), "Đến": timestamp(e["end_seconds"]),
                         "Nhóm": LABELS[e["label"]], "Nguồn": MODALITIES[e["modality"]],
                         "Điểm": e["score"], "Mô tả": e["description"]} for e in evidence
                    ], hide_index=True, use_container_width=True)
            else:
                st.caption("Pipeline không cung cấp bằng chứng cho lượt này. Điều này không đồng nghĩa video an toàn.")
            st.divider()
            server, total = st.columns(2)
            server.metric("Xử lý tại API", f"{result['processing_seconds']:.2f} giây")
            elapsed = st.session_state.get("elapsed_seconds")
            total.metric("Toàn lượt yêu cầu", "Chưa đo" if elapsed is None else f"{elapsed:.2f} giây")
            st.caption(f"Phiên bản: {result['model_version']} · Schema: {result['schema_version']}")
            st.download_button(
                "Tải báo cáo JSON", json.dumps(result, ensure_ascii=False, indent=2),
                "kidsafe-result.json", "application/json", use_container_width=True,
            )
            with st.expander("Thông tin kỹ thuật của lượt phân tích"):
                st.code(result["analysis_id"], language=None)
                st.json(result)

if upload is not None and 0 < upload.size <= 50 * 1024 * 1024:
    mime = {"webm": "video/webm", "mov": "video/quicktime"}.get(upload.name.rsplit(".", 1)[-1].lower(), "video/mp4")
    preview.video(upload.getvalue(), format=mime, start_time=start_time)

st.divider()
st.caption("KidSafeReels · Hỗ trợ người lớn sàng lọc nội dung. Luôn kết hợp kết quả với việc xem video và hiểu ngữ cảnh.")
