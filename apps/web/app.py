import hashlib
import math
import os
from pathlib import Path
from time import perf_counter

import requests
import streamlit as st

from apps.validation import validate_result
from apps.web.admin import render_admin
from apps.web.client import LABELS, MODALITIES, analyze_upload, check_health, error_message, timestamp
from apps.web.design import decision_card, empty_result, family_notes, filename_card, header, hero, report_text, score_cards, steps

st.set_page_config(page_title="KidSafeReels · Cùng con xem điều hay", page_icon="🛡️", layout="wide")
st.markdown(f"<style>{Path(__file__).with_name('style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def clear_result():
    for key in ("analysis_result", "elapsed_seconds", "evidence_index", "analysis_error", "error_detail"):
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
    st.error("Ứng dụng chưa được cấu hình đúng. Hãy nhờ người cài đặt kiểm tra thời gian chờ kết nối.")
    st.stop()

health = health_status(api_url)
admin_view = False
if os.getenv("KIDSAFE_ENABLE_ADMIN") == "1":
    view = st.sidebar.radio("Giao diện", ["Dành cho gia đình", "Quản trị kỹ thuật"], key="workspace_view")
    st.sidebar.caption("Chế độ phát triển local, chưa có đăng nhập hoặc phân quyền.")
    admin_view = view == "Quản trị kỹ thuật"
st.markdown(header(health), unsafe_allow_html=True)
st.markdown(hero(), unsafe_allow_html=True)
if health is not None and health.get("backend") == "mock":
    st.info("Bạn đang dùng bản mô phỏng để trải nghiệm giao diện. Video chưa được đánh giá nội dung.", icon="ℹ️")
elif health is None:
    st.error("Chưa kết nối được dịch vụ kiểm tra. Bạn vẫn có thể chọn và xem video; hãy thử kết nối lại trong mục hướng dẫn bên dưới.")

progress = st.empty()
progress.markdown(steps(2 if "analysis_result" in st.session_state else 0), unsafe_allow_html=True)
left, right = st.columns([1, 1.05], gap="medium")
with left, st.container(key="upload_panel"):
    st.markdown('<div class="section-kicker">BẮT ĐẦU TỪ ĐÂY</div><div class="section-heading"><span>▣</span><h2>Chọn video cho con</h2></div>', unsafe_allow_html=True)
    st.caption("Video đã lưu trong điện thoại hoặc máy tính của bạn.")
    upload = st.file_uploader(
        "Chọn video từ máy", type=["mp4", "mov", "webm"], label_visibility="collapsed",
        on_change=clear_result, help="Chọn file MP4, MOV hoặc WebM, tối đa 50 MiB và dài không quá 1 phút.",
    )
    st.markdown('<div class="upload-hint"><span>MP4, MOV, WebM</span><span>·</span><span>Tối đa 50 MiB</span><span>·</span><span>Đến 1 phút</span></div>', unsafe_allow_html=True)
    identity = None
    if upload is not None:
        identity = (upload.name, upload.size, hashlib.sha256(upload.getvalue()).hexdigest())
    if st.session_state.get("upload_identity") != identity:
        clear_result()
        st.session_state["upload_identity"] = identity
    if upload is not None:
        st.markdown(filename_card(upload.name, upload.size), unsafe_allow_html=True)
    preview = st.empty()
    if upload is not None:
        with st.expander("Video không phát được?"):
            st.write("Thử mở video bằng trình phát trên máy. Nếu video mở được ở đó nhưng không xem được ở đây, hãy lưu lại dưới dạng MP4/H.264 rồi chọn lại.")
    submitted = st.button("Kiểm tra video", key="analyze", type="primary", disabled=upload is None, use_container_width=True, icon="▶")
    if upload is None:
        st.caption("Chọn một video ở phía trên để bật nút kiểm tra.")
    st.markdown('<div class="privacy-note"><span>♧</span><span>Video được xử lý tạm thời và xóa khỏi nơi xử lý sau mỗi lượt. Kết quả được giữ trong phiên đang mở của bạn.</span></div>', unsafe_allow_html=True)
    if submitted:
        clear_result()
        if upload.size == 0:
            st.session_state["analysis_error"] = "File video rỗng. Hãy chọn một video khác."
        elif upload.size > 50 * 1024 * 1024:
            st.session_state["analysis_error"] = "Video lớn hơn 50 MiB. Hãy chọn video nhẹ hơn hoặc giảm dung lượng rồi thử lại."
        else:
            progress.markdown(steps(1), unsafe_allow_html=True)
            started = perf_counter()
            with st.status("Đang kiểm tra video của bạn…", expanded=True) as status:
                st.write("Video đang được gửi và xử lý. Bạn vui lòng giữ trang này mở trong lúc chờ.")
                try:
                    result = analyze_upload(api_url, upload, http_timeout)
                    st.session_state["analysis_result"] = result
                    st.session_state["elapsed_seconds"] = perf_counter() - started
                    status.update(label="Đã xong — xem phần kết quả kiểm tra", state="complete", expanded=False)
                except requests.Timeout:
                    st.session_state["analysis_error"] = "Hết thời gian chờ. Hãy chờ một chút rồi thử lại với video ngắn hơn."
                except requests.ConnectionError:
                    st.session_state["analysis_error"] = "Chưa kết nối được dịch vụ kiểm tra. Hãy thử kết nối lại trong mục hướng dẫn bên dưới."
                except requests.HTTPError as exc:
                    detail = error_message(exc.response)
                    friendly = {
                        429: "Đang có video khác được kiểm tra. Bạn hãy chờ một chút rồi bấm lại.",
                        500: "Lượt kiểm tra gặp sự cố. Bạn hãy thử lại; nếu vẫn lỗi, nhờ người cài đặt ứng dụng hỗ trợ.",
                        502: "Chưa đọc được kết quả kiểm tra. Bạn hãy thử lại hoặc nhờ người cài đặt ứng dụng hỗ trợ.",
                        503: "Bộ phân tích chưa sẵn sàng. Bạn hãy thử lại sau hoặc nhờ người cài đặt ứng dụng kiểm tra.",
                        504: "Video xử lý lâu hơn dự kiến. Hãy thử video ngắn hơn hoặc giảm độ phân giải.",
                    }
                    st.session_state["analysis_error"] = friendly.get(exc.response.status_code, detail)
                    if detail != st.session_state["analysis_error"]:
                        st.session_state["error_detail"] = detail
                except requests.RequestException:
                    st.session_state["analysis_error"] = "Không gửi được video. Hãy kiểm tra kết nối rồi thử lại."
                except (ValueError, KeyError, TypeError):
                    st.session_state["analysis_error"] = "Chưa đọc được kết quả kiểm tra. Hãy thử lại hoặc nhờ người cài đặt ứng dụng hỗ trợ."
                    st.session_state["error_detail"] = "Dữ liệu trả về không đúng hợp đồng API."
                if "analysis_error" in st.session_state:
                    status.update(label="Chưa hoàn tất — xem thông báo ở phần kết quả", state="error", expanded=False)

start_time = 0
with right, st.container(key="results_panel"):
    st.markdown('<div class="section-kicker">THÊM GÓC NHÌN CHO BẠN</div><div class="section-heading"><span>✧</span><h2>Kết quả kiểm tra</h2></div>', unsafe_allow_html=True)
    if not admin_view and "analysis_error" in st.session_state:
        st.error(st.session_state["analysis_error"])
    if admin_view:
        render_admin(health, api_url)
    elif "analysis_result" not in st.session_state:
        st.markdown(empty_result(), unsafe_allow_html=True)
    else:
        try:
            result = validate_result(st.session_state["analysis_result"]).model_dump(mode="json")
        except (ValueError, TypeError):
            clear_result()
            st.error("Chưa đọc được kết quả kiểm tra. Hãy kiểm tra lại video hoặc nhờ người cài đặt hỗ trợ.")
        else:
            if result["mode"] == "mock":
                st.warning("KẾT QUẢ MÔ PHỎNG — video chưa được đánh giá nội dung. Các nhóm bên dưới đều chưa có kết quả.", icon="ℹ️")
            else:
                st.caption("KẾT QUẢ TỪ BỘ PHÂN TÍCH NỘI DUNG")
            st.markdown(decision_card(result), unsafe_allow_html=True)
            st.markdown("**Các nhóm nội dung**")
            st.markdown(score_cards(result["scores"]), unsafe_allow_html=True)
            st.caption("“Đã kiểm tra” không có nghĩa là an toàn. Với nhóm “Chưa kiểm tra được”, bạn cần tự xem và cân nhắc.")
            for note in family_notes(result):
                st.caption(note)
            st.markdown("**Xem lại những đoạn cần chú ý**")
            evidence = result["evidence"]
            if evidence:
                index = st.selectbox(
                    "Chọn đoạn để xem lại", list(range(len(evidence))), key="evidence_index",
                    format_func=lambda i: f"{timestamp(evidence[i]['start_seconds'])}–{timestamp(evidence[i]['end_seconds'])} · {LABELS[evidence[i]['label']]}",
                )
                item = evidence[index]
                start_time = math.floor(item["start_seconds"])
                st.write(item["description"])
                st.caption(f"Dựa trên: {MODALITIES[item['modality']]} · Trình phát video bắt đầu từ {timestamp(start_time)}")
                if item["modality"] == "visual":
                    st.caption("Khoảng thời gian trên có thể bao phủ cả video. Hãy đọc mô tả; không phải mọi thời điểm trong khoảng đều có rủi ro.")
            else:
                st.caption("Chưa có đoạn cụ thể được chỉ ra. Bạn vẫn nên xem lại toàn bộ video trước khi cho trẻ xem.")
            elapsed = st.session_state.get("elapsed_seconds")
            if elapsed is not None:
                st.caption(f"Hoàn tất trong {elapsed:.1f} giây")
            st.download_button(
                "Lưu kết quả về máy", report_text(result, upload.name if upload else "Không xác định"),
                "kidsafe-ket-qua.txt", "text/plain; charset=utf-8", key="download_summary", use_container_width=True, icon=":material/download:",
            )

if upload is not None and 0 < upload.size <= 50 * 1024 * 1024:
    mime = {"webm": "video/webm", "mov": "video/quicktime"}.get(upload.name.rsplit(".", 1)[-1].lower(), "video/mp4")
    preview.video(upload.getvalue(), format=mime, start_time=start_time)
progress.markdown(steps(2 if "analysis_result" in st.session_state else 1 if upload is not None else 0), unsafe_allow_html=True)

st.markdown('''<div class="support-grid">
  <div class="support-item"><span class="support-icon">♡</span><div><b>Bạn luôn là người quyết định</b><p>Kết quả giúp bạn tham khảo. Hãy cân nhắc thêm độ tuổi, ngữ cảnh và nhu cầu của trẻ.</p></div></div>
  <div class="support-item"><span class="support-icon">◷</span><div><b>Video ngắn, thao tác nhẹ nhàng</b><p>Chọn video đến 1 phút, bấm kiểm tra và xem lại những điều cần lưu ý.</p></div></div>
  <div class="support-item"><span class="support-icon">♧</span><div><b>Giữ video trong phiên của bạn</b><p>Ứng dụng không tạo thư viện video. Đổi video sẽ xóa kết quả cũ trên màn hình.</p></div></div>
</div>''', unsafe_allow_html=True)
with st.expander("Lần đầu sử dụng? Xem hướng dẫn tại đây"):
    st.write("**1. Chọn video:** bấm Chọn video từ máy hoặc kéo video vào ô chọn tệp. Trên điện thoại, chọn video đã lưu trong thư viện hoặc tệp.")
    st.write("**2. Kiểm tra:** bấm nút Kiểm tra video màu tím. Giữ trang mở cho đến khi có kết quả.")
    st.write("**3. Xem và lưu:** đọc kết luận, các lưu ý và xem lại đoạn được chỉ ra. Bấm Lưu kết quả về máy để giữ bản tóm tắt dễ đọc.")
    st.caption("Chấp nhận MP4/MOV/WebM, tối đa 50 MiB, 60 giây. Video rất lớn hoặc quá nhiều khung hình cần giảm chất lượng trước khi gửi.")
    if st.button("Thử kết nối lại", key="refresh"):
        health_status.clear()
        st.rerun()
st.markdown('<div class="footer"><span>KidSafeReels · Cùng con xem điều hay</span><span>Đồng hành cùng người lớn, không thay thế sự quan tâm.</span></div>', unsafe_allow_html=True)
