import json
import os

import requests
import streamlit as st

st.set_page_config(page_title="KidSafeReels", page_icon="🛡️")
st.title("KidSafeReels")
st.caption("Hỗ trợ sàng lọc nội dung video ngắn trước khi cho trẻ xem.")
st.info("Bản thử nghiệm hỗ trợ người lớn kiểm tra video. Kết quả ghi rõ chế độ mô phỏng hoặc model.")

api_url = os.getenv("KIDSAFE_API_URL", "http://127.0.0.1:8000").rstrip("/")
upload = st.file_uploader(
    "Chọn video (tối đa 50 MiB)",
    type=["mp4", "mov", "webm"],
    on_change=lambda: st.session_state.pop("analysis_result", None),
)
st.caption("Mục tiêu MVP: video tối đa 60 giây. Chế độ mock chưa kiểm tra thời lượng.")

if st.button("Phân tích video", disabled=upload is None):
    st.session_state.pop("analysis_result", None)
    if upload.size > 50 * 1024 * 1024:
        st.error("Video vượt giới hạn 50 MiB.")
    else:
        try:
            with st.spinner("Đang gửi video và chờ kết quả…"):
                response = requests.post(
                    f"{api_url}/v1/analyze",
                    files={"file": (upload.name, upload.getvalue(), upload.type)},
                    timeout=(5, 180),
                )
                response.raise_for_status()
                st.session_state["analysis_result"] = response.json()
        except requests.Timeout:
            st.error("Hết thời gian chờ. Kiểm tra API và thử video ngắn hơn.")
        except requests.ConnectionError:
            st.error("Chưa kết nối được API. Hãy chạy lệnh uvicorn trong README.")
        except requests.HTTPError as exc:
            st.error(f"API trả lỗi {exc.response.status_code}: {exc.response.text[:300]}")
        except (ValueError, KeyError, TypeError):
            st.error("API trả dữ liệu không đúng hợp đồng. Kiểm tra phiên bản API.")

if "analysis_result" in st.session_state:
    try:
        result = st.session_state["analysis_result"]
        if result["mode"] == "mock":
            st.warning("KẾT QUẢ MÔ PHỎNG — chưa đánh giá nội dung video.")
        decisions = {
            "no_risk_detected": "Chưa phát hiện rủi ro trong phạm vi model đã đánh giá",
            "review": "Cần người lớn kiểm tra",
            "flagged": "Phát hiện dấu hiệu rủi ro",
        }
        st.subheader(decisions[result["decision"]])
        st.table([
            {"Nhãn": label, "Điểm model": "Chưa đánh giá" if score is None else f"{score:.3f}"}
            for label, score in result["scores"].items()
        ])
        for warning in result["warnings"]:
            st.warning(warning)
        if result["evidence"]:
            st.dataframe(result["evidence"])
        st.caption(f"Phiên bản: {result['model_version']} · {result['processing_seconds']:.2f} giây")
        st.download_button(
            "Tải báo cáo JSON",
            json.dumps(result, ensure_ascii=False, indent=2),
            "kidsafe-result.json",
            "application/json",
        )
    except (ValueError, KeyError, TypeError, AttributeError):
        st.session_state.pop("analysis_result", None)
        st.error("API trả dữ liệu không đúng hợp đồng. Kiểm tra phiên bản API.")
