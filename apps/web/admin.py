"""Local diagnostics view, enabled explicitly by the app operator."""

import json

import streamlit as st

from apps.validation import validate_result
from apps.web.client import LABELS, MODALITIES, timestamp


def render_admin(health, api_url):
    st.title("Quản trị kỹ thuật")
    st.caption("Công cụ chẩn đoán local cho người phát triển. Chỉ xem dữ liệu trong phiên đang mở; không có lịch sử video.")
    st.write(f"API: {api_url}")
    st.json(health or {"status": "offline"}, expanded=False)
    if "analysis_error" in st.session_state:
        st.error(st.session_state["analysis_error"])
    if "error_detail" in st.session_state:
        st.write(st.session_state["error_detail"])
    if "analysis_result" not in st.session_state:
        st.info("Chưa có kết quả trong phiên này. Chuyển về Dành cho gia đình để chọn và kiểm tra video.")
        return
    try:
        result = validate_result(st.session_state["analysis_result"]).model_dump(mode="json")
    except (ValueError, TypeError):
        st.error("Kết quả không đúng hợp đồng API. Kiểm tra tích hợp trước khi hiển thị.")
        return
    st.subheader("Cảnh báo và giới hạn từ bộ phân tích")
    for warning in dict.fromkeys(result["warnings"]):
        st.write("• " + warning)
    if not result["warnings"]:
        st.caption("Bộ phân tích không trả cảnh báo.")
    st.subheader("Điểm và bằng chứng")
    st.caption("Chỉ số 0–1 không phải xác suất đã hiệu chỉnh. Kết luận và ngưỡng do model cung cấp.")
    st.table([
        {"Nhóm nội dung": LABELS[label], "Chỉ số phân tích": "Chưa đánh giá" if score is None else f"{score:.3f}"}
        for label, score in result["scores"].items()
    ])
    if result["evidence"]:
        st.dataframe([
            {"Từ": timestamp(e["start_seconds"]), "Đến": timestamp(e["end_seconds"]),
             "Nhóm": LABELS[e["label"]], "Nguồn": MODALITIES[e["modality"]],
             "Chỉ số": e["score"], "Mô tả": e["description"]} for e in result["evidence"]
        ], hide_index=True, use_container_width=True)
    st.caption(f"Chế độ: {result['mode']} · Thời gian xử lý: {result['processing_seconds']:.2f} giây · Phiên bản: {result['model_version']} · Schema: {result['schema_version']}")
    st.code(result["analysis_id"], language=None)
    st.download_button("Tải dữ liệu JSON", json.dumps(result, ensure_ascii=False, indent=2), "kidsafe-result.json", "application/json")
    with st.expander("Thông tin cho người cài đặt"):
        st.write("Chạy Chay-KidSafeReels.cmd cho giao diện gia đình; thêm -Admin khi chạy scripts/run-model.ps1 để bật công cụ này.")
        st.caption("API xóa tệp tạm sau xử lý. Web giữ tệp trong phiên. Xem docs/API_CONTRACT.md về giới hạn media và timeout.")
