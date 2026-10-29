"""Small, escaped HTML views for the family-facing Streamlit interface."""

from html import escape

from apps.web.client import DECISIONS, LABELS, MODALITIES, timestamp

SHIELD = '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 3 4.5 6v5c0 4.6 3.1 8 7.5 10 4.4-2 7.5-5.4 7.5-10V6L12 3Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="m8.5 11.8 2.4 2.4 4.6-4.8" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>'

ICONS = {
    "violence": "✦", "sexual": "◇", "toxic_language": "≋", "dangerous_acts": "△",
}


def toxic_method_note(result):
    if result["mode"] == "model" and "+toxic-hybrid-" in result["model_version"]:
        return "Ngôn ngữ độc hại được kiểm tra bằng model và bộ luật từ/cụm từ. Chỉ số hiển thị là điểm model; cảnh báo từ bộ luật có thể xuất hiện dù điểm này thấp. Nguồn cảnh báo nằm trong mô tả từng đoạn."
    return None


def header(health):
    if health is None:
        status, style = "Chưa kết nối", "offline"
    elif health.get("backend") == "mock":
        status, style = "Bản dùng thử", "trial"
    else:
        status, style = "Kiểm tra trước khi chia sẻ", "ready"
    return f'''<div class="topbar">
        <a class="brand" href="/" target="_self" aria-label="KidSafeReels, trang chủ">
        <span class="brand-icon">{SHIELD}</span><span>KidSafe<span class="brand-accent">Reels</span><small>CÙNG CON XEM ĐIỀU HAY</small></span></a>
        <span class="connection {style}"><span class="status-dot"></span>{status}</span>
    </div>'''


def hero():
    return '''<section class="hero">
      <div class="hero-copy"><span class="eyebrow">MỘT CHÚT QUAN TÂM, THÊM NHIỀU AN TÂM</span>
      <h1>Video hay cho con,<br><span class="hero-accent">bắt đầu từ bạn.</span></h1>
      <p>Kiểm tra nội dung video trước khi cho trẻ xem.<br>Để mỗi khoảnh khắc giải trí đều có sự đồng hành của bạn.</p>
      <div class="hero-tags"><span>✧ Dễ sử dụng</span><span>◷ Video đến 1 phút</span><span>♡ Dành cho gia đình</span></div></div>
      <div class="hero-art" aria-hidden="true">
      <svg viewBox="0 0 410 255" fill="none">
        <circle cx="218" cy="129" r="111" fill="#DED7FC"/>
        <circle cx="334" cy="65" r="21" fill="#FFE7CE"/><path d="m333 55 2 7 7 3-7 2-2 7-3-7-7-2 7-3 3-7Z" fill="#DB9B61"/>
        <rect x="83" y="63" width="245" height="153" rx="17" fill="#BDB0EF" transform="rotate(-7 83 63)"/>
        <rect x="89" y="49" width="245" height="153" rx="17" fill="white" stroke="#D4C9F3" stroke-width="2"/>
        <rect x="102" y="62" width="219" height="122" rx="9" fill="#EFEAFB"/>
        <path d="M102 162c41-36 80-21 112-11 51-39 82-29 107-11v35a9 9 0 0 1-9 9H111a9 9 0 0 1-9-9v-13Z" fill="#D4E8DE"/>
        <circle cx="145" cy="91" r="13" fill="#F8D7A4"/>
        <circle cx="174" cy="111" r="18" fill="#E8AE89"/>
        <path d="M156 108c-3-20 32-29 37-2-7-1-15-4-18-10-4 8-10 10-19 12Z" fill="#403653"/>
        <path d="M142 172c0-32 15-44 32-44s32 12 32 44" fill="#8D7BC7"/>
        <circle cx="231" cy="135" r="14" fill="#F1BC98"/>
        <path d="M217 134c-1-20 30-22 29 1-9-2-17-5-19-9l-10 8Z" fill="#473A53"/>
        <path d="M207 180c0-24 10-32 24-32s25 8 25 32" fill="#EAB67C"/>
        <path d="M191 156c12 10 20 12 24 9" stroke="#E8AE89" stroke-width="11" stroke-linecap="round"/>
        <circle cx="305" cy="186" r="36" fill="#7255C5" stroke="white" stroke-width="5"/>
        <path d="m305 163-18 7v12c0 11 7 19 18 24 11-5 18-13 18-24v-12l-18-7Z" stroke="white" stroke-width="2.5" stroke-linejoin="round"/>
        <path d="m297 183 6 6 11-12" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
        <rect x="47" y="140" width="48" height="40" rx="12" fill="#FFF" stroke="#E3DCF4"/>
        <path d="m67 151 13 9-13 9v-18Z" fill="#A190D8"/>
        <path d="m80 58 3 7 7 3-7 3-3 7-3-7-7-3 7-3 3-7Z" fill="#A18BD1"/>
        <circle cx="358" cy="149" r="5" fill="#B4D0C3"/>
      </svg></div>
    </section>'''


def steps(stage):
    items = [("Chọn video", "Từ điện thoại hoặc máy tính"), ("Kiểm tra nội dung", "Ứng dụng xử lý video"), ("Xem kết quả", "Cùng cân nhắc trước khi chia sẻ")]
    return '<div class="steps" aria-label="Các bước kiểm tra video">' + ''.join(
        f'<div class="step {"active" if index == stage else "done" if index < stage else ""}"><span class="step-number">{"✓" if index < stage else index + 1}</span><div><b>{title}</b><small>{description}</small></div></div>'
        for index, (title, description) in enumerate(items)
    ) + '</div>'


def empty_result():
    return f'''<div class="empty-result"><div class="empty-icon">{SHIELD}</div>
      <h3>Một chút chờ đợi,<br>thêm góc nhìn cho bạn</h3>
      <p>Chọn video rồi bấm <b>“Kiểm tra video”</b>.<br>Kết quả sẽ xuất hiện ngay tại đây.</p>
      <div class="empty-benefits"><span>✓ Nhóm nội dung cần lưu ý</span><span>✓ Các đoạn cần xem lại, khi có</span><span>✓ Bản kết quả có thể tải về</span></div>
      <small>Chưa có video nào được đánh giá.</small></div>'''


def score_cards(scores):
    cards = []
    for label, score in scores.items():
        assessed = score is not None
        cards.append(f'''<div class="risk-card {"assessed" if assessed else "unknown"}">
          <span class="risk-icon">{ICONS[label]}</span><div><b>{LABELS[label]}</b>
          <small>{"Có kết quả phân tích" if assessed else "Chưa đánh giá"}</small></div>
          <span class="risk-mark">{"•" if assessed else "—"}</span></div>''')
    return '<div class="risk-grid">' + ''.join(cards) + '</div>'


def decision_card(result):
    decision = result["decision"]
    descriptions = {
        "review": "Bạn nên xem lại video và các lưu ý bên dưới trước khi cho trẻ xem.",
        "flagged": "Có dấu hiệu cần lưu ý. Hãy xem lại các đoạn được chỉ ra trước khi chia sẻ với trẻ.",
        "no_risk_detected": "Chỉ áp dụng cho những nhóm đã được đánh giá. Bạn vẫn cần cân nhắc nội dung và độ tuổi của trẻ.",
    }
    symbols = {"review": "◷", "flagged": "!", "no_risk_detected": "✓"}
    return f'<div class="decision {decision}"><span class="decision-symbol">{symbols[decision]}</span><div><h3>{DECISIONS[decision]}</h3><p>{descriptions[decision]}</p></div></div>'


def report_text(result, filename):
    lines = ["KIDSAFE REELS — KẾT QUẢ KIỂM TRA VIDEO", f"Video: {filename}"]
    if result["mode"] == "mock":
        lines += ["KẾT QUẢ MÔ PHỎNG: chưa đánh giá nội dung video."]
    lines += [f"Kết luận: {DECISIONS[result['decision']]}", "", "CÁC NHÓM NỘI DUNG"]
    for label, score in result["scores"].items():
        value = "Chưa đánh giá" if score is None else f"Có kết quả; chỉ số phân tích {score:.3f}"
        lines.append(f"- {LABELS[label]}: {value}")
    if toxic_method_note(result):
        lines += ["", toxic_method_note(result)]
    lines += ["", "LƯU Ý", *[f"- {warning}" for warning in result["warnings"]], "", "CÁC ĐOẠN CẦN XEM LẠI"]
    for item in result["evidence"]:
        lines.append(f"- {timestamp(item['start_seconds'])}–{timestamp(item['end_seconds'])} | {LABELS[item['label']]} | {MODALITIES[item['modality']]}: {item['description']}")
    if not result["evidence"]:
        lines.append("Không có đoạn cụ thể được cung cấp; điều này không đồng nghĩa video an toàn.")
    lines += ["", "Kết quả hỗ trợ người lớn cân nhắc, không bảo đảm an toàn tuyệt đối.",
              "Chỉ số 0–1 không phải xác suất đã được hiệu chỉnh.", f"Phiên bản phân tích: {result['model_version']}",
              f"Mã kết quả: {result['analysis_id']}"]
    return "\n".join(lines)


def filename_card(name, size):
    return f'<div class="file-info"><span>▤</span><div><b>{escape(name)}</b><small>{size / (1024 * 1024):.2f} MiB · Đã chọn video</small></div></div>'
