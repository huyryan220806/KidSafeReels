"""Render the 20 November Toxic report using the measured aggregates and figures."""
import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'reports/toxic-transcript-2026-11-20'
PREVIOUS = ROOT/'reports/toxic-human-2026-11-13'
TARGET = ROOT/'output/pdf/KidSafeReels_Toxic_2026-11-20.pdf'
TEAL = colors.HexColor('#006c63')
INK = colors.HexColor('#17353c')
MUTED = colors.HexColor('#415963')
WIDTH = 493


def rows(path):
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def pct(number):
    return f'{float(number)*100:.2f}%'.replace('.',',')


def paragraph(text, size=10.5, bold=False, color=INK, after=7):
    return Paragraph(text, ParagraphStyle('p',fontName='VN-B' if bold else 'VN',
        fontSize=size, leading=size*1.4, textColor=color,spaceAfter=after))


def heading(number, title, lead):
    return [paragraph(number,9,True,TEAL),paragraph(title,23,True),paragraph(lead,12,False,MUTED,13)]


def table(header, body, widths, highlight=None):
    matrix = [[paragraph(t,9,True,colors.white,0) for t in header]]
    for row in body:
        matrix.append([paragraph(str(value),9.3,False,INK,0) for value in row])
    t = Table(matrix,colWidths=widths,hAlign='LEFT',repeatRows=1)
    style = [('BACKGROUND',(0,0),(-1,0),INK),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
             ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f0f5f4')]),
             ('LINEBELOW',(0,-1),(-1,-1),.5,TEAL),
             ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]
    if highlight:
        style.append(('BACKGROUND',(0,highlight),(-1,highlight),colors.HexColor('#d6eee7')))
    t.setStyle(TableStyle(style))
    return t


def frame(canvas, doc):
    canvas.setFont('VN-B',9);canvas.setFillColor(TEAL)
    canvas.drawString(51,804,'KIDSAFEREELS   /   HỌC MÁY VÀ ỨNG DỤNG')
    canvas.setFont('VN',9);canvas.setFillColor(MUTED);canvas.drawRightString(544,804,'20.11.2026')
    canvas.setStrokeColor(colors.HexColor('#c9d9d6'));canvas.line(51,793,544,793)
    canvas.line(51,46,544,46);canvas.setFont('VN',8.5)
    canvas.drawString(51,31,'Đánh giá nhánh ngôn ngữ độc hại trong video ngắn tiếng Việt')
    canvas.drawRightString(544,31,str(doc.page))


def figure(name, inches_ratio):
    return Image(str(SOURCE/name),width=WIDTH,height=WIDTH*inches_ratio)


def main():
    for name,file in [('VN','arial.ttf'),('VN-B','arialbd.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
    validation = {r['method']:r for r in rows(SOURCE/'decision_metrics.csv') if r['split']=='valid'}
    methods = [('rule','Luật từ / cụm từ'),('baseline','PhoBERT v3 + luật'),
               ('selected','PhoBERT 13/11 + luật'),('text','Ký tự'),('text+rule','Ký tự + luật')]
    values = [[label]+[pct(validation[name][k]) for k in ('accuracy','precision','recall','f1')]
              for name,label in methods]
    story = heading('01   /   KẾT QUẢ TRÊN LỜI NÓI','Phát hiện lời nói độc hại trong video ngắn',
        'Trên 28 clip validation, bộ phân loại ký tự kết hợp luật phát hiện 13/15 clip Toxic. Cả 13 clip hội thoại bình thường đều qua bước sàng lọc mà không bị cảnh báo Toxic.')
    story += [paragraph('So sánh trên cùng 28 clip',14,True),table(
        ['Cấu hình','Accuracy','Precision¹','Recall¹','F1¹'],values,[167,82,82,79,83],5),
        paragraph('¹ Precision, Recall và F1 tính cho lớp Toxic. Mỗi clip đóng góp một quyết định.',8.8,False,MUTED),
        figure('errors.png',2.8/8),Spacer(1,6),
        paragraph('Báo nhầm giảm, số bỏ sót bằng ứng viên PhoBERT',14,True),
        paragraph('So với PhoBERT ngày 13/11 kết hợp luật, ứng viên mới sửa hai cảnh báo nhầm. Trong lớp Toxic, nó bắt thêm một clip cũ bị bỏ sót nhưng bỏ sót một clip cũ đã bắt được. Hai hệ thống cùng đạt Recall 86,67%.'),
        paragraph('Thành phần nào đóng góp?',12,True),
        paragraph('Luật phát hiện 11 clip Toxic. Nhánh ký tự bắt thêm hai clip ngoài các cảnh báo của luật. Tập validation này đã được dùng trong các lần phát triển trước. Bản đang chạy trong app là PhoBERT v3 kết hợp luật.'),
        PageBreak()]

    story += heading('02   /   THIẾT KẾ VÀ HUẤN LUYỆN','Từ lời nói đến điểm Toxic',
        'Whisper small chép lời tiếng Việt từ audio 16 kHz. PhoBERT chấm từng đoạn và lấy điểm lớn nhất. Ứng viên ký tự đọc bản chép của cả clip trong một lượt.')
    story += [table(['Tập dữ liệu','Số clip','Bình thường','Toxic','Clip có ASR'],
        [['Train','142','71','71','138'],['Validation','28','13','15','28'],['Test cũ','30','14','16','29']],
        [137,70,101,70,115]),
        Spacer(1,9),paragraph('Bộ phân loại ký tự',13,True),
        paragraph('138 transcript train có lời được chuẩn hóa Unicode NFC, giữ dấu và chuyển chữ thường. TF-IDF mã hóa cụm 3–5 ký tự trong ranh giới từ, giữ đặc trưng xuất hiện ở ít nhất hai clip. Logistic regression dùng L2, C = 1, ngưỡng 0,50. Luật được áp dụng riêng cho từng đoạn ASR. Clip thiếu lời giữ trạng thái cần xem lại.'),
        paragraph('PhoBERT và đường học',13,True),
        paragraph('Đợt 13/11 khởi tạo từ v3, cập nhật hai khối encoder cuối và đầu phân loại trong 5 epoch. Loss gồm BCE cấp clip và 0,5 lần cross-entropy trên bình luận ViCTSD train. Đường học dùng ngưỡng 0,50. Ngưỡng khi so cấu hình v3 và ứng viên 13/11 lần lượt là 0,78 và 0,80.'),
        figure('learning.png',2.8/8.4),
        paragraph('Epoch 1 → 5, log-loss validation giảm từ 1,250 xuống 0,627 và AP tăng từ 0,767 lên 0,830. Điểm của 28 clip vẫn ở cùng phía ngưỡng 0,50, nên Accuracy 64,29% và F1 58,33% của riêng PhoBERT giữ nguyên. AP đo thứ tự xếp hạng theo điểm Toxic.'),
        paragraph('Riêng nhánh ký tự đạt log-loss 0,490 và AP 0,967 trên 28 clip. Quyết định kết hợp luật là nhị phân nên chỉ số loss trong báo cáo áp dụng cho nhánh có điểm xác suất.',9.5,False,MUTED),PageBreak()]

    transfer = json.loads((SOURCE/'transfer-and-pairs.json').read_text(encoding='utf-8'))['written_valid_model_only']
    written = next(r for r in rows(PREVIOUS/'written-validation.csv') if r['variant']=='selected')
    story += heading('03   /   GIỚI HẠN VÀ QUYẾT ĐỊNH','Mức cải thiện phụ thuộc vào dữ liệu',
        'Kết quả trên video đủ để ưu tiên thử tiếp bộ phân loại ký tự. Kiểm tra trên bình luận viết cho thấy việc đưa nó vào ứng dụng cần thêm dữ liệu lời nói độc lập.')
    story += [paragraph('Kiểm chéo theo nguồn video',13,True),
        paragraph('Mỗi clip train được dự đoán ở một trong 5 lượt giữ riêng nhóm nguồn. Ký tự + luật đạt F1 92,09% trên 142 clip, với 4 cảnh báo nhầm và 7 bỏ sót. Có 4 clip Toxic thiếu transcript. Khi chỉ dùng độ dài văn bản và số đoạn ASR kết hợp luật, F1 còn 82,19%.'),
        paragraph('Trên validation, 17 clip từ kênh chưa xuất hiện ở train có đến 14 clip Toxic. 11 clip từ kênh đã thấy chỉ có 1 clip Toxic. Sự chênh lệch này và việc thiếu ID người nói đã xác minh khiến việc đánh giá trên nguồn mới đặc biệt quan trọng. Hai nhóm video ban đầu cũng đã được sàng theo ASR và từ khóa.'),
        paragraph('Chuyển sang bình luận ViCTSD',13,True),
        table(['Model đơn','Ngưỡng','Precision','Recall','F1 Toxic','FP / FN'],
             [[title,t]+[pct(r[k]) for k in ('precision','recall','f1')]+[f"{r['fp']} / {r['fn']}"]
              for title,t,r in [('PhoBERT 13/11','0,80',written),('Ký tự','0,50',transfer)]],
             [122,65,77,66,72,91]),
        Spacer(1,7),paragraph('Cùng 1.975 bình luận validation, gồm 232 nhãn Toxic. PhoBERT đã học từ bình luận ViCTSD train. Bộ phân loại ký tự chỉ học từ video. Các ngưỡng đã được cố định trước khi đối chiếu.'),
        paragraph('Đánh giá test hiện có',13,True),
        paragraph('Trên 30 clip test cũ, v3 + luật và PhoBERT 13/11 + luật cùng đạt Accuracy 63,33%, F1 64,52%, FP 5 và FN 6. Thử nghiệm ký tự ngày 20/11 chỉ đánh giá trong train và validation. Điểm test của ứng viên mới còn để trống.'),
        paragraph('Lần kiểm chứng kế tiếp',13,True),
        paragraph('Khóa hai cấu hình và chạy song song trên video mới, có cả lời nói bình thường lẫn Toxic từ cùng kiểu nội dung. Ưu tiên các kênh và người nói mới. Người duyệt gán nhãn trước khi xem dự đoán. So sánh số clip báo nhầm và bỏ sót theo từng nguồn, rồi mới quyết định thay mô hình trong app.'),
        paragraph('Nguồn số liệu',9,True,TEAL),
        paragraph('KidSafeReels · toxic-human-2026-11-13 (huấn luyện và test cũ) · phobert-audit-2026-11-15 (đối chiếu từng epoch) · toxic-transcript-2026-11-20 (ứng viên ký tự, kiểm chéo và ViCTSD).',8.5,False,MUTED)]
    TARGET.parent.mkdir(parents=True,exist_ok=True)
    doc = SimpleDocTemplate(str(TARGET),pagesize=A4,leftMargin=45,rightMargin=45,topMargin=65,bottomMargin=57,
                            title='KidSafeReels | Đánh giá phát hiện lời nói độc hại',author='KidSafeReels')
    doc.build(story,onFirstPage=frame,onLaterPages=frame)


if __name__=='__main__':
    main()
