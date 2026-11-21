"""Render the Vietnamese technical report with ReportLab and installed Arial fonts."""
from pathlib import Path
import argparse
import json
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=ROOT / "output/pdf/KidSafeReels_BaoCaoKyThuat.pdf")
parser.add_argument("--profile", type=Path)
args = parser.parse_args()
profile = json.loads(args.profile.read_text(encoding="utf-8")) if args.profile else None
OUT = args.output
OUT.parent.mkdir(parents=True, exist_ok=True)
pdfmetrics.registerFont(TTFont("Arial", "C:/Windows/Fonts/arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", "C:/Windows/Fonts/arialbd.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold")
ink, teal = colors.HexColor("#183942"), colors.HexColor("#147D83")
body = ParagraphStyle("Body", fontName="Arial", fontSize=10.2, leading=15.6,
                      textColor=ink, spaceAfter=12, alignment=TA_LEFT)
heading = ParagraphStyle("Heading", parent=body, fontName="Arial-Bold", fontSize=23,
                         leading=30, spaceAfter=22)
cell = ParagraphStyle("Cell", parent=body, fontSize=8.6, leading=12, spaceAfter=0)
title = ParagraphStyle("Title", parent=heading, fontSize=36, leading=44)
small = ParagraphStyle("Small", parent=body, fontSize=9, leading=13)


def rich(text):
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return text.replace("`", "")


def footer(canvas, doc):
    canvas.setFont("Arial", 8)
    canvas.setFillColor(teal)
    canvas.drawString(52, 812, "KIDSAFEREELS  /  HỌC MÁY VÀ ỨNG DỤNG")
    canvas.setStrokeColor(colors.HexColor("#D9E7E6"))
    canvas.line(52, 800, 543, 800)
    canvas.setFillColor(ink)
    canvas.drawString(52, 30, "Bản tích hợp CPU • 18/10/2026 • Pilot đang chờ nhãn")
    canvas.drawRightString(543, 30, str(doc.page))


people = (f"Học phần: Học máy và ứng dụng<br/>Giảng viên: {escape(profile['lecturer'])}<br/>"
          f"Lớp: {escape(profile['class'])}<br/>"
          f"Ứng dụng: {escape(profile['name'])} - MSSV {escape(profile['student_id'])}<br/>"
          "Học máy và điều phối: Huy") if profile else "Học phần: Học máy và ứng dụng<br/>Huy: học máy và điều phối nhóm<br/>Thành viên app: ứng dụng và giao diện"
cover_status = ("Bản tích hợp CPU đã được cài và kiểm thử trên máy thành viên app. "
                "Nhãn pilot và đánh giá chung vẫn chờ người thực hiện.") if profile else "Báo cáo trình bày hai baseline đã huấn luyện, pipeline CPU và sản phẩm tích hợp. Nhãn pilot, đánh giá chung và nghiệm thu trên máy thứ hai chưa hoàn tất."
story = [Spacer(1, 50), Paragraph("KidSafeReels", title),
         Paragraph("Hệ thống kiểm duyệt đa phương thức<br/>nội dung video ngắn nhằm bảo vệ trẻ em", heading),
         Spacer(1, 28), Paragraph("BÁO CÁO KỸ THUẬT", body),
         Paragraph(people, body),
         Spacer(1, 24), Paragraph("Phiên bản ngày 18/10/2026", body),
         Paragraph(cover_status, body),
         Paragraph("Thông tin đầy đủ của thành viên Huy sẽ bổ sung khi nhóm chốt hồ sơ nộp." if profile else "Bổ sung họ tên đầy đủ, MSSV, lớp và thông tin giảng viên theo mẫu học phần trước khi nộp.", small)]
lines = (ROOT / "reports/TECHNICAL_REPORT.md").read_text(encoding="utf-8").splitlines()
i = 0
while i < len(lines):
    line = lines[i].strip()
    if line.startswith("## "):
        story += [PageBreak(), Paragraph(rich(line[3:]), heading)]
    elif line.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            values = [v.strip() for v in lines[i].strip().strip("|").split("|")]
            if not all(re.fullmatch(r"[:\-]+", v) for v in values):
                rows.append([Paragraph(rich(v), cell) for v in values])
            i += 1
        widths = [115, 57, 63, 55, 201] if len(rows[0]) == 5 else [129, 58, 58, 68, 94, 84]
        table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DFEFED")),
            ("LINEBELOW", (0, 0), (-1, 0), 1, teal),
            ("LINEBELOW", (0, 1), (-1, -1), .4, colors.HexColor("#CDDCD9")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ]))
        story += [table, Spacer(1, 16)]
        continue
    elif line and not line.startswith("# ") and any(x.startswith("## ") for x in lines[:i]):
        story.append(Paragraph(rich(line), body))
    i += 1
SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=52, leftMargin=52,
                  topMargin=66, bottomMargin=56, title="KidSafeReels - Bao cao ky thuat",
                  author="Nhom KidSafeReels").build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
