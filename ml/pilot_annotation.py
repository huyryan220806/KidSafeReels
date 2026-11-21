"""Create independent local annotation packs without predictions or ASR text."""
import argparse
import csv
import html
import json
from pathlib import Path
import zipfile

from ml.pilot_data import ROOT, MANIFEST, OUTPUT, load_manifest, read_json, sha, write_json

PRIVATE = ROOT / "data/private/pilot-v1"
DOUBLE_IDS = {f"vi-pilot-{number:03d}" for number in (1, 5, 11, 16, 22, 23)}
LABELS = ("violence", "toxic_language")


def annotation_rows(data, annotator):
    if annotator not in {"huy", "app"}:
        raise ValueError("Unknown annotator")
    return [{"video_id": c["video_id"], "annotator": annotator, "watched_full": False,
             "language": None, "violence": None, "toxic_language": None,
             "violence_evidence": [], "toxic_language_evidence": [], "notes": ""}
            for c in data["clips"] if annotator == "huy" or c["video_id"] in DOUBLE_IDS]


def validate_annotations(payload, data, annotator):
    """Unknown and unfinished labels never turn into negatives or gold."""
    if payload.get("manifest_sha256") != sha(MANIFEST) or payload.get("annotator") != annotator:
        raise ValueError("Annotation version/annotator mismatch")
    expected = {r["video_id"] for r in annotation_rows(data, annotator)}
    rows = payload["rows"]
    if len(rows) != len(expected) or {r["video_id"] for r in rows} != expected:
        raise ValueError("Missing/duplicate/unassigned annotation IDs")
    durations = {c["video_id"]: c["end_seconds"] - c["start_seconds"] for c in data["clips"]}
    completed = 0
    for row in rows:
        if row["annotator"] != annotator or type(row["watched_full"]) is not bool:
            raise ValueError("Invalid annotation identity/status")
        if row["language"] not in {None, "vi", "other", "no_speech", "uncertain"}:
            raise ValueError("Unknown language status")
        for label in LABELS:
            value = row[label]
            if not (value is None or value == "uncertain" or type(value) is int and value in (0, 1)):
                raise ValueError("Labels must be null, uncertain, 0 or 1")
            evidence = row[label + "_evidence"]
            if value == 1 and not evidence:
                raise ValueError("Positive label requires a timestamp and reason")
            for interval in evidence:
                start, end = interval["start_seconds"], interval["end_seconds"]
                if not 0 <= start < end <= durations[row["video_id"]] or not interval["reason"].strip():
                    raise ValueError("Invalid evidence interval/reason")
            if value is not None and not row["watched_full"]:
                raise ValueError("Label entered before full viewing confirmation")
        if row["watched_full"] and row["language"] is not None and all(row[k] is not None for k in LABELS):
            completed += 1
    return completed


def review_html(data, annotator, rows):
    sources = {s["source_id"]: s for s in data["sources"]}
    clips = {c["video_id"]: c for c in data["clips"]}
    cards = []
    for row in rows:
        video_id = row["video_id"]
        clip = clips[video_id]
        source = sources[clip["source_id"]]
        choices = '<option value="">Chưa gán</option><option value="0">0 — không có</option><option value="1">1 — có</option><option value="uncertain">Chưa rõ</option>'
        fields = "".join(f'<label>{label}<select data-field="{label}">{choices}</select></label>'
                         f'<label>Mốc {label} (giây): <input data-field="{label}_start" type="number" min="0" max="25" step="0.01" placeholder="Bắt đầu">'
                         f' <input data-field="{label}_end" type="number" min="0" max="25" step="0.01" placeholder="Kết thúc"></label>'
                         f'<label>Lý do / lời nói nghe được<input data-field="{label}_reason"></label>' for label in LABELS)
        attribution = html.escape(f"{source['title']} — {source['author']}")
        cards.append(f'''<article data-id="{video_id}"><h2>{video_id}</h2>
<video controls preload="metadata" src="media/{video_id}.mp4"></video>
<p>Đoạn 25 giây; mốc gán nhãn tính từ đầu đoạn. <span class="time"></span></p>
<label><input type="checkbox" data-field="watched_full"> Tôi đã xem/nghe toàn bộ clip</label>
<label>Ngôn ngữ <select data-field="language"><option value="">Chưa xác nhận</option><option value="vi">Có lời nói tiếng Việt</option><option value="other">Ngôn ngữ khác</option><option value="no_speech">Không có lời nói</option><option value="uncertain">Không nghe rõ</option></select></label>
{fields}<label>Ghi chú<textarea data-field="notes"></textarea></label>
<details><summary>Nguồn và ghi công</summary><p>{attribution}</p>
<a href="{html.escape(source['source_url'], quote=True)}">Trang nguồn</a> · <a href="{source['license_url']}">CC BY 3.0</a>
<p>Đã cắt {clip['start_seconds']}–{clip['end_seconds']} giây, đổi độ phân giải/codec; không lồng tiếng.</p></details></article>''')
    initial = json.dumps({"manifest_sha256": sha(MANIFEST), "annotator": annotator, "rows": rows}, ensure_ascii=False).replace("<", "\\u003c")
    return '''<!doctype html><html lang="vi"><meta charset="utf-8"><title>KidSafeReels — Gán nhãn độc lập</title>
<style>body{font:16px system-ui;max-width:900px;margin:32px auto;padding:0 20px;background:#f5f7fa;color:#17263b}header{position:sticky;top:0;background:#f5f7fa;padding:12px 0;z-index:2}article{background:white;border:1px solid #ccd4df;border-radius:12px;padding:20px;margin:20px 0}video{width:100%;max-height:400px;background:#111}label{display:block;margin:12px 0}input:not([type=checkbox]),textarea,select{font:inherit;padding:6px;max-width:100%;box-sizing:border-box}input:not([type=number]):not([type=checkbox]),textarea{width:100%}input[type=number]{width:130px}select{margin-left:8px}button{font:inherit;padding:10px 18px;background:#174b8d;color:white;border:0;border-radius:6px;cursor:pointer}.notice{color:#734800}details{font-size:14px}</style>
<header><h1>Gán nhãn độc lập — ''' + html.escape(annotator) + '''</h1>
<button id="save">Tải JSON tiến độ</button> <label>Nạp lại JSON <input id="load" type="file" accept=".json"></label><span id="status"></span></header>
<p>Đọc <a href="ANNOTATION_GUIDE.md">hướng dẫn nhãn</a>. Xem/nghe video trước khi gán, không mở dự đoán/ASR hoặc nhãn của người khác. Bỏ trống khác với nhãn 0. Có rủi ro: ghi ít nhất một mốc và lý do; các mốc bổ sung có thể ghi vào ghi chú.</p>
<p class="notice">Đây là công cụ chuẩn bị dữ liệu cho người lớn. Nhãn chỉ bao phủ Violence/Toxic; không kết luận video an toàn cho trẻ. Tải JSON thường xuyên để lưu tiến độ.</p>''' + "\n".join(cards) + '''
<script>
const initial = ''' + initial + ''';
const key = 'kidsafe-annotation-' + initial.annotator + '-' + initial.manifest_sha256;
const status = document.querySelector('#status');
function collect(){return {...initial, saved_at:new Date().toISOString(), rows:[...document.querySelectorAll('article')].map(a=>{
 const get=k=>a.querySelector('[data-field="'+k+'"]');
 const row={video_id:a.dataset.id,annotator:initial.annotator,watched_full:get('watched_full').checked,language:get('language').value||null,notes:get('notes').value,draft_fields:{}};
 for(const label of ['violence','toxic_language']){const v=get(label).value;row[label]=v===''?null:v==='uncertain'?v:Number(v);
 const start=get(label+'_start').value,end=get(label+'_end').value,reason=get(label+'_reason').value;
 row.draft_fields[label]={start,end,reason};row[label+'_evidence']=start!==''&&end!==''?[{start_seconds:Number(start),end_seconds:Number(end),reason}]:[];}
 return row;})};}
function restore(p){
 if(p.manifest_sha256!==initial.manifest_sha256||p.annotator!==initial.annotator)throw Error('Sai người gán hoặc phiên bản dữ liệu');
 if(!Array.isArray(p.rows)||p.rows.length!==initial.rows.length||new Set(p.rows.map(r=>r.video_id)).size!==initial.rows.length||p.rows.some(r=>!initial.rows.some(i=>i.video_id===r.video_id)))throw Error('Sai danh sách clip');
 for(const row of p.rows){const a=[...document.querySelectorAll('article')].find(a=>a.dataset.id===row.video_id);const get=k=>a.querySelector('[data-field="'+k+'"]');
 get('watched_full').checked=row.watched_full===true;get('language').value=row.language??'';get('notes').value=row.notes??'';
 for(const label of ['violence','toxic_language']){get(label).value=row[label]??'';const e=(row[label+'_evidence']||[])[0]||{},d=(row.draft_fields||{})[label]||{};get(label+'_start').value=d.start??e.start_seconds??'';get(label+'_end').value=d.end??e.end_seconds??'';get(label+'_reason').value=d.reason??e.reason??'';}}
}
function localSave(){try{localStorage.setItem(key,JSON.stringify(collect()));status.textContent='Đã lưu tạm trong trình duyệt; tải JSON để sao lưu.'}catch{status.textContent='Trình duyệt không lưu tạm được. Hãy tải JSON.'}}
document.querySelector('#save').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(collect(),null,2)],{type:'application/json'}));a.download='annotations-'+initial.annotator+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
document.querySelector('#load').onchange=async e=>{try{restore(JSON.parse(await e.target.files[0].text()));localSave()}catch(err){status.textContent='Không nạp được: '+err.message}};
document.querySelectorAll('[data-field]').forEach(el=>el.addEventListener('change',localSave));
document.querySelectorAll('video').forEach(v=>v.addEventListener('timeupdate',()=>v.closest('article').querySelector('.time').textContent='Đang ở '+v.currentTime.toFixed(2)+'s'));
try{const old=localStorage.getItem(key);if(old)restore(JSON.parse(old))}catch{status.textContent='Không nạp được bản tạm. Có thể nạp JSON đã tải.'}
</script></html>'''


def make_pack(data):
    PRIVATE.mkdir(parents=True, exist_ok=True)
    assignments = []
    for annotator in ("huy", "app"):
        rows = annotation_rows(data, annotator)
        for row in rows:
            assignments.append({"video_id": row["video_id"], "annotator": annotator})
        target = PRIVATE / f"review-{annotator}.html"
        target.write_text(review_html(data, annotator, rows), encoding="utf-8")
        template = PRIVATE / f"annotations-{annotator}.blank.json"
        if not template.exists():
            write_json(template, {"manifest_sha256": sha(MANIFEST), "annotator": annotator, "rows": rows})
    with (ROOT / "data/pilot/annotation_assignments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "annotator"])
        writer.writeheader()
        writer.writerows(assignments)
    # HTML works both locally and in ZIP; copies are small, no link to private data outside pack.
    import shutil
    (PRIVATE / "media").mkdir(exist_ok=True)
    for clip in data["clips"]:
        shutil.copy2(ROOT / clip["video_path"], PRIVATE / "media" / (clip["video_id"] + ".mp4"))
    shutil.copy2(ROOT / "docs/ANNOTATION_GUIDE.md", PRIVATE / "ANNOTATION_GUIDE.md")
    shutil.copy2(ROOT / "data/pilot/ATTRIBUTION.md", PRIVATE / "ATTRIBUTION.md")
    archive = ROOT / "data/private/kidsafe-pilot-v1-annotation.zip"
    # Explicit allowlist: never include submitted labels, ASR, predictions or arbitrary private files.
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in ["review-huy.html", "review-app.html", "annotations-huy.blank.json", "annotations-app.blank.json",
                     "ANNOTATION_GUIDE.md", "ATTRIBUTION.md"]:
            bundle.write(PRIVATE / name, name)
        for clip in data["clips"]:
            name = "media/" + clip["video_id"] + ".mp4"
            bundle.write(PRIVATE / name, name)
    print(json.dumps({"archive": str(archive), "size_bytes": archive.stat().st_size,
                      "huy_clips": len(data["clips"]), "double_annotation_clips": len(DOUBLE_IDS)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["pack", "check"])
    parser.add_argument("--annotator", choices=["huy", "app"])
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    data = load_manifest()
    if args.command == "pack":
        make_pack(data)
    else:
        if not args.annotator or not args.input:
            parser.error("check needs --annotator and --input")
        completed = validate_annotations(read_json(args.input), data, args.annotator)
        print(json.dumps({"completed": completed, "assigned": len(annotation_rows(data, args.annotator)),
                          "gold_status": "Independent annotations; adjudication still required"}))


if __name__ == "__main__":
    main()
