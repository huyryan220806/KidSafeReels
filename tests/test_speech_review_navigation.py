"""Exercise batching and JSON round trips without changing earlier annotations."""
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from ml.speech_intake import ROOT, blank_row


def test_batched_review_preserves_edits_and_exports_every_clip(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the annotation JavaScript behavioral check")
    rows = [blank_row(f"review-{i:03d}", "huy") for i in range(45)]
    rows[0].update(watched_full=True, language="vi", toxic_language=0, notes="Earlier human label")
    config = dict(dataset="synthetic-test", batch_size=20,
        initial=dict(manifest_sha256="fixture", annotator="huy", rows=rows),
        clips={r["video_id"]: dict(duration=20, src="synthetic.mp4") for r in rows})
    html = (ROOT / "ml/templates/speech_review.html").read_text(encoding="utf-8")
    script = re.search(r"<script>(.*?)</script>", html, re.S)[1].replace("__REVIEW_CONFIG__", json.dumps(config))
    source = tmp_path / "review.js"
    source.write_text(script, encoding="utf-8")
    harness = tmp_path / "check.cjs"
    harness.write_text(r'''
const fs=require("fs"),vm=require("vm"),assert=require("assert/strict");
class Element {
 constructor(){this.listeners={};this.children=[];this.value="";this.checked=false;}
 addEventListener(event,fn){this.listeners[event]=fn;}
 append(...items){this.children.push(...items);}
 replaceChildren(...items){this.children=items;}
 setAttribute(){}
 click(){if(this.download)download=this;}
 fire(event,data={target:this}){return this.listeners[event](data);}
}
const elements=new Map(),saved=new Map(),blobs=new Map();let download;
const el=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
const context={document:{getElementById:el,createElement:()=>new Element()},structuredClone,Blob,
 localStorage:{getItem:key=>saved.get(key),setItem:(key,value)=>saved.set(key,value)},
 URL:{createObjectURL:blob=>{const key="blob:"+blobs.size;blobs.set(key,blob);return key;},revokeObjectURL(){}},
 setTimeout(){}};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[2],"utf8"),context);
(async()=>{
 assert.equal(el("batch-controls").hidden,false);
 assert.equal(el("batch").children.length,3);
 assert.equal(el("clip").value,"review-001");
 el("watched").checked=true;el("language").value="vi";el("label").value="0";
 el("notes").value="Saved while switching batch";
 el("batch").value="2";el("batch").fire("change");
 assert.equal(el("clip").value,"review-040");
 assert.match(el("batch-progress").textContent,/0 \/ 5/);
 el("clip").value="review-000";el("clip").fire("change");
 assert.equal(el("batch").value,"0");
 el("next-unlabelled").fire("click");
 assert.equal(el("clip").value,"review-002");
 el("export").fire("click");
 assert.equal(download.download,"annotations-huy-synthetic-test.json");
 const exported=JSON.parse(await blobs.get(download.href).text());
 assert.equal(exported.rows.length,45);
 assert.equal(exported.rows[0].notes,"Earlier human label");
 assert.equal(exported.rows[1].notes,"Saved while switching batch");
 assert.equal(exported.rows[1].toxic_language,0);
 assert.equal(exported.rows[40].toxic_language,null);
 exported.rows.reverse();
 await el("import").fire("change",{target:{files:[{text:async()=>JSON.stringify(exported)}],value:"file"}});
 el("batch").value="2";el("batch").fire("change");
 assert.equal(el("clip").value,"review-040");
 el("export").fire("click");
 const restored=JSON.parse(await blobs.get(download.href).text());
 assert.equal(restored.rows[1].notes,"Saved while switching batch");
 console.log("PASS: batching, earlier labels, edits, full export, reordered import");
})().catch(error=>{console.error(error);process.exitCode=1;});
''', encoding="utf-8")
    result = subprocess.run([node, str(harness), str(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
