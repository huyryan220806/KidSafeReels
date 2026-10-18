// Run a copy from runs/presentation-build with bundled node_modules linked there.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {FileBlob, PresentationFile} from '@oai/artifact-tool';

const root=process.cwd();
const skill=process.env.KIDSAFE_PRESENTATION_SKILL;
const python=process.env.KIDSAFE_ARTIFACT_PYTHON;
if(!skill||!python) throw new Error('Set the bundled presentation runtime paths.');
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const p=await PresentationFile.importPptx(await FileBlob.load(path.join(root,'output/presentation/KidSafeReels_ThuyetTrinh.pptx')));
const profile=JSON.parse(await fs.readFile(path.join(root,'reports/app_profile.json'),'utf8'));
const acceptance=JSON.parse(await fs.readFile(path.join(root,'reports/app_acceptance_tien_results.json'),'utf8'));
const tests=JSON.parse(await fs.readFile(path.join(root,'reports/app_test_summary.json'),'utf8'));
function edit(slideIndex,oldText,newText,position,fontSize){
 const candidates=p.slides.items[slideIndex].shapes.items.filter(s=>s.text.toString()===oldText);
 if(candidates.length!==1)throw new Error(`Expected one source shape: ${oldText}`);
 const shape=candidates[0];shape.text.replace(oldText,newText);
 if(position)shape.position=position;
 if(fontSize)shape.text.style={fontSize,typeface:'Arial',color:slideIndex===0?'#EAF4F1':'#183942'};
}
edit(0,'Bản tích hợp CPU • 18/10/2026',`Ứng dụng: ${profile.name}, ${profile.student_id}\nLớp ${profile.class}, GV: ${profile.lecturer}\nHọc máy: Huy, bản CPU ngày 18/10/2026`,{left:64,top:430,width:1150,height:112},24);
edit(0,'Violence và Toxic tiếng Việt\nPilot đang chờ nhãn người thật','Violence và Toxic tiếng Việt\nPilot đang chờ nhãn người thật',{left:64,top:565,width:1100,height:80},25);
edit(8,'117 tests pass\n13/13 ca HTTP pass',`${tests.passed} tests pass\n${acceptance.passed}/${acceptance.total} ca HTTP pass`,null,43);
edit(8,'5 clip train thật qua worker/model\nCó kiểm tra file rỗng, hỏng, dài\nKhông tiếng giữ Toxic = null','Máy Tiến, Intel Core i5-12450H\nVideo tổng hợp và giọng Việt FLEURS\nKhông chạy dự đoán trên pilot',null,27);
const cold=acceptance.cases.find(r=>r.cold_worker);
const speech=acceptance.cases.find(r=>r.case==='speech-warm-repeat');
const number=value=>value.toFixed(2).replace('.',',');
edit(8,'23,34 giây\nLần gọi đầu',`${number(cold.request_seconds)} giây\nLần gọi đầu`,null,32);
edit(8,'5,20 giây\nMedian 4 lần sau',`${number(speech.request_seconds)} giây\nMẫu tiếng Việt chạy lại`,null,29);
edit(8,'Độ trễ chỉ mô tả lần chạy này. Chưa nghiệm thu máy thứ hai.','Đã thử trên máy app. Đây là kiểm thử vận hành, chưa phải chất lượng ML.',null,23);
p.slides.items[8].speakerNotes.text='Nguồn: reports/app_acceptance_tien_results.json, reports/app_test_summary.json, docs/APP_ACCEPTANCE_TIEN.md. Số liệu đo thật trên máy Intel Core i5-12450H của Tiến. Fixture hình ảnh tổng hợp và FLEURS có giấy phép chỉ kiểm tra luồng model; không đánh giá độ chính xác. Không so trực tiếp latency với clip train 25 giây trên máy Huy.';
edit(10,'Thành viên app',profile.name);
edit(10,'Còn lại: nhãn pilot và metric chung, máy thứ hai, quay demo\nBáo cáo cần bổ sung thông tin sinh viên theo mẫu giảng viên','Còn lại: nhãn pilot, đối chiếu và metric chung\nHoàn thiện thông tin Huy và diễn tập bảo vệ',null,26);
const build=path.join(root,'runs/presentation-build/app-update');
await fs.mkdir(build,{recursive:true});
const candidate=path.join(build,'candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
// The two benchmark charts were not edited. Restore the original chart XML and
// embedded workbooks after import/export rather than inventing workbook snapshots.
const repair=spawnSync(python,[path.join(root,'scripts/restore_source_chart_workbooks.py'),path.join(root,'output/presentation/KidSafeReels_ThuyetTrinh.pptx'),candidate],{encoding:'utf8'});
if(repair.status!==0)throw new Error(repair.stdout+'\n'+repair.stderr);
console.log(repair.stdout);
const finalPath=path.join(root,'output/presentation/KidSafeReels_ThuyetTrinh_Tien.pptx');
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath,pythonExecutable:python,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--require-native-table-slide','3'],
 explicitTotalSlideCount:11,requiredNativeTableOwnerSlides:[3],requiredNativeChartOwnerSlides:[5,6],
 fontPolicy:{basis:'reference',families:['Arial'],referencePath:path.join(root,'output/presentation/KidSafeReels_ThuyetTrinh.pptx'),referenceSha256:createHash('sha256').update(await fs.readFile(path.join(root,'output/presentation/KidSafeReels_ThuyetTrinh.pptx'))).digest('hex')},
 verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
const final=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for (const [index,slide] of final.slides.items.entries()) {
 const png=await final.export({slide,format:'png',scale:1});
 await fs.writeFile(path.join(build,`slide-${index+1}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log(finalPath);
