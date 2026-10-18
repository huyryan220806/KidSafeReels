// Run a copy of this file inside runs/presentation-build with the bundled
// @oai/artifact-tool node_modules linked there. CWD must be the repository root.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const root = process.cwd();
const build = path.join(root, 'runs/presentation-build');
const skill = process.env.KIDSAFE_PRESENTATION_SKILL;
const python = process.env.KIDSAFE_ARTIFACT_PYTHON;
if(!skill || !python)throw new Error('Set KIDSAFE_PRESENTATION_SKILL and KIDSAFE_ARTIFACT_PYTHON to the installed Codex artifact runtime paths.');
const {finalizePresentation, applyPresentationChartFont} = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
const p = Presentation.create({slideSize:{width:1280,height:720}});
const C={ink:'#183942',teal:'#147D83',muted:'#557078',paper:'#F5F8F6',amber:'#AD6B25'};
function text(slide, value, x, y, w, h, size=28, bold=false, color=C.ink){
 const s=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 s.text=value;s.text.style={typeface:'Arial',fontSize:size,bold,color};return s;
}
function slide(title, source, notes=''){
 const s=p.slides.add();s.background.fill=C.paper;
 text(s,title,64,52,1152,115,44,true);
 text(s,'KidSafeReels     Học máy và ứng dụng',64,670,1050,25,18,false,C.muted);
 text(s,String(p.slides.items.length).padStart(2,'0'),1150,670,65,25,18,false,C.muted);
 s.speakerNotes.text=`${notes}\n\nNguồn: ${source}`;return s;
}
let s=slide('KidSafeReels','reports/TECHNICAL_REPORT.md','Mở đầu khoảng 20 giây. Giới thiệu bài toán sàng lọc video trước khi người lớn cho trẻ xem. Nhóm làm phạm vi hai nhãn và dùng CPU.');
s.background.fill=C.ink;
for(const [i,shape] of s.shapes.items.entries())shape.text.style={typeface:'Arial',fontSize:i===0?64:18,bold:i===0,color:'#EAF4F1'};
text(s,'Kiểm duyệt đa phương thức\nnội dung video ngắn',64,240,1130,160,56,true,'#FFFFFF');
text(s,'Bản tích hợp CPU • 18/10/2026',64,475,1100,55,28,false,'#EAF4F1');
text(s,'Violence và Toxic tiếng Việt\nPilot đang chờ nhãn người thật',64,555,1100,85,27,false,'#EAF4F1');

s=slide('Công cụ sàng lọc cho phụ huynh và giáo viên','docs/APP_GUIDE.md; docs/screenshots/app-redesign-desktop.jpg (ảnh giao diện từ app owner)', 'Trình bày khoảng 25 giây. Người lớn tải file video lên, xem kết quả và tự quyết định. Chỉ ra phạm vi hai nhãn, không có tích hợp chặn feed. Ảnh là giao diện upload, không phải kết quả phân tích model.');
text(s,'Tải video ≤60 giây\nMP4, MOV hoặc WebM\nTối đa 50 MiB',64,205,420,160,30,true);
text(s,'Xem lại evidence\nTải báo cáo TXT / JSON\nChưa hỗ trợ Sexual / Dangerous',64,408,440,160,27);
const img=await fs.readFile(path.join(root,'docs/screenshots/app-redesign-desktop.jpg'));
s.images.add({blob:img.buffer.slice(img.byteOffset,img.byteOffset+img.byteLength),contentType:'image/jpeg',alt:'Giao diện upload KidSafeReels',fit:'contain',position:{left:545,top:178,width:670,height:455}});

s=slide('Dữ liệu và tập test giữ riêng','reports/visual_baseline_metrics.json; reports/toxic_baseline_metrics.json; data/pilot/manifest.json','Khoảng 35 giây. Nhấn mạnh ba nguồn có đơn vị khác nhau. RLVS thiếu ID nguồn gốc, nhóm chỉ kiểm soát trùng bằng hash và heuristic. Pilot có nhóm nguồn rõ nhưng chưa có nhãn và chưa đại diện Shorts.');
const values=[['Nguồn','Train','Validation','Test','Đơn vị'],['RLVS','1.157','386','386','Video'],['ViCTSD','6.965','1.975','988','Bình luận'],['Pilot v1','17','4','4','Đoạn 25 giây']];
const t=s.tables.add({rows:4,columns:5,left:64,top:200,width:1152,height:280,columnWidths:[280,180,220,180,292],values});
t.borders.assign({fill:'#CCDCD8',width:1,style:'solid'});
for(let r=0;r<4;r++)for(let c=0;c<5;c++){
 const cell=t.getCell(r,c);cell.fill=r===0?'#DFEFED':'#FFFFFF';cell.text.style={typeface:'Arial',fontSize:27,color:C.ink,bold:r===0};
}
text(s,'Pilot: 25 đoạn từ 8 video gốc, 5 nhóm nguồn\nChia theo nhóm trước khi lấy frame',64,530,1135,90,28);

s=slide('Hai nhánh phân tích trên CPU','ml/train_visual.py; ml/toxic_baseline.py; ml/inference.py; models/artifacts.json','Khoảng 40 giây. Nhóm huấn luyện hai classifier Logistic Regression. ResNet18 và Whisper là pretrained. Hai nhánh dự đoán các nhãn khác nhau, nên phép OR là quy tắc quyết định thay vì learned fusion. Ngưỡng chọn trên validation benchmark.');
text(s,'HÌNH ẢNH',64,195,530,45,28,true,C.teal);
text(s,'Tối đa 8 frame mỗi video\nResNet18 đóng băng\nMean + max pooling, 1.024 chiều\nLogistic Regression\nNgưỡng Violence: 0,43',64,262,540,270,29);
text(s,'LỜI NÓI',680,195,530,45,28,true,C.teal);
text(s,'Whisper base, CPU int8\nTranscript tiếng Việt\nTF-IDF ký tự 3–5 gram\nLogistic Regression\nNgưỡng Toxic: 0,59',680,262,535,270,29);
text(s,'Một nhánh vượt ngưỡng: flagged. Các trường hợp còn lại: review.',64,580,1152,62,27,true);

const visual=JSON.parse(await fs.readFile(path.join(root,'reports/visual_baseline_metrics.json'),'utf8'));
const toxic=JSON.parse(await fs.readFile(path.join(root,'reports/toxic_baseline_metrics.json'),'utf8'));
function resultChart(s,values,color){
 values=values.map(value=>Number(value.toFixed(6))); // Editable workbook stores six decimals; labels show 0.1%.
 const ch=s.charts.add('bar',{position:{left:64,top:210,width:740,height:380},categories:['Precision','Recall','F1'],series:[{name:'Lớp dương',values,fill:color}],hasLegend:false,
 barOptions:{direction:'column',grouping:'clustered',gapWidth:130},
 xAxis:{textStyle:{fontSize:24,fill:C.ink}},yAxis:{min:0,max:1,majorUnit:.2,numberFormatCode:'0%',textStyle:{fontSize:22,fill:C.muted},majorGridlines:{fill:'#D5E2DF',width:1}},
 dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.0%',numberFormatSourceLinked:false,textStyle:{fontSize:25}},chartFill:'none',plotAreaFill:'none'});
 applyPresentationChartFont(ch,{fontFamily:'Arial'});
}
const v=visual.test.classification_report.violence;
s=slide('Violence trên RLVS','reports/visual_baseline_metrics.json; reports/VISUAL_BASELINE.md','Khoảng 30 giây. Nêu F1 0,9598 trên 386 video test, gồm 198 dương tính. Có 7 ca bỏ sót và 9 cảnh báo nhầm. Kết quả tốt trong benchmark này, chưa suy ra khả năng bảo vệ trẻ trên Shorts.');
resultChart(s,[v.precision,v.recall,v['f1-score']],C.teal);
text(s,'386 video test\n198 ca Violence',855,222,355,100,31,true);
text(s,'Bỏ sót: 7\nCảnh báo nhầm: 9\nNgưỡng: 0,43',855,365,350,150,28);
text(s,'Benchmark RLVS. Chưa phải chất lượng kiểm duyệt video trẻ em.',64,605,1150,45,24,false,C.muted);

const tr=toxic.test.classification_report;
const tx=tr.toxic ?? tr.Toxic ?? tr['1'];
if(!tx)throw new Error('Missing Toxic class metrics');
s=slide('Toxic còn bỏ sót nhiều trường hợp','reports/toxic_baseline_metrics.json; reports/TOXIC_BASELINE.md','Khoảng 35 giây. F1 dương 0,4492, recall 0,4862. Accuracy 0,8684 thấp hơn luôn dự đoán âm 0,8897. Macro-F1 tốt hơn baseline nhưng cần trình bày rõ yếu điểm. Đây là bình luận viết, chưa phải transcript.');
resultChart(s,[tx.precision,tx.recall,tx['f1-score']],C.amber);
text(s,'988 bình luận test\n109 ca Toxic',855,222,355,100,31,true);
text(s,'Bỏ sót: 56\nCảnh báo nhầm: 74\nNgưỡng: 0,59',855,365,350,150,28);
text(s,'ViCTSD là bình luận viết. Chưa đo F1 trên transcript video.',64,605,1150,45,24,false,C.muted);

s=slide('Pilot tiếng Việt đã sẵn sàng để gán nhãn','reports/pilot_data_audit.json; docs/ANNOTATION_GUIDE.md','Khoảng 25 giây. Có 25 clip giấy phép CC BY 3.0, nhưng chỉ từ 8 video gốc. Huy cần gán 25, app owner gán chéo 6. Chưa có phiếu nộp. Ngôn ngữ ASR và tiêu đề không được dùng làm gold.');
text(s,'625 frame\n58 đoạn ASR',64,205,570,145,48,true,C.teal);
text(s,'25 đoạn, mỗi đoạn 25 giây\n8 video gốc, 5 nhóm\nNguồn CC BY 3.0 có ghi công',64,393,565,170,29);
text(s,'Nhãn người thật',700,205,510,60,32,true);
text(s,'Huy: 0/25 phiếu hoàn tất\nThành viên app: 0/6\nChưa tính metric pilot',700,300,510,185,30);
text(s,'Tập thiên về nội dung thông tin. Chưa xác nhận đủ ca nguy hại.',64,595,1150,55,26,false,C.muted);

s=slide('Quy trình đánh giá sau khi có nhãn','docs/EVALUATION_GUIDE.md; ml/pilot_evaluation.py','Khoảng 30 giây. So hai phiếu độc lập, ghi lý do phân xử rồi đóng gold. Báo coverage và support cùng F1. Cùng cohort đầy đủ nhãn và score để so visual-only, text-only, OR với đích ít nhất một rủi ro P0. Nếu test không có dương tính, không suy ra recall/F1 dương.');
text(s,'01   So phiếu và phân xử bất đồng\n02   Khóa gold, artifact, media và code\n03   Đánh giá một lần trên test giữ riêng',64,205,1150,205,34,true);
text(s,'Báo support, coverage và số abstain\nSo visual-only / text-only / OR trên cùng cohort\nScore null không chuyển thành nhãn âm',64,465,1150,150,29);

s=slide('Bản tích hợp chạy với model thật','reports/release_smoke_results.json; docs/RELEASE_HANDOFF.md; pytest toàn repo ngày 18/10/2026','Khoảng 30 giây. 117 test tự động và 13 ca HTTP/model thật. Năm clip train thật chỉ kiểm tra kỹ thuật, không tính độ đúng nội dung. Độ trễ gồm 1 lần đầu và 4 lần sau trên cùng nhóm nguồn, chưa đại diện. Máy Ryzen 5 7535HS, Windows, Python 3.12, CPU.');
text(s,'117 tests pass\n13/13 ca HTTP pass',64,200,620,145,47,true,C.teal);
text(s,'5 clip train thật qua worker/model\nCó kiểm tra file rỗng, hỏng, dài\nKhông tiếng giữ Toxic = null',64,400,670,170,29);
text(s,'23,34 giây\nLần gọi đầu',850,205,360,110,34,true);
text(s,'5,20 giây\nMedian 4 lần sau',850,385,360,110,34,true);
text(s,'Độ trễ chỉ mô tả lần chạy này. Chưa nghiệm thu máy thứ hai.',64,605,1150,45,24,false,C.muted);

s=slide('Sai số ASR có thể dẫn đến cảnh báo nhầm','reports/VIDEO_SMOKE.md; reports/video_smoke_results.json','Khoảng 30 giây. Một mẫu FLEURS trung tính bị ASR chép sai. Điểm đoạn ASR 0,6333, văn bản chuẩn toàn câu 0,1525. Khác cả transcript lẫn phân đoạn nên đây chỉ là ví dụ, chưa định lượng riêng tác động ASR và chưa có WER đại diện.');
text(s,'0,6333',64,210,540,100,68,true,C.amber);
text(s,'Điểm Toxic của đoạn ASR\nVượt ngưỡng 0,59',64,345,540,110,31);
text(s,'0,1525',700,210,510,100,68,true,C.teal);
text(s,'Điểm của văn bản chuẩn toàn câu\nCùng mẫu giọng nói FLEURS',700,345,510,115,29);
text(s,'Một ca lỗi. Chưa tách riêng tác động ASR và phân đoạn.\nVisual lấy frame thưa cũng có thể bỏ sót hành động ngắn.',64,535,1150,95,27);

s=slide('Đóng góp và công việc trước nghiệm thu','docs/PLAN.md; docs/PROGRESS.md; reports/TECHNICAL_REPORT.md','Kết thúc khoảng 30 giây. Huy trình bày dữ liệu, baseline, pipeline và thực nghiệm. App owner trình bày upload, worker và UI. Còn gán nhãn, chạy đánh giá, máy thứ hai và demo. Kết quả hiện tại là bản tích hợp kỹ thuật, chưa tuyên bố an toàn tuyệt đối.');
text(s,'Huy',64,200,540,50,32,true,C.teal);
text(s,'Dữ liệu, chia tập và hai baseline\nASR, kết hợp quyết định\nĐánh giá, báo cáo và tích hợp',64,280,565,160,29);
text(s,'Thành viên app',700,200,510,50,32,true,C.teal);
text(s,'Upload, preview và evidence\nAPI worker, timeout và validation\nGiao diện và hướng dẫn sử dụng',700,280,510,160,29);
text(s,'Còn lại: nhãn pilot và metric chung, máy thứ hai, quay demo\nBáo cáo cần bổ sung thông tin sinh viên theo mẫu giảng viên',64,535,1150,100,27,true);

await fs.mkdir(build,{recursive:true});
const candidate=path.join(build,'candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:path.join(root,'output/presentation/KidSafeReels_ThuyetTrinh.pptx'),pythonExecutable:python,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','3'],
 requiredNativeTableOwnerSlides:[3],requiredNativeChartOwnerSlides:[5,6],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
const finalDeck=await PresentationFile.importPptx(await FileBlob.load(path.join(root,'output/presentation/KidSafeReels_ThuyetTrinh.pptx')));
for(const [i,slide]of finalDeck.slides.items.entries()){
 const png=await finalDeck.export({slide,format:'png',scale:1});
 await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log(`Created ${p.slides.items.length} slides`);
