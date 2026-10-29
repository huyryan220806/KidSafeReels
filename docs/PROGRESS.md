# Tiến độ KidSafeReels

## Toxic ưu tiên giảm báo nhầm, giữ Violence v3 — đang thực hiện 29/10/2026

- Theo Huy chốt: giữ nguyên Violence v3/ngưỡng0,41; cải thiện riêng Toxic. Đã khóa protocol `toxic-precision-2026-10-29`, tái dùng split142/28/30; không mở test cũ cho đợt này.
- Đã chuẩn bị phiếu nghe duyệt10 clip (4train+6validation,10nhóm nguồn,175giây),5nghi FP+5nghi FN theo nhãn nguồn. **0/10 nhãn người**, không tự sửa nhãn theo dự đoán. Phiếu local cổng8767; có xuất/nạp JSON và kiểm bằng chứng. [Hướng dẫn](TOXIC_PRECISION_2026-10-29.md).
- Đang tạo lại ASR CPU INT8 trên170clip train/validation, cache từng clip. Kế tiếp fine-tune MIL5epoch tăng loss âm tính lên2, chọn theo ràng buộc FP/Recall validation, kiểm lại CPU và lưu ứng viên riêng. Chưa có kết quả train mới; app vẫn v3+hybrid.

## Hoàn tất train speech200 + AIRTLab100 — 29/10/2026

- Theo yêu cầu Huy, đã train thực sự và lưu checkpoint mới trong run `local-adaptation-2026-10-29`. Speech100 Toxic +100 bình thường dùng **nhãn yếu cấp clip từ người cung cấp**, chưa đổi phiếu `human_label` trống thành gold. Gom nguồn/SHA/dHash, khóa142/28/30;11 clip liên quan ca đã xem vào train. ASR đủ điều kiện138/28/29, không gán thiếu ASR thành0.
- PhoBERT v3 fine-tune hai encoder block cuối + classifier,5epoch MIL max logit cấp clip; chọn epoch5/ngưỡng0,32 bằng validation. Có history loss/accuracy/F1 thật và metrics train/validation/test GPU. Kiểm CPU giữ riêng30clip: **hybrid F1 60%→72,22%, FP6→8, FN6→2**; model-only loss trên29clip có ASR1,7724→0,7604. Nhãn yếu và ASR còn sai nên chưa xác nhận giảm báo nhầm.
- Violence fit4 cấu hình R3D frozen-feature +LR trên **RLVS1157 + AIRTLab100**. AIRTLab cả bối cảnh vào train, không còn là externaltest của ứng viên này. Chọn C0,01/ngưỡng0,64 trên validation nhưng không đạt điều kiện FP/Recall baseline. **RLVS testF1 97,97%→95,83%, FN5→14**. AIRTLab accuracy91% là số đo train, không phải test.
- **Giữ app CPUv3 + hybrid**, lưu hai ứng viên riêng; không chỉnh ngưỡng theo kết quả test hoặc tự nâng model yếu hơn. Checkpoint: `runs/local-adaptation-2026-10-29/toxic/selected/`, `violence/selected.joblib` + backbone; raw/transcript/weights không Git.
- **244 tests pass**, `pip check` môi trường train pass;17 artifact đang dùng và8tệpToxic/2tệpViolence mới đúng checksum. Kiểm CPU Violence10 mẫu train gần ngưỡng khớp10/10, lệchmax4,17e-7. Có CSV/PNG/SVG/loss/history và10 bất đồng CPU cần nghe duyệt trong `cpu_error_review.csv` local.
- [Báo cáo và CSV/đồ thị](../reports/LOCAL_ADAPTATION_2026-10-29.md), [cách tái tạo/resume](LOCAL_ADAPTATION_2026-10-29.md). Điểm tiếp tục local `runs/local-adaptation-2026-10-29/CONTINUE.md`. Không chạy lại holdout đã xem để dò ngưỡng; việc còn lại là nhãn thật/bằng chứng dev, dữ liệu Violence ngoài bối cảnh và kiểm ứng viên tiếp theo theo protocol mới.

## Triển khai Toxic hybrid theo hướng Huy chốt — 29/10/2026

- Whisper → transcript tiếng Việt → PhoBERT + bộ luật từ/cụm từ đã tích hợp vào pipeline và app. Bộ khởi đầu có 14 luật cảnh báo, 4 viết tắt chỉ review; khớp Unicode giữ dấu/ranh giới từ, có điều kiện chất lượng ASR cho cảnh báo mới từ luật. Không thêm hai nhãn ngoài phạm vi hoặc thay trọng số/ngưỡng CPU v3.
- Điểm Toxic/evidence vẫn là điểm PhoBERT thật; app/TXT ghi rõ nguồn cảnh báo khi bộ luật khớp nhưng điểm thấp. Schema 1.0 giữ nguyên. Kết quả hybrid có hậu tố `+toxic-hybrid-v1-56b3a177c7b2`; launcher mặc định hybrid, có `-ToxicMode model_only` sau khi dừng/mở lại để so baseline.
- Ba ca phát triển được replay rồi chạy HTTP: **048 flagged → flagged; 019 review → review; 022 review → flagged**. Score max giữ 0,901374 / 0,002520 / 0,022752. Không tính 2/3 ca đã xem thành Accuracy; 11 clip liên quan vẫn bị loại khỏi test độc lập tương lai.
- Khóa lexicon/protocol trước ablation ViCTSD cũ, tái dùng cache PhoBERT thật có kiểm ID/nhãn/ngưỡng/checksum. Hybrid test **Accuracy 89,57%, F1 53,39%, FP 53/FN 50**, bằng model-only; rule-only chỉ bắt 1/109 Toxic. Validation cũng không tăng. Có CSV/JSON/PNG/SVG; không có xác suất/loss gộp hoặc lượt train mới. [Báo cáo](../reports/TOXIC_HYBRID_2026-10-29.md).
- Kiểm chứng: **238 tests pass**, **7/7 HTTP** (3 ca lỗi, im lặng/âm đơn tần, 2 pilot train), 17 artifact đúng checksum. Web thật clip022: cảnh báo/nguồn luật/điểm 0,023/mốc phát 10 giây; TXT và JSON tải qua trình duyệt được kiểm lại. Ảnh/transcript/downloads lưu private trong run, không Git.
- [Thiết kế/cách mở/baseline](TOXIC_HYBRID.md), [bàn giao cập nhật cho app](TOXIC_HYBRID_HANDOFF.md). Điểm tiếp tục local: `runs/toxic-hybrid-2026-10-29/CONTINUE.md`.
- **Còn:** nghe đối chiếu/sửa transcript chuẩn ca019 và dev; duyệt nhãn speech200/nguồn gần trùng, khóa split theo nguồn, train/evaluate nếu đủ dữ liệu. Chưa chứng minh giảm FP hoặc cải thiện toàn pipeline trên test video đích. Không mở rộng luật theo test vừa xem; không tự điền nhãn người thật.

## Sửa lọc nhầm lời nói sau chẩn đoán ca lỗi — 28/10/2026

- Huy chỉ rõ `vi-speech-048`, `019`, `022`. Đã tái hiện bằng local và API model thật. `048`/`019`: decoder trả 6/4 đoạn nhưng gate `no_speech_prob < 0.6` lọc lại làm mất tất cả; `022` đã tới classifier nhưng score thấp.
- Đã bỏ gate thứ hai, dùng segment không rỗng do decoder/VAD chấp nhận, giữ kiểm tiếng Việt. Trọng số/ngưỡng/schema không đổi; phiên bản kết quả thêm `+asr-native-filter-v2`. Nạp lại worker hiện hành để app dùng code sửa.
- Qua HTTP: **048 null → 0,901374 / flagged; 019 null → 0,002520 / review; 022 giữ 0,022752 / review**. Một lỗi nhận diện đã khắc phục, hai ca vẫn bỏ sót theo phản hồi người dùng; chưa có kết quả train/test mới hoặc bằng chứng giảm FP chung.
- Regression tái hiện trước sửa (1 failed/4 passed), sau sửa **63 tests pass** cho pipeline/config/API/web. Ba ca thật qua HTTP, thêm kiểm im lặng/âm đơn tần và hai pilot train; chi tiết [báo cáo sửa lỗi](../reports/ASR_FILTER_FIX_2026-10-28.md).
- Trước suy luận đã ghi exposure riêng và giữ **11 clip liên quan** khỏi test độc lập tương lai. Manifest/phiếu gốc giữ nguyên; không tự gán nhãn hoặc dùng ASR làm transcript chuẩn. Raw chẩn đoán ở `runs/speech-toxic-diagnostic-2026-10-28/`.
- Bước tiếp: sửa transcript/bằng chứng ba ca, hoàn tất nhãn speech200 và nhóm nguồn, tạo split giữ riêng test rồi train/evaluate theo protocol mới. Các thí nghiệm ASR cũ giữ bộ lọc cũ để tái tạo; không dùng số đo cũ cho hành vi mới.

## Tiếp nhận đủ hai nhóm lời nói speech200 — 28/10/2026

- Huy bổ sung 100 clip bình thường tại `data/Video_Train/Toxic/KidSafeReels-100-Clip-LoiNoi/Non_Toxic`. Bộ hiện tại: **200 clip, 132 video nguồn, 65 phút 41 giây**; 100 Toxic từ 74 nguồn và 100 bình thường từ 58 nguồn theo mô tả người cung cấp. Metadata cả hai bộ có nhãn chuẩn trống; giữ `user_reported_clip_label` riêng, không tự gọi là 200 nhãn người đã duyệt.
- **200/200 hash/kích thước/thời lượng và giải mã toàn bộ hình + tiếng pass**. Không trùng SHA trong speech200, không trùng ID nguồn giữa hai nhóm, không chồng khoảng cắt trong một nguồn. dHash 1 fps phát hiện 28 cặp cần xem, trong đó **7 cặp khác ID nguồn**, 0 cặp giữa hai lớp gợi ý; đây là gợi ý gần trùng, không phải xác nhận bản đăng lại hoặc bằng chứng tuyệt đối không trùng.
- 50 file Violence trùng SHA hoàn toàn với AIRTLab đã dùng đánh giá ngoài; không tính thành dữ liệu mới, không đưa vào train để rồi vẫn dùng test AIRTLab cũ.
- Đã tạo manifest mới `speech200-intake-v2`, giữ nguyên manifest speech100 trước đó. Phiếu Huy **200**, app **40** (20 mỗi nhóm mô tả, 40 ID nguồn), thứ tự trộn/mã video trung tính. HTML/video không lộ nhãn nguồn, ASR hoặc dự đoán. Có lưu bản nháp, xuất/nạp JSON, bằng chứng có mốc, kiểm cấu trúc và ZIP riêng cho app. Chưa gửi file/tin nhắn thay Huy.
- **31 tests pass** cho intake/nhãn/đường dẫn/gói ẩn nhãn; kiểm trình duyệt thật: video có metadata và không lỗi, đổi clip, khôi phục bản nháp, chặn nhãn thiếu xác nhận/bằng chứng hoặc nhãn 0 có bằng chứng Toxic, xuất/nạp JSON trống. Phiếu thật vẫn **0/200**, app **0/40**; dữ liệu kiểm thử giao diện tách riêng, không tính thành nhãn thật. Không chạy lại suite model/app vì không đổi pipeline.
- Bộ local: `data/private/speech200-2026-10-28/`, gói `speech200-app-review.zip`; raw/nguồn/phiếu không lên Git. [Hướng dẫn mở và nhãn](SPEECH200_ANNOTATION.md), [báo cáo số liệu](../reports/SPEECH200_INTAKE_2026-10-28.md). Điểm tiếp tục local: `runs/speech200-2026-10-28/CONTINUE.md`.
- **Chưa train hoặc có Accuracy/Loss mới trên speech200.** Cần nghe duyệt nhãn/bằng chứng, xử lý nguồn gần trùng rồi khóa split theo nguồn trước ASR/đặc trưng model. Không gán mọi câu của clip Toxic thành Toxic. Model app vẫn CPU v3; mục tiêu tiếp theo là đo giảm false positive trên lời nói bình thường bằng test giữ riêng.

## Hoàn tất CPU v3 bằng PhoBERT và ResNet3D — 28/10/2026

- Huy yêu cầu tiếp tục nâng chất lượng. Đã hoàn tất môi trường CUDA riêng `.venv-train`; app vẫn CPU, thêm Transformers/PyVi và khóa phụ thuộc tương thích Whisper.
- PhoBERT đã train đủ 6 epoch, chọn epoch 4/ngưỡng 0,78 trên validation. Test Toxic F1 44,92% → 53,39%, Accuracy 86,84% → 89,57%; FP 74 → 53, FN 56 → 50. Có đường học/CSV thật; loss tăng 0,4193 → 0,4641 và có quá khớp, ghi rõ trong báo cáo.
- Violence: đã chọn R3D-18 Kinetics400, 5 cửa sổ × 16 frame/video ở 15 fps, LR C=0,01/ngưỡng 0,41 trên validation. Test RLVS Accuracy 95,85% → 97,93%, F1 95,98% → 97,97%; FP 9 → 3, FN 7 → 5. Test không dùng để điều chỉnh ngưỡng.
- Huy cung cấp Drive có 100 video AIRTLab (50/50 theo nhãn nguồn), xác nhận nhãn dự án chưa rõ. Dành cả bối cảnh AIRTLab làm đánh giá bổ sung theo nhãn nguồn sau khi khóa model, không chia ngẫu nhiên vào train/validation. Không coi là nhãn độc lập của nhóm hoặc video tiếng Việt tự nhiên.
- AIRTLab đã xác minh 100/100 clip với Git blob tác giả; không phát hiện trùng qua SHA/dHash. Đánh giá nhãn nguồn: Accuracy 49% → 53%, FP 43 → 39/50, FN giữ 8/50; khác biệt bối cảnh vẫn rất lớn. Không dùng kết quả này để chỉnh ngưỡng hoặc trộn vào train.
- **CPU v3 đã kích hoạt**, giữ Whisper small beam5 và hai nhãn còn lại null. ZIP 17 tệp/1,01 GiB, giữ ZIP/manifest v2 để khôi phục. [Cách mở/bàn giao v3](MODEL_UPDATE_2026-10-28.md), [báo cáo](../reports/MODEL_IMPROVEMENT_2026-10-28.md), [quy trình tái tạo](MODEL_IMPROVEMENT_2026-10-28.md).
- Kiểm chứng: **152 tests pass**, pip check, cài ZIP vào thư mục riêng và checksum pass; **13/13 HTTP/model**, **8 UI AppTest pass + 1 chưa chạy** vì speech fixture không có evidence. CPU Toxic median ~0,059 giây/bình luận; Violence ~2,56 giây/clip ở mẫu nhỏ, chưa gồm ASR/API.
- Sai khác score CPU/GPU ban đầu do cuDNN TF32: giữ báo cáo kiểm tra ban đầu failed, có diagnostic xác nhận frame giống hệt. So GPU FP32 trên 386 test và CPU trên 20 mẫu gần ngưỡng không đổi quyết định; CPU/FP32 lệch tối đa 8,23e-7. Không chỉnh model/ngưỡng để vượt kiểm tra.
- Còn nhãn pilot/ca lỗi tiếng Việt, kiểm v3 trên máy Tiến, cập nhật PDF/PPTX và quay demo. Điểm tiếp tục: `runs/model-improvement-2026-10-28/CONTINUE.md`; không huấn luyện lại run-id này hoặc dùng AIRTLab đã xem như test mới.

## Hoàn tất đợt thực nghiệm và cập nhật CPU v2 — 20/10/2026

- Huy yêu cầu thực hiện kế hoạch và xác nhận chưa lưu các video lỗi; tiếp tục với dữ liệu hiện có.
- Đã thêm công cụ chọn cấu hình trên validation, khóa artifact rồi đánh giá test; xuất CSV/PNG/SVG, Accuracy/Loss và các metric theo lớp. Hướng dẫn: [MODEL_EXPERIMENTS.md](MODEL_EXPERIMENTS.md).
- Toxic: đã train 6 ứng viên. Ứng viên chọn trên validation giảm FP test 74 → 57 nhưng TP cũng giảm 53 → 48; F1 0,4492 → 0,4486. Quyết định giữ classifier v1 vì recall ứng viên giảm.
- ASR: đã so 3 cấu hình trên 16 câu dev, chốt small/beam5 rồi đánh giá 16 câu test FLEURS riêng. WER test 0,3836 → 0,1983; CER 0,2240 → 0,1070. Có CSV, hình và audit ảnh hưởng ASR lên đầu ra Toxic; chưa phải accuracy moderation.
- Violence: đã hoàn tất 9 ứng viên và đánh giá test sau khi tiếp tục từ cache. Candidate-04 24 frame tốt hơn nhẹ trên validation nhưng test F1 giảm 0,9598 → 0,9474; FP 9 → 12, FN 7 → 9. Giữ Violence v1, không chỉnh lại dựa trên test.
- Đã xuất CSV tổng hợp, Accuracy/Loss và metric từng lớp cho train/validation/test, confusion matrix, PR/ROC, ngưỡng validation, learning curve theo số mẫu; PNG/SVG. Review lỗi private: Toxic 749 hàng, Violence 70 hàng; chưa tự điền nhãn/nhận xét người.
- **CPU v2 đã kích hoạt:** hai classifier v1 + Whisper small beam5. Gói `models/checkpoints/bundle/kidsafe-cpu-v2.zip`, 9 tệp có checksum, giữ ZIP v1 để khôi phục. [Báo cáo đầy đủ](../reports/MODEL_IMPROVEMENT_2026-10-20.md), [cài v2/mở app/rollback](MODEL_UPDATE_2026-10-20.md).
- Full suite **146 tests pass**, `pip check` pass; **13/13 HTTP/model**, **8 UI AppTest pass + 1 chưa chạy** do fixture không có evidence. Cài ZIP thật vào thư mục riêng và kiểm hash pass. API/web khởi động lại với v2 trên máy Huy; đây không phải nghiệm thu trình duyệt thủ công hoặc máy Tiến.
- Còn nhãn pilot, ca lỗi thực tế chưa lưu, kiểm v2 trên máy Tiến, cập nhật PDF/PPTX cuối và quay demo. Không công bố độ chính xác toàn app từ benchmark. Điểm tiếp tục local: `runs/model-improvement-2026-10-20/CONTINUE.md`.

## Ưu tiên tiếp theo theo phản hồi Huy — 19/10/2026

**Đã ghi nhận, chưa chạy train/đánh giá mới; tiếp tục khi Huy quay lại sau khi quota hồi.** Kế hoạch chi tiết và checklist: [MODEL_IMPROVEMENT_PLAN.md](MODEL_IMPROVEMENT_PLAN.md).

- Toxic đang cảnh báo nhầm video nói chuyện bình thường; Huy thấy transcript sai. Cần tách lỗi ASR, classifier và cách gộp đoạn để cải thiện, chưa chốt nguyên nhân cho từng clip.
- Violence đang bỏ sót video bạo lực rõ theo phản hồi dùng app; cần kiểm tra lấy mẫu frame, recall và khác biệt dữ liệu thực tế so với RLVS.
- Chỉ tập trung Toxic + Violence; Sexual/Dangerous hoãn khỏi đợt này, tiếp tục `null`/chưa đánh giá.
- Yêu cầu đầu ra: kết quả train/validation/test, Accuracy, Loss, Precision/Recall/F1, confusion matrix, đồ thị và CSV; so sánh baseline/bản cải thiện và ghi lỗi điển hình.
- Baseline chưa lưu lịch sử loss theo epoch; phải đo/log thật khi bổ sung, không dựng đường học. Giữ test khỏi vòng chọn model/ngưỡng. Nhãn pilot vẫn cần người thực hiện trước khi báo metric toàn pipeline.
- Điểm bắt đầu lần tới: kiểm tra artifact/cache/split, bổ sung báo cáo metric/CSV/đồ thị, rồi làm các thí nghiệm cải thiện. Chưa có clip lỗi mới của Huy được lưu thành tập dữ liệu.

## Sửa mở lại app khi cổng đã được dùng — 19/10/2026

- Ảnh lỗi của Huy do phiên kiểm thử trước vẫn chạy ở 8000/8501; health xác nhận API backend `model` và web đều hoạt động.
- Launcher nhận diện OpenAPI KidSafeReels, backend yêu cầu và web health; nếu khớp thì dùng phiên hiện có và trả thành công. Không tạo server trùng, không tự dừng tiến trình chiếm cổng.
- `Chay-KidSafeReels.cmd` truyền `-OpenBrowser`: mở trình duyệt khi khởi động xong hoặc mở lại phiên đang chạy. Khi cổng thuộc dịch vụ khác/sai backend/chưa sẵn sàng, in hướng dẫn thay vì thêm stack trace PowerShell cho lỗi cổng.
- **12 tests launcher pass (8,44 giây)**: mở lại, sai backend/dịch vụ, health lỗi, trình duyệt không mở được, khởi động mới chờ readiness và dọn web. Không chạy lại toàn bộ suite 118 tests của lần merge vì phần API/model/UI không thay đổi.
- Chạy thật `scripts/run-model.ps1` cả có/không có `-OpenBrowser`: exit 0, 8 artifact đúng checksum; PID API/web giữ nguyên. Cập nhật README và APP_GUIDE. Các việc nhãn pilot, metric pilot và quay demo vẫn đang chờ.

## Review và merge app — 19/10/2026

- Đã nhận app `47326a6` và merge fast-forward vào `feat/ml-release`; giữ nhánh app remote nguyên trạng.
- Chạy lại trên máy Huy: **118 tests**, **13/13 HTTP/model**, **9/9 UI AppTest** đều qua; pip check và 8 artifact qua.
- Đã kiểm trình duyệt thật: upload, kết quả model, evidence tại giây 4, tải TXT, đổi file xóa kết quả cũ, video không tiếng. Không chạy pilot.
- Sửa launcher để dùng cả `.venv-model`/`.venv`; sửa script nghiệm thu để không ghi đè hoặc gắn nhầm tên máy Tiến.
- Hướng dẫn mở web và giới hạn review: [APP_MERGE_REVIEW.md](APP_MERGE_REVIEW.md). Số đo tóm tắt: `reports/app_merge_review_2026-10-19.json`.
- Còn nhãn của hai người, metric pilot và quay demo; không coi việc thành viên app đã bàn giao code là các mục này đã xong.


## Nghiệm thu máy app - 18/10/2026

- Thành viên app Lê Quyết Tiến nhận ZIP khớp SHA-256, kiểm 17 tệp bàn giao và 8 artifact model.
- Ghép `feat/ml-release` f5d9310 lên `feat/app` theo fast-forward, giữ giao diện d82798d và toàn bộ ML của Huy.
- Cài mới `.venv-model` trên Windows 11 / i5-12450H / Python 3.12.5. Đây là máy khác với Ryzen 5 7535HS của Huy.
- **118 tests pass**, `pip check` và verifier qua; **13/13 ca HTTP model thật**, **9/9 ca UI AppTest nối HTTP/model thật** qua. Xem `APP_ACCEPTANCE_TIEN.md` và ba JSON `reports/app_*` tương ứng.
- Thêm launcher model, biên bản riêng, thông tin sinh viên và sửa ánh xạ lỗi media từ model thành 422. Không sửa ngưỡng hoặc model.
- Chưa chạy classifier trên pilot; 6 nhãn độc lập của Tiến và 25 nhãn Huy vẫn cần người thực hiện. Không có metric pilot mới.
- Công cụ trình duyệt từ chối quyền truy cập nên không xác nhận kiểm thử tay/ảnh UI model hoặc quay video mới. Ghi rõ kiểm thử giao diện ở mức AppTest, không coi là thao tác trình duyệt thủ công.
- Báo cáo/slide bản dành cho Tiến cập nhật bằng chứng máy app và thông tin đã xác nhận. Phần ML gốc vẫn được giữ để đối chiếu.

Nhật ký dưới đây là lịch sử công việc của Huy; số đo của từng máy không trộn lẫn.

Cập nhật **18/10/2026**, nhánh tích hợp **`feat/ml-release`**. Người dùng xác nhận chưa gán nhãn và yêu cầu hoàn thiện các phần khác. Không tự tạo nhãn hoặc metric pilot.

## Trạng thái phần việc của Huy

| Mốc | Đã thực hiện | Còn cần hoàn tất |
|---|---|---|
| D1–D2 | Phạm vi, 25 clip có giấy phép, manifest nhóm/split khóa, frame + ASR, player gán nhãn | Huy gán 25, app gán 6, phân xử |
| D3 | Huấn luyện và đánh giá hai baseline, giữ checkpoint và checksum | Không có việc kỹ thuật còn thiếu trong baseline đã chọn |
| D4 | Hai modality thật, ghép app d82798d, 5 clip train qua HTTP/model thật | Không có lỗi tích hợp đã biết trong 13 ca smoke |
| D5 | Metric benchmark, ca lỗi ASR, công cụ agreement/gold/evaluation/ablation, 117 tests | Metric và ablation trên pilot có nhãn |
| D6 | Model card, cài bundle có kiểm hash, phụ thuộc PyAV 16.1.0, lock môi trường, handoff | Nghiệm thu cài/chạy trên máy thứ hai |
| D7 | Báo cáo kỹ thuật, PDF, slide và speaker notes, kịch bản demo + Q&A đã kiểm tra | Bổ sung metric pilot/thông tin sinh viên, diễn tập và quay demo |

Không quy đổi bảng này thành 100% dự án: nhãn, đánh giá đích và nghiệm thu của người dùng vẫn còn.

## Kiểm chứng bản tích hợp

- Full suite: **117 passed**, 71,50 giây. `pip check` pass. 8 artifact khớp manifest.
- **13/13 ca HTTP thật pass**: 5 clip train có cả hai score, 2 clip tổng hợp không tiếng/có âm thử, 6 ca đầu vào lỗi. Đây là kiểm tra kỹ thuật, không phải số dự đoán nội dung đúng.
- Lần gọi đầu 23,34 giây; 4 lần sau median 5,20, p95 mô tả 11,95 giây. Cùng nhóm nguồn, mỗi clip chạy một lần, chưa là benchmark hiệu năng đại diện.
- Nhánh app giữ nguyên trên remote. Nhánh tích hợp ghép từ ML `32d1ad4` và app `d82798d`; schema v1 không đổi, không fallback mock.
- Giữ kết quả/ASR/nhãn/video/checkpoint ở thư mục private hoặc ignored. Chỉ 005–009 train đã chạy classifier cho smoke; test pilot chưa chạy classifier.
- Audit D1–D2 bên dưới là ảnh chụp trạng thái 17/10, giữ nguyên. Dòng `predictions_run=false` trong audit áp dụng lúc audit, trước smoke train ngày 18/10.

## Bàn giao cho app ngày 18/10

- Đã cập nhật `docs/HANDOFF_APP.md` thành hướng dẫn tiếp nhận bản tích hợp; thay yêu cầu làm app từ đầu bằng nghiệm thu máy thứ hai, nhãn chéo và demo.
- Có tin nhắn/prompt chuyển cho Codex app tại `docs/APP_HANDOFF_MESSAGE.md` và mẫu trống `docs/APP_ACCEPTANCE_TEMPLATE.md`. Không điền pass thay cho người kiểm tra máy khác.
- Gói gửi riêng nằm tại `data/private/handoff-app/kidsafe-app-handoff-2026-10-18.zip`, kèm SHA-256; chứa model ZIP, annotation ZIP, tài liệu và PDF/PPTX. Không chứa dự đoán/ASR private hoặc raw dataset huấn luyện. Code lấy từ GitHub.
- Gói để Huy gửi cho thành viên app; chưa tự gửi tin nhắn, upload dịch vụ ngoài hoặc xác nhận người nhận đã nhận.

## Đầu ra để tiếp tục khi hết quota

- [Chạy bản tích hợp và bàn giao máy khác](RELEASE_HANDOFF.md).
- [Quy trình so phiếu, tạo gold và đánh giá](EVALUATION_GUIDE.md).
- [Báo cáo kỹ thuật](../reports/TECHNICAL_REPORT.md), [PDF](../output/pdf/KidSafeReels_BaoCaoKyThuat.pdf), [slide](../output/presentation/KidSafeReels_ThuyetTrinh.pptx).
- [Kịch bản demo và câu hỏi bảo vệ](DEFENSE_NOTES.md).
- Model ZIP local: `models/checkpoints/bundle/kidsafe-cpu-v1.zip`; không nằm trên Git.
- Phiếu/video local: `data/private/kidsafe-pilot-v1-annotation.zip`; không chứa dự đoán.
- Chưa có phiếu người thật. Khi nhận hai JSON, chạy hướng dẫn evaluation; không tải/train lại baseline, không thay split hoặc chỉnh ngưỡng bằng test.
- Đã commit và push bản tích hợp `fdec72b` lên `origin/feat/ml-release`. PDF 9 trang và slide 11 trang đã render/kiểm tra từng trang. Nhật ký này được lưu bằng commit tiếp theo; checkpoint/video vẫn chỉ ở local.

---

## Lịch sử đã lưu ngày 17/10


Cập nhật 17/10/2026. Nhánh đang làm: `feat/ml-pilot-data`, nối tiếp `cb64e05` (pipeline CPU). Đây là nhật ký trạng thái thực; chỉ đánh dấu xong khi có đầu ra và kiểm tra tương ứng.

## Công việc hiện tại: bù phần D1–D2

- [x] Chốt P0: Violence + Toxic tiếng Việt; API/schema v1 và baseline CPU đã có.
- [x] Người dùng xác nhận chưa có clip riêng; dùng nguồn công khai có quyền sử dụng.
- [x] Kiểm tra giấy phép: 8 nguồn CC BY 3.0 có lịch sử review; tạo 25 đoạn video thật, mỗi đoạn 25 giây.
- [x] Khóa manifest trước trích frame; 5 nhóm nguồn, train/validation/test = 17/4/4 (68/16/16).
- [x] Có quy tắc nhãn, player/phiếu cho Huy 25 clip và app 6 clip độc lập (24%).
- [x] Chạy thực 25/25 clip: 625 frame theo PTS, 58 đoạn ASR, cache theo checksum và cấu hình.
- [x] Audit hash/nguồn/nhóm/timestamp: 0 lỗi, 0 ứng viên gần trùng giao split theo dHash; **44 tests pass** (19,10 giây), gồm regression kiểm tra cú pháp JS.
- [ ] Hai người xem/nghe clip, gán nhãn độc lập và giải quyết bất đồng (chưa thực hiện).
- [x] Đã commit/push lên `origin/feat/ml-pilot-data`: `ddd0256` (D1–D2), `d380b64` (giữ checksum manifest khi clone Windows/Linux). Raw media, frame, ASR, ZIP và checkpoint không lên Git.

**Phần kỹ thuật của Huy trong D1–D2 đã có đầu ra; phần nhãn người thật chưa hoàn tất.** Không gộp hai trạng thái thành 100% D1–D2 của cả nhóm. 25 đoạn chỉ đến từ 8 video gốc; tập chủ yếu thông tin đời thường/giáo dục, chưa chứng minh có đủ lớp dương. Không dùng số lượng clip để suy ra độ chính xác.

Whisper trả transcript cho 15 clip (14 nhận diện `vi`, 1 `en`); 10 clip chưa phát hiện lời nói. Đây là kết quả ASR, chưa xác nhận ngôn ngữ/nội dung bằng người thật. Có 7 segment bị giới hạn mốc ở cuối clip, giữ cả mốc thô để đối chiếu. Không chạy classifier trên pilot và chưa dùng test chọn model/ngưỡng.

## Đầu ra và cách tiếp tục khi hết quota

- [Hướng dẫn chạy lại](PILOT_DATA.md), [quy tắc gán nhãn](ANNOTATION_GUIDE.md), [báo cáo D1–D2](../reports/PILOT_D1_D2.md).
- Manifest khóa: `data/pilot/manifest.json`; SHA-256 `6ef3f86b0ac931dd720a866231719d8ba2d01c72b102a55c911d08746c911e20`.
- Audit: `reports/pilot_data_audit.json`. Media/frame/ASR trong `data/processed/pilot-v1/`, không lên Git.
- Gói giao hai người: `data/private/kidsafe-pilot-v1-annotation.zip` (~61 MiB); giải nén, Huy mở `review-huy.html`, app mở `review-app.html`. Có ghi công nguồn; không chứa dự đoán/ASR/nhãn đã nộp.
- Nếu xử lý bị gián đoạn: chạy lại `python -m ml.pilot_data prepare`, `preprocess`, `audit`. Các clip hoàn thành có cache riêng. Không cần tải/huấn luyện model lại.
- Kiểm tra Git: byte manifest trong commit khớp checksum local; `git diff HEAD~2 --check` sạch sau cấu hình `cr-at-eol` riêng cho manifest khóa. Gói ZIP được kiểm tra CRC và đủ 25 media. Nhánh app không bị sửa/merge trong lượt này.
- Việc kế tiếp: hai người nộp JSON riêng → kiểm tra bằng `ml.pilot_annotation check` → phân xử bất đồng → báo support → bổ sung ca nguy hại có quyền nếu thiếu lớp dương → đánh giá theo split đã chốt. Giữ nhãn test khỏi vòng điều chỉnh.

## Mốc đã hoàn thành trước đó

- Toxic ViCTSD: train/evaluate thật, F1 Toxic 0,4492; `3bba5b5`.
- Visual RLVS + ASR + API: F1 Violence 0,9598 trên RLVS, 24 test và 9 ca API; `cb64e05`.
- Model artifacts và gói ZIP nằm local; tài liệu tại `docs/VIDEO_PIPELINE.md`.
- Có ca ASR gây cảnh báo Toxic nhầm; chưa có đánh giá moderation trên tập video tiếng Việt chung.

## Phân công đang chờ

Huy: dữ liệu/nhãn, đánh giá chung, báo cáo ML. Thành viên app: tích hợp model, UI/playback, máy thứ hai; có nhánh remote `feat/app` riêng, không bị sửa trong công việc dữ liệu này. D3–D7 chưa được coi là hoàn tất chỉ vì đã có baseline.
