"""CPU-ASR development experiment and a private, unlabelled ten-clip review pack."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import random
import shutil
import time

from ml.experiment_metrics import write_json
from ml.local_adaptation import locations, read
from ml.speech_intake import blank_row, validate
from ml.toxic_baseline import ROOT, sha256

RUN = 'toxic-precision-2026-10-29'
PREVIOUS = ROOT / 'runs/local-adaptation-2026-10-29'
BASE = ROOT / 'models/checkpoints/toxic-phobert-v3'
VIOLENCE = ROOT / 'models/checkpoints/violence-r3d18-v3.joblib'


def development_rows(manifest):
    return [r for r in manifest['speech'] if r['split'] in ('train', 'valid')]


def choose_review(manifest, predictions):
    """Five suspected FP and five FN; source-disjoint, never sample test."""
    by_id = {r['video_id']: r for r in development_rows(manifest)}
    eligible = [p for p in predictions if p['video_id'] in by_id and p['variant'] == 'baseline'
                and (p['hybrid_flag'] == 'True') != bool(int(p['label']))]
    chosen, groups = [], set()
    for label in (0, 1):
        pool = sorted((p for p in eligible if int(p['label']) == label),
                      key=lambda p: (p['split'] != 'valid',
                                     -float(p['score']) if label == 0 else float(p['score']), p['video_id']))
        count = 0
        for p in pool:
            row = by_id[p['video_id']]
            if row['training_group'] in groups:
                continue
            chosen.append(row); groups.add(row['training_group']); count += 1
            if count == 5:
                break
        if count != 5:
            raise ValueError('Not enough distinct development error groups')
    random.Random(42).shuffle(chosen)
    return chosen


def prepare():
    private, public = locations(RUN)
    if (public/'protocol.json').exists():
        raise ValueError('Protocol already frozen; use asr or review-check')
    manifest = read(PREVIOUS/'manifest.json')
    private.mkdir(parents=True, exist_ok=True)
    # Keep the existing split, including the untouched test metadata for leakage checks.
    shutil.copy2(PREVIOUS/'manifest.json', private/'manifest.json')
    protocol = dict(run_id=RUN, frozen_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=sha256(private/'manifest.json'), seed=42,
        violence_sha256=sha256(VIOLENCE), violence_policy='keep active v3, no training or threshold changes',
        model_files={p.name: sha256(p) for p in BASE.iterdir() if p.is_file()},
        lexicon_sha256=sha256(ROOT/'ml/resources/toxic_lexicon_v1.json'),
        asr=dict(model='small', device='cpu', compute_type='int8', cpu_threads=3,
                 beam_size=5, temperature=0, vad_filter=True, condition_on_previous_text=False,
                 task='transcribe', language_gate='vi and probability >= 0.5',
                 segments='all decoder-accepted nonempty segments; no second no_speech filter'),
        training=dict(initial='active PhoBERT v3', epochs=5, negative_weight=2.0,
            encoder_lr=5e-6, classifier_lr=2e-5, accumulate_clips=8, weight_decay=.01,
            trainable='last two encoder blocks and classifier', pooling='max segment log odds',
            labels='user-reported weak clip labels; no invented human annotations'),
        selection=dict(split='valid only', thresholds='0.05 through 0.95 step 0.01',
            constraints='hybrid FP <= baseline hybrid FP; hybrid recall >= baseline hybrid recall; model-only FP <= baseline model-only FP',
            rank='eligible first, lowest hybrid FP, highest hybrid recall, lowest log loss, earliest epoch',
            if_no_eligible='retain active v3, candidate only'),
        test_policy='Do not transcribe, select on or evaluate the previously seen test in this run. No new independent test claim.',
        deployment='candidate only until manual review and broader regression; do not auto-activate',
        review_policy='ten train/validation disagreements only; human labels start blank; not an accuracy sample')
    write_json(public/'protocol.json', protocol)
    with (PREVIOUS/'toxic/predictions.csv').open(encoding='utf-8-sig') as f:
        selected = choose_review(manifest, list(csv.DictReader(f)))
    review = private/'review10'; review.mkdir(exist_ok=True)
    review_manifest = dict(rows=selected, assignments={'huy':[r['video_id'] for r in selected]},
                           source_manifest_sha256=protocol['manifest_sha256'])
    write_json(review/'manifest.json', review_manifest)
    payload = dict(manifest_sha256=sha256(review/'manifest.json'), annotator='huy',
                   rows=[blank_row(r['video_id'], 'huy') for r in selected])
    write_json(review/'annotations-huy.blank.json', payload)
    config = dict(dataset='toxic-dev10-2026-10-29', initial=payload, clips={})
    for r in selected:
        source = ROOT/r['path']; target = review/'media'/f"{r['video_id']}.mp4"
        target.parent.mkdir(exist_ok=True)
        if sha256(source) != r['sha256']:
            raise ValueError('Review source changed')
        shutil.copy2(source, target)
        if sha256(target) != r['sha256']:
            raise ValueError('Review copy checksum mismatch')
        config['clips'][r['video_id']] = dict(duration=r['duration_seconds'], src=f"media/{r['video_id']}.mp4")
    import json
    template = (ROOT/'ml/templates/speech_review.html').read_text(encoding='utf-8')
    page = template.replace('__REVIEW_CONFIG__', json.dumps(config, ensure_ascii=False).replace('<', '\\u003c'))
    (review/'review-huy.html').write_text(page, encoding='utf-8')
    guide = '''# Nghe duyệt 10 clip Toxic\n\nMở `review-huy.html`, hoặc dùng máy chủ local theo hướng dẫn repo.\n\n1. Nghe hết từng clip, đánh dấu đã nghe và ngôn ngữ.\n2. Chọn 1 nếu nghe rõ chửi tục/lăng mạ/đe dọa/miệt thị; 0 nếu không có; chưa rõ nếu không nghe rõ. Không đoán theo tên tệp hay nhãn thư mục.\n3. Nhãn 1: thêm mốc bắt đầu/kết thúc và câu nghe được. Chép lời thực sự nghe được vào ô transcript; nếu chỉ chép một phần, ghi rõ trong ghi chú và không tick đã đối chiếu toàn bộ transcript.\n4. Bấm Tải JSON nhãn hiện tại; gửi đường dẫn tệp JSON cho Codex. Có thể tải dở rồi Nạp lại JSON để làm tiếp.\n\n10 clip chỉ thuộc train/validation, được chọn vì bất đồng với nhãn nguồn. Đây không phải mẫu ngẫu nhiên để tính Accuracy, và nhãn nguồn chưa được coi là nhãn người đã xác nhận. Phiếu không hiện dự đoán hay ASR để tránh dẫn dắt. Không tải video/phiếu lên Git.\n'''
    (review/'HUONG_DAN.md').write_text(guide, encoding='utf-8')
    (private/'CONTINUE.md').write_text('# Tiếp tục Toxic CPU-ASR\n\nViolence v3 giữ nguyên. Protocol đã khóa; phiếu review10 sẵn sàng, nhãn vẫn trống.\nChạy hai worker: `.venv/Scripts/python -m ml.toxic_precision asr --worker 0` và worker1.\nCache từng clip, chạy lại tiếp tục phần thiếu. Chỉ train/valid; test không mở.\n', encoding='utf-8')
    print(dict(review=str(review/'review-huy.html'), ids=[r['video_id'] for r in selected],
               duration_seconds=sum(r['duration_seconds'] for r in selected)), flush=True)


def load():
    private, public = locations(RUN)
    protocol = read(public/'protocol.json'); manifest = read(private/'manifest.json')
    if sha256(private/'manifest.json') != protocol['manifest_sha256'] or sha256(VIOLENCE) != protocol['violence_sha256']:
        raise ValueError('Manifest or frozen Violence v3 changed')
    if sha256(ROOT/'ml/resources/toxic_lexicon_v1.json') != protocol['lexicon_sha256']:
        raise ValueError('Frozen lexicon changed')
    for name, digest in protocol['model_files'].items():
        if sha256(BASE/name) != digest:
            raise ValueError('Active Toxic model changed')
    return private, public, protocol, manifest


def asr(worker):
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    private, public, protocol, manifest = load()
    model = WhisperModel(str(ROOT/'models/checkpoints/faster-whisper-small'), device='cpu',
                         compute_type='int8', cpu_threads=3, local_files_only=True)
    rows = sorted(development_rows(manifest), key=lambda r: (r['split'] != 'valid', r['video_id']))[worker::2]
    started = time.perf_counter()
    for i, row in enumerate(rows, 1):
        out = private/'asr'/f"{row['video_id']}.json"
        if out.exists():
            cache = read(out)
            if cache['sha256'] != row['sha256'] or cache['configuration'] != protocol['asr']:
                raise ValueError('ASR cache mismatch')
            continue
        path = ROOT/row['path']
        if sha256(path) != row['sha256']:
            raise ValueError('Source changed')
        segments, info = model.transcribe(decode_audio(str(path), sampling_rate=16000), task='transcribe',
            beam_size=5, temperature=0, vad_filter=True, condition_on_previous_text=False)
        accepted = [dict(start=s.start, end=s.end, text=s.text, avg_logprob=s.avg_logprob,
                         compression_ratio=s.compression_ratio, no_speech_prob=s.no_speech_prob)
                    for s in segments if s.text.strip()] if info.language == 'vi' and info.language_probability >= .5 else []
        record = dict(video_id=row['video_id'], sha256=row['sha256'], configuration=protocol['asr'],
                      language=info.language, language_probability=info.language_probability, segments=accepted)
        # Same-directory atomic replace prevents partial cache files on interruption.
        temp = out.with_suffix('.tmp'); write_json(temp, record); temp.replace(out)
        write_json(public/f'asr-progress-{worker}.json', dict(completed=i, total=len(rows),
                   elapsed_seconds=time.perf_counter()-started))
        print(f'CPU ASR worker{worker}: {i}/{len(rows)} {row["split"]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'asr', 'review-check'])
    parser.add_argument('--worker', type=int, choices=[0, 1], default=0)
    parser.add_argument('--input', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'asr':
        asr(args.worker)
    else:
        if not args.input:
            parser.error('--input is required')
        private, _, _, _ = load(); review = private/'review10'
        print(validate(read(args.input), read(review/'manifest.json'), 'huy', sha256(review/'manifest.json')))
