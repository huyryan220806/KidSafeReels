"""Import reviewed development labels and diagnose ASR without touching frozen test data."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

from ml.experiment_metrics import write_csv, write_json
from ml.local_adaptation import read
from ml.speech_intake import validate
from ml.toxic_baseline import ROOT, sha256
from ml.toxic_precision import load as load_precision
from ml.toxic_rules import ToxicRules, lexical_decision
from ml.asr_recovery import audio_windows, transcribe_chunks

RUN = 'toxic-reviewed-2026-10-29'
PRIVATE = ROOT/'runs'/RUN
PUBLIC = ROOT/'reports'/RUN
VARIANTS = ('vi_repeat_guard', 'vi_chunk10')


def resolve_annotations(original, corrections):
    resolved = deepcopy(original)
    by_id = {r['video_id']:r for r in resolved['rows']}
    if not corrections.get('source') or not corrections.get('confirmed_by_user'):
        raise ValueError('Corrections require explicit user confirmation provenance')
    seen = set()
    for change in corrections['rows']:
        sid = change['video_id']; label = change['toxic_language']
        if sid not in by_id or sid in seen or type(label) is not int or label not in (0, 1):
            raise ValueError('Invalid correction')
        seen.add(sid); row = by_id[sid]
        row['toxic_language'] = label
        if label == 0:
            # Preserve the earlier quote as context, not as Toxic evidence for a negative.
            row['prior_evidence_for_adjudication'] = deepcopy(row['evidence'])
            row['evidence'] = []
        row['adjudication_note'] = change['reason']
    return resolved


def prepare(input_path, corrections_path):
    if (PUBLIC/'protocol.json').exists():
        raise ValueError('Review import already frozen')
    source, _, prior_protocol, manifest = load_precision()
    review_manifest_path = source/'review10/manifest.json'
    review_manifest = read(review_manifest_path); digest = sha256(review_manifest_path)
    original, corrections = read(input_path), read(corrections_path)
    original_status = validate(original, review_manifest, 'huy', digest)
    resolved = resolve_annotations(original, corrections)
    status = validate(resolved, review_manifest, 'huy', digest)
    if status['missing'] or status['uncertain']:
        raise ValueError('This run requires ten explicit binary reviewed labels')
    by_id = {r['video_id']:r for r in manifest['speech']}
    rows = []
    for annotation in resolved['rows']:
        row = by_id[annotation['video_id']]
        if row['split'] not in ('train', 'valid'):
            raise ValueError('Reviewed error queue must not include test')
        rows.append(dict(row, reviewed_label=annotation['toxic_language'], annotation=annotation))
    PRIVATE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, PRIVATE/'annotations-original.json')
    shutil.copy2(corrections_path, PRIVATE/'chat-adjudications.json')
    write_json(PRIVATE/'annotations-resolved.json', resolved)
    write_json(PRIVATE/'manifest.json', dict(rows=rows))
    protocol = dict(run_id=RUN, created_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=sha256(PRIVATE/'manifest.json'), annotation_sha256=sha256(PRIVATE/'annotations-original.json'),
        resolved_sha256=sha256(PRIVATE/'annotations-resolved.json'), original_status=original_status, resolved_status=status,
        violence_sha256=prior_protocol['violence_sha256'], lexicon_sha256=prior_protocol['lexicon_sha256'],
        split_counts=dict(Counter(r['split'] for r in rows)), full_verified_transcripts=0,
        baseline='Frozen CPU INT8 small/beam5 traces from toxic-precision-2026-10-29',
        variants=dict(vi_repeat_guard=dict(language='vi', vad_filter=True, repetition_penalty=1.1, no_repeat_ngram_size=3),
                      vi_chunk10=dict(language='vi', vad_filter=False, seconds=10, overlap=1)),
        shared=dict(task='transcribe', beam_size=5, temperature=0, condition_on_previous_text=False,
                    device='cpu', compute_type='int8', cpu_threads=4, prompt=None),
        language_policy='Only retry clips whose original full ASR confirmed Vietnamese probability >= 0.5',
        selection='Only four reviewed TRAIN clips: fewer FN without more FP using active v3+rules; then fewer FP/FN and lower elapsed time',
        validation='Evaluate the selected ASR configuration once on six reviewed VALID clips; not independent test',
        limitations='Error-selected sample, one reviewer, no full reference transcripts; no WER/CER or general app accuracy',
        deployment='diagnostic candidate only; no model or active pipeline changed')
    write_json(PUBLIC/'protocol.json', protocol)
    diagnostics = []
    with (source/'toxic/predictions.csv').open(encoding='utf-8-sig') as stream:
        predicted = {(r['video_id'],r['variant']):r for r in csv.DictReader(stream) if r['split'] in ('train','valid')}
    for row in rows:
        for variant in ('baseline','selected'):
            pred = predicted[row['video_id'],variant]
            truth = row['reviewed_label']; flag = pred['hybrid_flag']=='True'
            diagnostics.append(dict(video_id=row['video_id'], split=row['split'], variant=variant, label=truth,
                score=float(pred['score']), rule_flag=pred['rule_flag']=='True', flagged=flag,
                error='FP' if flag and not truth else 'FN' if not flag and truth else 'correct'))
    write_csv(PRIVATE/'reviewed-predictions.csv', diagnostics)
    counts = [dict(variant=v,split=s,total=sum(r['variant']==v and r['split']==s for r in diagnostics),
                fp=sum(r['variant']==v and r['split']==s and r['error']=='FP' for r in diagnostics),
                fn=sum(r['variant']==v and r['split']==s and r['error']=='FN' for r in diagnostics))
              for v in ('baseline','selected') for s in ('train','valid')]
    write_csv(PUBLIC/'reviewed-error-counts.csv', counts)
    write_json(PUBLIC/'import-audit.json', dict(original=original_status, resolved=status,
        source_label_disagreements=sum(r['reviewed_label'] != r['training_label'] for r in rows),
        full_verified_transcripts=sum(r['annotation']['transcript_verified'] for r in rows),
        scope='Ten reviewed development clips, not 200 reviewed clips or representative accuracy'))
    print(dict(status=status, counts=counts), flush=True)


def load():
    source, _, _, _ = load_precision()
    protocol = read(PUBLIC/'protocol.json')
    if sha256(PRIVATE/'manifest.json') != protocol['manifest_sha256']:
        raise ValueError('Reviewed manifest changed')
    return source, protocol, read(PRIVATE/'manifest.json')['rows']


def decode_variant(asr, audio, variant):
    options = dict(task='transcribe', beam_size=5, temperature=0, condition_on_previous_text=False, language='vi')
    if variant == 'vi_repeat_guard':
        windows = [(0, len(audio))]
        options.update(vad_filter=True, repetition_penalty=1.1, no_repeat_ngram_size=3)
    elif variant == 'vi_chunk10':
        return transcribe_chunks(asr, audio)
    else:
        raise ValueError('Unknown ASR variant')
    output = []
    for i, (start, end) in enumerate(windows):
        segments, _ = asr.transcribe(audio[start:end], **options)
        for s in segments:
            begin, finish = start/16000+s.start, min(end/16000, start/16000+s.end)
            midpoint = (begin+finish)/2
            left = (start+windows[i-1][1])/32000 if i else 0
            right = (end+windows[i+1][0])/32000 if i+1<len(windows) else len(audio)/16000
            if not s.text.strip() or finish <= begin or not left <= midpoint < right:
                continue
            output.append(dict(start=begin, end=finish, text=s.text, avg_logprob=s.avg_logprob,
                               compression_ratio=s.compression_ratio, no_speech_prob=s.no_speech_prob))
    return output


def predict_segments(model, rules, segments):
    values = [model.predict(s['text'])['score'] for s in segments]
    score = max(values) if values else None
    lexical = any(lexical_decision(rules.match(s['text']),s)['flagged'] for s in segments)
    return dict(score=score, rule_flag=lexical, flagged=lexical or (score is not None and score>=model.threshold))


def experiment(split):
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    from ml.deep_inference import PhoBertClassifier
    source, protocol, rows = load()
    if split == 'valid':
        selection = read(PUBLIC/'asr-selection.json')
        if not selection['eligible']:
            raise ValueError('No eligible TRAIN configuration; retain baseline')
        variants = [selection['variant']]
    else:
        if (PUBLIC/'asr-selection.json').exists():
            raise ValueError('Selection already frozen')
        variants = list(VARIANTS)
    if (PUBLIC/f'asr-{split}-counts.csv').exists():
        raise ValueError('Stage already evaluated')
    asr = WhisperModel(str(ROOT/'models/checkpoints/faster-whisper-small'), device='cpu', compute_type='int8', cpu_threads=4, local_files_only=True)
    model = PhoBertClassifier(ROOT/'models/checkpoints/toxic-phobert-v3/kidsafe-config.json'); rules = ToxicRules()
    results = []
    for row in (r for r in rows if r['split']==split):
        sid=row['video_id']; original=read(source/'asr'/f'{sid}.json')
        path = ROOT/row['path']
        if sha256(path)!=row['sha256'] or original['sha256']!=row['sha256']:
            raise ValueError('Audio or original ASR changed')
        audio = decode_audio(str(path), sampling_rate=16000)
        results.append(dict(video_id=sid,label=row['reviewed_label'],variant='baseline',elapsed_seconds=None,
                            **predict_segments(model,rules,original['segments'])))
        for variant in variants:
            target = PRIVATE/'asr'/variant/f'{sid}.json'
            if target.exists():
                cached = read(target)
                if cached['sha256']!=row['sha256'] or cached['protocol_sha256']!=sha256(PUBLIC/'protocol.json'):
                    raise ValueError('Retry cache mismatch')
            else:
                started=time.perf_counter()
                eligible=original['language']=='vi' and original['language_probability']>=.5
                segments=decode_variant(asr,audio,variant) if eligible else original['segments']
                cached=dict(video_id=sid,sha256=row['sha256'],protocol_sha256=sha256(PUBLIC/'protocol.json'),
                    segments=segments,elapsed_seconds=time.perf_counter()-started,language_retry_eligible=eligible,
                    prediction=predict_segments(model,rules,segments))
                write_json(target,cached)
            results.append(dict(video_id=sid,label=row['reviewed_label'],variant=variant,
                                elapsed_seconds=cached['elapsed_seconds'],**cached['prediction']))
            print(f'ASR {split} {variant} {sid}: flagged={cached["prediction"]["flagged"]}',flush=True)
    write_csv(PRIVATE/f'asr-{split}-predictions.csv',results)
    counts=[]
    for variant in ['baseline']+variants:
        part=[r for r in results if r['variant']==variant]
        counts.append(dict(variant=variant,n=len(part),
            fp=sum(r['flagged'] and r['label']==0 for r in part),fn=sum(not r['flagged'] and r['label']==1 for r in part),
            missing_asr=sum(r['score'] is None for r in part),
            elapsed_seconds=None if variant=='baseline' else sum(r['elapsed_seconds'] for r in part)))
    write_csv(PUBLIC/f'asr-{split}-counts.csv',counts)
    if split=='train':
        baseline=counts[0]
        eligible=[r for r in counts[1:] if r['fp']<=baseline['fp'] and r['fn']<baseline['fn']]
        best=min(eligible or counts[1:],key=lambda r:(r['fp'],r['fn'],r['elapsed_seconds']))
        write_json(PUBLIC/'asr-selection.json',dict(**best,eligible=bool(eligible),selected_on='four reviewed train clips only',
            protocol_sha256=sha256(PUBLIC/'protocol.json'),deployment='candidate_only'))
    print(counts,flush=True)


def recovery_audit():
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    from ml.deep_inference import PhoBertClassifier
    from ml.asr_recovery import repeated_asr, recover_repetition
    source, _, reviewed = load()
    audit_path=PUBLIC/'recovery-audit.json'
    if audit_path.exists():
        raise ValueError('Recovery audit already frozen')
    reviewed_by_id={r['video_id']:r for r in reviewed}
    manifest=read(source/'manifest.json')
    rows=[r for r in manifest['speech'] if r['split'] in ('train','valid')]
    asr=WhisperModel(str(ROOT/'models/checkpoints/faster-whisper-small'),device='cpu',compute_type='int8',cpu_threads=4,local_files_only=True)
    model=PhoBertClassifier(ROOT/'models/checkpoints/toxic-phobert-v3/kidsafe-config.json');rules=ToxicRules()
    results=[]
    for row in rows:
        original=read(source/'asr'/f"{row['video_id']}.json")
        if original['language']!='vi' or original['language_probability']<.5 or not repeated_asr(original['segments']):
            continue
        path=ROOT/row['path']
        if sha256(path)!=row['sha256']:raise ValueError('Audio changed')
        audio=decode_audio(str(path),sampling_rate=16000)
        started=time.perf_counter()
        recovered,warning=recover_repetition(asr,audio,original['segments'],language=original['language'],language_probability=original['language_probability'])
        if any(not 0<=s['start']<s['end']<=len(audio)/16000 for s in recovered):
            raise ValueError('Invalid recovered timestamps')
        known=reviewed_by_id.get(row['video_id'])
        record=dict(video_id=row['video_id'],split=row['split'],sha256=row['sha256'],
            label=known['reviewed_label'] if known else row['training_label'],
            label_status='human_reviewed' if known else 'user_reported_weak_clip_label',
            baseline=predict_segments(model,rules,original['segments']),recovered=predict_segments(model,rules,recovered),
            warning=warning,elapsed_seconds=time.perf_counter()-started,segments=recovered)
        write_json(PRIVATE/'recovery'/f"{row['video_id']}.json",record);results.append(record)
        print(f'Recovery {row["video_id"]}: {record["baseline"]["flagged"]} -> {record["recovered"]["flagged"]}',flush=True)
    counts=[]
    for variant in ('baseline','recovered'):
        counts.append(dict(variant=variant,n=len(results),fp=sum(r[variant]['flagged'] and r['label']==0 for r in results),
            fn=sum(not r[variant]['flagged'] and r['label']==1 for r in results)))
    write_json(audit_path,dict(development_scanned=len(rows),triggered=len(results),unchanged=len(rows)-len(results),
        trigger_splits=dict(Counter(r['split'] for r in results)),
        trigger_labels=dict(Counter(str(r['label']) for r in results)),
        label_status=dict(Counter(r['label_status'] for r in results)),counts=counts,
        gate_pass=bool(results) and counts[1]['fp']<=counts[0]['fp'] and counts[1]['fn']<counts[0]['fn'],
        scope='Conditional retry on cached development traces; all non-triggered traces unchanged, not independent test',
        protocol_sha256=sha256(PUBLIC/'recovery-protocol.json')))
    print(counts,flush=True)


def summarize():
    from ml.experiment_metrics import clean_svg
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _, _, rows=load(); labels={r['video_id']:r['reviewed_label'] for r in rows}
    predictions=[]
    for split in ('train','valid'):
        with (PRIVATE/f'asr-{split}-predictions.csv').open(encoding='utf-8-sig') as stream:
            predictions.extend(csv.DictReader(stream))
    baseline={r['video_id']:r['flagged']=='True' for r in predictions if r['variant']=='baseline'}
    chunked={r['video_id']:r['flagged']=='True' for r in predictions if r['variant']=='vi_chunk10'}
    conditional=dict(baseline)
    for path in (PRIVATE/'recovery').glob('*.json'):
        record=read(path)
        if record['video_id'] in conditional:
            conditional[record['video_id']]=record['recovered']['flagged']
    summary=[]
    for variant,values in (('baseline',baseline),('all_chunked',chunked),('conditional_recovery',conditional)):
        if set(values)!=set(labels):raise ValueError('Missing reviewed predictions')
        summary.append(dict(variant=variant,n=len(labels),fp=sum(values[k] and labels[k]==0 for k in labels),
            fn=sum(not values[k] and labels[k]==1 for k in labels)))
    write_csv(PUBLIC/'reviewed-asr-comparison.csv',summary)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    x=list(range(3))
    ax.bar([i-.18 for i in x],[r['fp'] for r in summary],.36,label='Báo nhầm FP')
    ax.bar([i+.18 for i in x],[r['fn'] for r in summary],.36,label='Bỏ sót FN')
    ax.set_xticks(x,['ASR gốc','Chia đoạn mọi clip','Chỉ phục hồi lặp'])
    ax.set_yticks(range(6));ax.set_ylabel('Số clip');ax.legend()
    ax.set_title('10 clip được chọn để tìm lỗi · nhãn Huy đã duyệt\nCùng model Toxic v3 và bộ luật; không phải test độc lập')
    for suffix in ('png','svg'):fig.savefig(PUBLIC/f'reviewed-asr-comparison.{suffix}',dpi=160)
    clean_svg(PUBLIC/'reviewed-asr-comparison.svg');plt.close(fig)
    print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','train-dev','valid-dev','recovery-audit','summarize'])
    parser.add_argument('--input',type=Path); parser.add_argument('--corrections',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':
        if not args.input or not args.corrections:
            parser.error('prepare requires input and corrections')
        prepare(args.input,args.corrections)
    elif args.command=='recovery-audit':
        recovery_audit()
    elif args.command=='summarize':
        summarize()
    else:
        experiment('train' if args.command=='train-dev' else 'valid')
