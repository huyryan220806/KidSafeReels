"""Import full human annotations and freeze a new experiment without test selection."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil

from ml.speech_intake import ROOT, read, sha, validate, write
from ml.asr_recovery import repeated_asr

RUN = 'toxic-human-2026-11-13'
PRIVATE = ROOT / 'runs' / RUN
PUBLIC = ROOT / 'reports' / RUN
BASE = ROOT / 'models/checkpoints/toxic-phobert-v3'
PRIOR = ROOT / 'runs/toxic-precision-2026-10-29'
RECOVERY = ROOT / 'runs/toxic-reviewed-2026-10-29/recovery'
WRITTEN = ROOT / 'runs/model-improvement-2026-10-28/toxic'
REVIEW_MANIFEST = ROOT / 'data/private/huy-toxic-full200-2026-11-05/annotation/review-manifest.json'


def attach_annotations(source, annotations):
    rows = source['speech']
    by_id = {r['video_id']: r for r in annotations['rows']}
    if len(by_id) != len(annotations['rows']) or set(by_id) != {r['video_id'] for r in rows}:
        raise ValueError('Missing or duplicate human annotation IDs')
    groups, hashes, updated = {}, {}, []
    for row in rows:
        sid = row['video_id']; annotation = by_id[sid]
        for registry, key in ((groups, row['training_group']), (hashes, row['sha256'])):
            if key in registry and registry[key] != row['split']:
                raise ValueError('Source group or exact media spans multiple splits')
            registry[key] = row['split']
        if row['split'] not in ('train', 'valid', 'test'):
            raise ValueError('Unknown split')
        if type(annotation['toxic_language']) is not int or annotation['toxic_language'] not in (0, 1):
            raise ValueError('This experiment requires explicit binary human labels')
        updated.append(dict(deepcopy(row), previous_weak_label=row['training_label'],
            training_label=annotation['toxic_language'], label_status='single_human_reviewed_clip',
            annotation=deepcopy(annotation)))
    return dict(speech=updated)


def prepare(input_path):
    if PRIVATE.exists() or PUBLIC.exists():
        raise ValueError('Existing run preserved; resume its next stage instead of preparing again')
    annotations = read(input_path)
    status = validate(annotations, read(REVIEW_MANIFEST), 'huy', sha(REVIEW_MANIFEST))
    if status['missing'] or status['uncertain']:
        raise ValueError('Resolve missing/uncertain labels before this full-label experiment')
    source = read(PRIOR / 'manifest.json')
    manifest = attach_annotations(source, annotations)
    for row in manifest['speech']:
        if sha(ROOT / row['path']) != row['sha256']:
            raise ValueError('Annotated media differs from frozen media')
    PRIVATE.mkdir(parents=True); PUBLIC.mkdir(parents=True)
    shutil.copy2(input_path, PRIVATE / 'annotations-original.json')
    shutil.copy2(REVIEW_MANIFEST, PRIVATE / 'annotation-manifest.json')
    write(PRIVATE / 'manifest.json', manifest)
    provenance, recovered = {}, []
    for row in manifest['speech']:
        if row['split'] == 'test':
            continue
        sid = row['video_id']; path = PRIOR / 'asr' / f'{sid}.json'
        original = read(path)
        if original['sha256'] != row['sha256']:
            raise ValueError('Development ASR source changed')
        record = deepcopy(original)
        provenance[sid] = {'original_sha256': sha(path)}
        trigger = original['language'] == 'vi' and original['language_probability'] >= .5 and repeated_asr(original['segments'])
        if trigger:
            path = RECOVERY / f'{sid}.json'; retry = read(path)
            if retry['sha256'] != row['sha256'] or not retry['segments'] or repeated_asr(retry['segments']):
                raise ValueError('Invalid deployed recovery cache')
            record['segments'] = retry['segments']; recovered.append(sid)
            provenance[sid]['recovery_sha256'] = sha(path)
        write(PRIVATE / 'asr' / f'{sid}.json', record)
    write(PRIVATE / 'asr-provenance.json', provenance)
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in (
        PRIOR / 'manifest.json', WRITTEN / 'tokens-train.json', WRITTEN / 'tokens-valid.json',
        ROOT / 'models/artifacts.json', ROOT / 'models/checkpoints/violence-r3d18-v3.joblib',
        ROOT / 'models/checkpoints/faster-whisper-small/model.bin',
        ROOT / 'ml/resources/toxic_lexicon_v1.json', ROOT / 'ml/asr_recovery.py')}
    protocol = dict(run_id=RUN, frozen_utc=datetime.now(timezone.utc).isoformat(),
        input_annotation_sha256=sha(input_path), manifest_sha256=sha(PRIVATE / 'manifest.json'),
        annotation_manifest_sha256=sha(REVIEW_MANIFEST), inputs=inputs,
        model_files={p.name: sha(p) for p in BASE.iterdir() if p.is_file()},
        asr_hashes={p.stem: sha(p) for p in (PRIVATE / 'asr').glob('*.json')},
        label_scope='Single human listened to 200 clips; not independently adjudicated; no full reference transcripts',
        training=dict(seed=42, epochs=5, encoder_lr=5e-6, classifier_lr=2e-5,
            weight_decay=.01, accumulate_clips=8, trainable='last two encoder blocks and classifier',
            pooling='max segment log odds, same as deployed max probability',
            loss='speech clip BCE + 0.5 * balanced written rehearsal CE; no invented segment negatives',
            replay_batch=8, replay_positive=4, replay_weight=.5,
            replay_source='ViCTSD train only, deterministic balanced samples with replacement'),
        selection=dict(thresholds='0.05 through 0.95 step .01',
            constraints='speech hybrid FP/FN <= active v3; speech model-only F1 >= v3; written validation recall and F1 >= v3',
            rank='eligible, speech hybrid F1, fewer speech FP, written F1, lower speech log-loss, earlier epoch, closer to .78',
            baseline_calibration='report epoch0 threshold scan separately; never hide threshold-only improvements',
            failure='retain v3; keep ineligible candidate for diagnosis only'),
        test_policy='After checkpoint and threshold freeze only. Existing 30 test clips already seen in earlier experiments; report as reused holdout, not fresh independent test. Never tune on these results.',
        deployment='candidate first; require CPU parity and no regression before any activation; keep Violence v3',
        asr_policy='Cached small CPU INT8 beam5 with deployed repetition recovery; no ASR tuning',
        max_length=192)
    write(PUBLIC / 'protocol.json', protocol)
    changes = [dict(video_id=r['video_id'], split=r['split'], old=r['previous_weak_label'], new=r['training_label'])
               for r in manifest['speech'] if r['previous_weak_label'] != r['training_label']]
    write(PRIVATE / 'label-changes.json', changes)
    audit = dict(validation=status, input_sha256=sha(input_path),
        split_counts=dict(Counter(r['split'] for r in manifest['speech'])),
        label_changes=len(changes), dev_label_changes=sum(r['split'] != 'test' for r in changes),
        verified_full_transcripts=sum(r['transcript_verified'] for r in annotations['rows']),
        reference_wer_cer_available=False, development_asr_caches=len(provenance),
        deployed_recovery_caches=len(recovered), no_test_predictions_accessed=True)
    write(PUBLIC / 'import-audit.json', audit)
    (PRIVATE / 'CONTINUE.md').write_text(
        '# Tiếp tục Toxic human — 13/11/2026\n\nĐã nhập đủ 200 nhãn, giữ nguyên bản gốc và split. '
        'Protocol khóa trước train. Chạy train bằng .venv-train, evaluate bằng .venv. '
        'Không prepare lại hoặc dùng test chọn ngưỡng. Violence v3 giữ nguyên. '
        'Các file progress/selection/evaluation trong reports cho biết stage đã xong.\n', encoding='utf-8')
    print(audit, flush=True)


def load():
    protocol = read(PUBLIC / 'protocol.json')
    if sha(PRIVATE / 'manifest.json') != protocol['manifest_sha256']:
        raise ValueError('Human manifest changed')
    if sha(PRIVATE / 'annotations-original.json') != protocol['input_annotation_sha256']:
        raise ValueError('Original human annotations changed')
    for path, digest in protocol['inputs'].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f'Frozen input changed: {path}')
    for name, digest in protocol['model_files'].items():
        if sha(BASE / name) != digest:
            raise ValueError('Active baseline model changed')
    for sid, digest in protocol['asr_hashes'].items():
        if sha(PRIVATE / 'asr' / f'{sid}.json') != digest:
            raise ValueError('Development ASR cache changed')
    return protocol, read(PRIVATE / 'manifest.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    prepare(parser.parse_args().input)
