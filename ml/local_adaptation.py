"""Freeze source-group splits and resumable ASR for user-authorized weak-label training."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from ml.experiment_metrics import write_json
from ml.toxic_baseline import ROOT, sha256

SPEECH = ROOT / 'data/private/speech200-2026-10-28'
AIRTLAB = ROOT / 'data/private/violence-drive-2026-10-28/verified-manifest.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def locations(run_id):
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in run_id):
        raise ValueError('Invalid run ID')
    return ROOT / 'runs' / run_id, ROOT / 'reports' / run_id


def speech_groups(rows, pairs):
    parent = {r['video_id']: r['video_id'] for r in rows}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    seen = {}
    for r in rows:
        for key in (('source', r['source_group_id']), ('sha', r['sha256'])):
            if key in seen:
                union(r['video_id'], seen[key])
            seen[key] = r['video_id']
    for pair in pairs:
        union(pair['left'], pair['right'])
    return {r['video_id']: find(r['video_id']) for r in rows}


def split_speech(rows, pairs, exposed):
    groups = speech_groups(rows, pairs)
    forced = {groups[x] for x in exposed}
    available = [r for r in rows if groups[r['video_id']] not in forced]
    labels = np.array([r['user_reported_clip_label'] for r in available])
    keys = np.array([groups[r['video_id']] for r in available])
    best = None
    # Metadata-only balancing. Never use model scores/text to choose a split.
    for i, (rest, test) in enumerate(GroupShuffleSplit(n_splits=100, test_size=30/len(available), random_state=42).split(labels, labels, keys)):
        for train_rel, valid_rel in GroupShuffleSplit(n_splits=10, test_size=30/len(rest), random_state=1000+i).split(rest, labels[rest], keys[rest]):
            valid = rest[valid_rel]
            counts = [np.bincount(labels[x], minlength=2) for x in (valid, test)]
            if any(min(c) < 8 for c in counts):
                continue
            cost = sum(int(np.abs(c-15).sum()) for c in counts)
            rank = (cost, i, tuple(valid), tuple(test))
            if best is None or rank < best[0]:
                best = (rank, valid, test)
    if best is None:
        raise ValueError('Cannot create two-class grouped holdouts')
    assignment = {r['video_id']: 'train' for r in rows}
    for split, indices in (('valid', best[1]), ('test', best[2])):
        for idx in indices:
            assignment[available[idx]['video_id']] = split
    result = [dict(r, split=assignment[r['video_id']], training_group=groups[r['video_id']],
                   training_label=r['user_reported_clip_label'], label_status='user_reported_weak_clip_label') for r in rows]
    sets = {s: {r['training_group'] for r in result if r['split'] == s} for s in ('train', 'valid', 'test')}
    assert not (sets['train'] & sets['valid'] or sets['train'] & sets['test'] or sets['valid'] & sets['test'])
    assert all(assignment[x] == 'train' for x in exposed)
    return result


def prepare(run_id):
    private, public = locations(run_id)
    if (private/'manifest.json').exists() or (public/'protocol.json').exists():
        raise ValueError('Run already frozen; resume subsequent commands')
    original = read(SPEECH/'manifest.json')['rows']
    if len(original) != 200 or Counter(r['user_reported_clip_label'] for r in original) != {0:100,1:100}:
        raise ValueError('Unexpected supplied speech cohort')
    exposure = read(SPEECH/'development-exposure-user-failures-2026-10-28.json')
    rows = split_speech(original, read(SPEECH/'duplicate_review.json')['pairs'], exposure['excluded_from_future_independent_test'])
    violence = read(AIRTLAB)['rows']
    if len(violence) != 100 or Counter(r['source_label'] for r in violence) != {0:50,1:50}:
        raise ValueError('Unexpected AIRTLab cohort')
    rlvs = read(ROOT/'data/splits/rlvs-v1.json')['rows']
    if {r['sha256'] for r in rlvs} & {r['sha256'] for r in violence}:
        raise ValueError('AIRTLab overlaps RLVS')
    for row in rows + violence:
        if sha256(ROOT/row['path']) != row['sha256']:
            raise ValueError('Local media changed since intake')
    manifest = dict(speech=rows, violence=[dict(r, split='train', training_label=r['source_label'],
                        label_status='verified_author_source_label', eligible_external=False) for r in violence])
    write_json(private/'manifest.json', manifest)
    model_dir = ROOT/'models/checkpoints/toxic-phobert-v3'
    protocol = dict(run_id=run_id, created_utc=datetime.now(timezone.utc).isoformat(), seed=42,
        authorization='User requested training on supplied 200 speech and 100 violence clips',
        label_policy='Speech: user-reported weak clip labels, not human-verified segment labels. Violence: verified author labels.',
        split_policy='Speech source-ID + exact SHA + transitive dHash candidate groups; 11 exposed clips forced into train; balance labels only',
        independent_test_caveat='User previously tried an unknown number of clips. Source groups are provisional, speakers may overlap; not blind gold evaluation.',
        violence_policy='All AIRTLab100 repurposed to training as one group; previous external metrics remain historical only. RLVS train/valid/test unchanged.',
        manifest_sha256=sha256(private/'manifest.json'), source_speech_sha256=sha256(SPEECH/'manifest.json'),
        source_airtlab_sha256=sha256(AIRTLAB), rlvs_split_sha256=sha256(ROOT/'data/splits/rlvs-v1.json'),
        model_files={p.name:sha256(p) for p in model_dir.iterdir() if p.is_file()},
        violence_baseline_sha256=sha256(ROOT/'models/checkpoints/violence-r3d18-v3.joblib'),
        asr=dict(model='faster-whisper-small', model_sha256=sha256(ROOT/'models/checkpoints/faster-whisper-small/model.bin'),
                 device='cuda', compute_type='float16', beam_size=5, temperature=0, vad_filter=True,
                 condition_on_previous_text=False, language_gate='vi with probability >= 0.5',
                 caveat='GPU FP16 transcripts may differ from the CPU INT8 app; no claim of production CPU pipeline accuracy'),
        toxic_training=dict(initialization='PhoBERT v3', epochs=5, trainable='last two encoder blocks + classifier',
                            encoder_lr=5e-6, classifier_lr=2e-5, weight_decay=.01, accumulation_clips=8,
                            max_length=192, objective='BCEWithLogitsLoss on max(segment positive-minus-negative logit) per clip; no inherited positive label for every segment',
                            missing_asr='Exclude from probabilistic fitting/metrics and report coverage; retain missing as null',
                            selection='Validation only: precision and recall >= baseline when possible, then macro-F1, F1, lower log loss, earlier epoch; 0.05..0.95 thresholds',
                            rules='Frozen hybrid lexicon unchanged; rules not used to manufacture labels'),
        violence_training=dict(representation='frozen R3D18 temporal mean/max, same as v3',
                               train='RLVS train1157 + AIRTLab100, source labels',
                               configurations=[.001,.01,.1,1.], class_weight='balanced',
                               selection='RLVS validation only: FP <= baseline and recall >= baseline if feasible, then macro-F1/F1',
                               test='Previously reported RLVS test386, never used in selection; AIRTLab metric is training only'),
        deployment='Save candidates separately; do not replace active artifacts automatically based on weak-label evaluation')
    write_json(public/'protocol.json', protocol)
    summary = dict(speech={s:dict(n=sum(r['split']==s for r in rows),
                     class_counts=dict(Counter(str(r['training_label']) for r in rows if r['split']==s)),
                     groups=len({r['training_group'] for r in rows if r['split']==s})) for s in ('train','valid','test')},
                   violence=dict(n=100,positive=50,negative=50,split='train',groups=1),
                   human_reviewed_speech=sum(r['human_label'] is not None for r in rows),
                   forced_exposed_train=len(exposure['excluded_from_future_independent_test']))
    write_json(public/'data_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def load_frozen(run_id):
    private, public = locations(run_id)
    protocol = read(public/'protocol.json')
    if sha256(private/'manifest.json') != protocol['manifest_sha256']:
        raise ValueError('Frozen manifest changed')
    return private, public, protocol, read(private/'manifest.json')


def asr(run_id, test=False):
    import os
    import torch
    private, public, protocol, manifest = load_frozen(run_id)
    if test and not (public/'toxic/selection.json').exists():
        raise ValueError('Freeze model selection before transcribing test')
    torch.set_num_threads(4)
    dll_path = str(Path(torch.__file__).parent/'lib')
    os.environ['PATH'] = dll_path + os.pathsep + os.environ['PATH']
    dll_handle = os.add_dll_directory(dll_path) if os.name == 'nt' else None
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    config = protocol['asr']
    directory = ROOT/'models/checkpoints/faster-whisper-small'
    if sha256(directory/'model.bin') != config['model_sha256']:
        raise ValueError('ASR checkpoint changed')
    model = WhisperModel(str(directory),device=config['device'],compute_type=config['compute_type'],local_files_only=True,cpu_threads=4)
    selected = [r for r in manifest['speech'] if (r['split']=='test') == test]
    start = time.perf_counter()
    for i,row in enumerate(selected,1):
        path = private/'asr'/f"{row['video_id']}.json"
        if path.exists():
            if read(path)['sha256'] != row['sha256']:
                raise ValueError('ASR cache source mismatch')
            continue
        source = ROOT/row['path']
        if sha256(source) != row['sha256']:
            raise ValueError('Speech media changed')
        audio = decode_audio(str(source),sampling_rate=16000)
        chunks,info = model.transcribe(audio,task='transcribe',beam_size=5,temperature=0,
                                      vad_filter=True,condition_on_previous_text=False)
        eligible = info.language=='vi' and info.language_probability>=.5
        segments = [dict(start=s.start,end=s.end,text=s.text,avg_logprob=s.avg_logprob,
                         compression_ratio=s.compression_ratio,no_speech_prob=s.no_speech_prob)
                    for s in chunks if s.text.strip()] if eligible else []
        write_json(path,dict(video_id=row['video_id'],sha256=row['sha256'],language=info.language,
                            language_probability=info.language_probability,segments=segments,
                            configuration=protocol['asr']))
        if i%10==0 or i==len(selected):
            status=dict(stage='test' if test else 'train_valid',completed=i,total=len(selected),
                        elapsed_seconds=time.perf_counter()-start)
            write_json(public/'asr_progress.json',status)
            print(json.dumps(status),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','asr','asr-test'])
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    if args.command=='prepare': prepare(args.run_id)
    else: asr(args.run_id,test=args.command=='asr-test')
