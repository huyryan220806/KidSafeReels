"""Verify the supplied AIRTLab pack and reserve it for external source-label evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import requests
import numpy as np
import joblib

from ml.experiment_metrics import binary_metrics, write_csv, write_json
from ml.toxic_baseline import ROOT, sha256

PACK=ROOT/'data/private/violence-drive-2026-10-28'
REPOSITORY='airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos'
REVISION='1f7747e104301ccaa82ef5a2f6804b51ced1c398'


def prepare(run_id):
    from ml.media import fingerprint,sample_frames,inspect_video
    source=json.loads((PACK/'source-manifest.json').read_text(encoding='utf-8'))
    listing=json.loads((PACK/'full-inventory.json').read_text(encoding='utf-8'))
    ids={row['name']:row['id'] for row in listing}
    if source['source_commit']!=REVISION or len(source['clips'])!=100:
        raise ValueError('Unexpected source manifest')
    tree_path=PACK/'official-git-tree.json'
    if not tree_path.exists():
        response=requests.get(f'https://api.github.com/repos/{REPOSITORY}/git/trees/{REVISION}',params={'recursive':'1'},timeout=60)
        response.raise_for_status(); tree=response.json()
        if tree.get('truncated'):raise ValueError('Incomplete official tree')
        write_json(tree_path,tree)
    tree=json.loads(tree_path.read_text())
    official={row['path']:row for row in tree['tree'] if row['type']=='blob'}
    def fetch(row):
        path=row['source_path']; original=official[path]
        label=1 if path.split('/')[-3]=='violent' else 0
        assert path.split('/')[-2]=='cam1' and label==row['source_violence_label']
        assert original['sha']==row['source_git_blob_sha1'] and original['size']==row['size_bytes']
        name=Path(row['video_path']).name
        if name!=row['video_id']+'.mp4' or not row['video_id'].startswith('airtlab-'):
            raise ValueError('Unexpected media filename')
        target=PACK/'videos'/name
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            try:
                response=requests.get('https://drive.google.com/uc',params={'export':'download','id':ids[name]},timeout=60)
                response.raise_for_status()
            except (requests.HTTPError,requests.Timeout):
                # The public author mirror has the exact blob checked below.
                print(f'Drive temporarily unavailable; using verified author blob for {name}',flush=True)
                response=requests.get(f'https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{path}',timeout=60)
                response.raise_for_status()
            if len(response.content)!=original['size']:raise ValueError(f'Drive download size mismatch: {name}')
            temp=target.with_suffix('.partial'); temp.write_bytes(response.content); temp.replace(target)
        payload=target.read_bytes()
        git_blob=hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()
        if git_blob!=original['sha'] or hashlib.sha256(payload).hexdigest()!=row['sha256']:
            raise ValueError(f'Official/source hash mismatch: {name}')
        info=inspect_video(target)
        hashes=fingerprint(sample_frames(target))
        return dict(video_id=row['video_id'],path=str(target.relative_to(ROOT)),sha256=row['sha256'],
                    source_label=label,source_path=path,source_blob_sha1=git_blob,group_id='airtlab-shared-room-recording-context',
                    project_label=None,label_origin='original AIRTLab class directory verified against official git tree',
                    duration_seconds=info.duration,fingerprint=hashes)
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,row in enumerate(pool.map(fetch,source['clips']),1):
            rows.append(row)
            if i%10==0:print(f'AIRTLab verified {i}/100',flush=True)
    previous=json.loads((ROOT/'data/processed/rlvs/manifest.json').read_text(encoding='utf-8'))['rows']
    known={r['sha256'] for r in previous}
    for row in rows:
        matches=[r['sha256'] for r in previous if all((a^b).bit_count()<=10 for a,b in zip(row['fingerprint'],r['fingerprint']))]
        row['exact_rlvs_overlap']=row['sha256'] in known
        row['heuristic_near_rlvs_matches']=matches
        row['eligible_external']=not(row['exact_rlvs_overlap'] or matches)
    write_json(PACK/'verified-manifest.json',dict(source_revision=REVISION,rows=rows,
        use='Entire acquisition context reserved for external source-label evaluation after RLVS selection. Never mixed into train/validation.',
        labels='Author labels verified by file membership and hashes; not independent human KidSafeReels annotations.'))
    eligible=[r for r in rows if r['eligible_external']]
    write_json(ROOT/'reports'/run_id/'airtlab_data_audit.json',dict(recorded_at=datetime.now().astimezone().isoformat(),
        source=f'https://github.com/{REPOSITORY}',revision=REVISION,downloaded=len(rows),
        source_class_counts={str(label):sum(r['source_label']==label for r in rows) for label in (0,1)},
        verified_against_official_git_blobs=True,exact_rlvs_overlaps=sum(r['exact_rlvs_overlap'] for r in rows),
        heuristic_near_overlap_count=sum(bool(r['heuristic_near_rlvs_matches']) for r in rows),eligible_external=len(eligible),
        group_count=1,use='whole context held out; no train/validation mixing',
        label_origin='original AIRTLab dataset class directory, not new user annotations',
        limitation='Staged actions, one room/camera context; heuristic duplicate checks cannot guarantee source independence.',
        manifest_sha256=sha256(PACK/'verified-manifest.json')))
    print(f'AIRTLab prepared: {len(eligible)} eligible external source-label clips',flush=True)


def evaluate(run_id):
    import torch
    from ml.visual import ViolenceClassifier,VisualEncoder
    from ml.temporal_visual import TemporalEncoder,sample_windows
    from ml.media import sample_frames
    public=ROOT/'reports'/run_id; private=ROOT/'runs'/run_id/'violence'
    output=public/'airtlab_evaluation.json'
    if output.exists():raise ValueError('External evaluation already recorded')
    selection=json.loads((public/'violence/selection.json').read_text())
    if sha256(private/'selected.joblib')!=selection['artifact_sha256']:raise ValueError('Frozen model changed')
    artifact=joblib.load(private/'selected.joblib')
    baseline=ViolenceClassifier(); mode=selection['configuration']['features']
    temporal=TemporalEncoder(device='cuda' if torch.cuda.is_available() else 'cpu') if mode!='baseline' else None
    torch.set_num_threads(4)
    data=json.loads((PACK/'verified-manifest.json').read_text())
    rows=[r for r in data['rows'] if r['eligible_external']]
    predictions=[]
    for i,row in enumerate(rows,1):
        path=ROOT/row['path']
        if sha256(path)!=row['sha256']:raise ValueError('External file changed')
        frames=sample_frames(path)
        static=baseline.encoder.encode(frames)
        old=float(baseline.model.predict_proba(static[None])[0,1])
        if temporal:
            feature=temporal.encode(sample_windows(path))
            if mode=='static-temporal':feature=np.concatenate([static,feature])
        else:feature=static
        new=float(artifact['model'].predict_proba(feature[None])[0,1])
        predictions.append(dict(video_id=row['video_id'],source_label=row['source_label'],baseline_score=old,selected_score=new))
        if i%10==0:print(f'AIRTLab frozen inference {i}/{len(rows)}',flush=True)
    write_csv(PACK/f'{run_id}-predictions.csv',predictions)
    labels=[r['source_label'] for r in predictions]
    metrics=[dict(dataset='AIRTLab100-source-labels',variant=variant,split='external',
        **binary_metrics(labels,[r[f'{variant}_score'] for r in predictions],threshold))
        for variant,threshold in (('baseline',baseline.threshold),('selected',artifact['threshold']))]
    write_csv(public/'airtlab_comparison.csv',metrics)
    write_json(output,dict(metrics=metrics,selection_sha256=sha256(public/'violence/selection.json'),
        evaluated_at=datetime.now().astimezone().isoformat(),
        limitations=['Author benchmark labels, not KidSafeReels human labels','One staged acquisition context; not target short-video accuracy',
                    'Entire set held out of fitting/threshold choice; do not retune using these outcomes']))
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','evaluate']); parser.add_argument('--run-id',required=True)
    args=parser.parse_args(); {'prepare':prepare,'evaluate':evaluate}[args.command](args.run_id)
