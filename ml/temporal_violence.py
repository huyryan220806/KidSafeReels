"""RLVS temporal feature experiments, with test isolated until selection freezes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from ml.experiment_metrics import binary_metrics, choose_operating_point, threshold_rows, write_csv, write_json, plot_results
from ml.model_experiments import locations, load_visual, learning_curve, plot_thresholds
from ml.temporal_visual import BACKBONE, SAMPLING, TemporalEncoder, prepare_backbone, sample_windows
from ml.toxic_baseline import ROOT, sha256

FRAME_CACHE=ROOT/'data/processed/rlvs-temporal-frames'
FEATURE_CACHE=ROOT/'data/processed/rlvs-temporal-features'


def rows_for(splits):
    rows=json.loads((ROOT/'data/splits/rlvs-v1.json').read_text(encoding='utf-8'))['rows']
    groups={s:{r['group_id'] for r in rows if r['split']==s} for s in ('train','valid','test')}
    assert not(groups['train']&groups['valid'] or groups['train']&groups['test'] or groups['valid']&groups['test'])
    return [r for r in rows if r['split'] in splits]


def frame_path(row):
    key=hashlib.sha256((row['sha256']+SAMPLING).encode()).hexdigest()
    return FRAME_CACHE/f'{key}.npy'


def prepare_row(row):
    cache=frame_path(row)
    if cache.exists():
        stored=np.load(cache,mmap_mode='r',allow_pickle=False)
        if stored.shape!=(5,16,112,112,3) or stored.dtype!=np.uint8:
            raise ValueError('Temporal frame cache is invalid')
        return
    path=ROOT/'data/raw/rlvs'/row['path'].replace('\\','/')
    if sha256(path)!=row['sha256']:
        raise ValueError('RLVS video checksum mismatch')
    windows=sample_windows(path)
    cache.parent.mkdir(parents=True,exist_ok=True)
    with cache.with_suffix('.partial').open('wb') as stream:
        np.save(stream,windows,allow_pickle=False)
    cache.with_suffix('.partial').replace(cache)


def prepare(run_id,test=False):
    _,public=locations(run_id,'violence')
    if test and not (public/'selection.json').exists():
        raise ValueError('Freeze selection before preparing test')
    protocol=dict(run_id=run_id,task='violence',seed=42,sampling=SAMPLING,
        representation='frozen Kinetics400 R3D18 mean/max over5windows; optional concatenated static8 ResNet18 features',
        configurations=[dict(features=f,C=c) for f in ('temporal','static-temporal') for c in (.01,.1,1.)],
        selection_rule='max validation recall, FP <= baseline and macro-F1 >= baseline - .005; baseline remains eligible',
        test_policy='RLVS previously reported benchmark; AIRTLab100 held together as external source-label evaluation only after freezing; no tuning against either',
        data_manifest_sha256=sha256(ROOT/'data/splits/rlvs-v1.json'))
    if not (public/'protocol.json').exists():
        write_json(public/'protocol.json',dict(protocol,started_at=datetime.now().astimezone().isoformat()))
    else:
        saved=json.loads((public/'protocol.json').read_text(encoding='utf-8'))
        if any(saved.get(k)!=v for k,v in protocol.items()):
            raise ValueError('Frozen preparation protocol mismatch')
    rows=rows_for(('test',) if test else ('train','valid'))
    started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,_ in enumerate(pool.map(prepare_row,rows),1):
            if i%50==0:
                write_json(public/'preparation_progress.json',dict(prepared=i,total=len(rows),test=test,
                    elapsed_seconds=time.perf_counter()-started,updated_at=datetime.now().astimezone().isoformat()))
                print(f'temporal frames {i}/{len(rows)} ({time.perf_counter()-started:.1f}s)',flush=True)
    write_json(public/'preparation_progress.json',dict(prepared=len(rows),total=len(rows),test=test,complete=True,
        elapsed_seconds=time.perf_counter()-started,updated_at=datetime.now().astimezone().isoformat()))


def features(splits):
    rows=rows_for(splits); prepare_backbone()
    digest=sha256(BACKBONE); encoder=None; matrices={s:[] for s in splits}
    started=time.perf_counter()
    for i,row in enumerate(rows,1):
        key=hashlib.sha256((row['sha256']+SAMPLING+digest).encode()).hexdigest()
        cache=FEATURE_CACHE/f'{key}.npy'
        if cache.exists():
            feature=np.load(cache,allow_pickle=False)
        else:
            prepare_row(row)
            if encoder is None:
                encoder=TemporalEncoder(device='cuda' if torch.cuda.is_available() else 'cpu')
            windows=np.load(frame_path(row),allow_pickle=False)
            feature=encoder.encode(windows)
            cache.parent.mkdir(parents=True,exist_ok=True)
            with cache.with_suffix('.partial').open('wb') as stream:np.save(stream,feature,allow_pickle=False)
            cache.with_suffix('.partial').replace(cache)
        if feature.shape!=(1024,) or not np.isfinite(feature).all():raise ValueError('Invalid temporal features')
        matrices[row['split']].append(feature)
        if i%100==0:print(f'temporal embeddings {i}/{len(rows)} ({time.perf_counter()-started:.1f}s)',flush=True)
    return {s:dict(x=np.array(matrices[s]),y=np.array([r['label'] for r in rows if r['split']==s]),
                 ids=[r['sha256'] for r in rows if r['split']==s],groups=[r['group_id'] for r in rows if r['split']==s]) for s in splits}


def combine(static,temporal,mode):
    if mode=='baseline':return static
    return {s:dict(d,x=np.concatenate([static[s]['x'],d['x']],axis=1) if mode=='static-temporal' else d['x']) for s,d in temporal.items()}


def select(run_id):
    private,public=locations(run_id,'violence'); private.mkdir(parents=True,exist_ok=True)
    if (public/'selection.json').exists():raise ValueError('Selection frozen')
    if not (public/'protocol.json').exists():raise ValueError('Prepare and freeze protocol first')
    torch.set_num_threads(4)
    base=joblib.load(ROOT/'models/checkpoints/violence-resnet18-v1.joblib')
    static,_=load_visual(needed=('train','valid')); temporal=features(('train','valid'))
    base_metric=binary_metrics(static['valid']['y'],base['model'].predict_proba(static['valid']['x'])[:,1],base['threshold'])
    rows=[dict(candidate='baseline',**base_metric)]; models={'baseline':(base['model'],{'features':'baseline','C':.1},0.)}
    protocol=json.loads((public/'protocol.json').read_text(encoding='utf-8'))
    for i,configuration in enumerate(protocol['configurations'],1):
        name=f'candidate-{i:02d}'; data=combine(static,temporal,configuration['features'])
        model=make_pipeline(StandardScaler(),LogisticRegression(C=configuration['C'],class_weight='balanced',solver='liblinear',max_iter=1000,random_state=42))
        start=time.perf_counter(); model.fit(data['train']['x'],data['train']['y']); elapsed=time.perf_counter()-start
        candidates=[dict(candidate=name,**r) for r in threshold_rows(data['valid']['y'],model.predict_proba(data['valid']['x'])[:,1])]
        rows.extend(candidates); models[name]=(model,configuration,elapsed)
        print(name,configuration,'validation maxF1',max(r['f1'] for r in candidates),flush=True)
    selected=choose_operating_point('violence',rows,base_metric)
    model,config,seconds=models[selected['candidate']]
    artifact=dict(base,model=model,threshold=selected['threshold'],version=f'violence-temporal-{run_id}',configuration=config,
                  sklearn_version=sklearn.__version__)
    if config['features']!='baseline':
        artifact.update(sampling=SAMPLING,temporal_backbone_sha256=sha256(BACKBONE),temporal_backbone='r3d18-kinetics400.pth')
    joblib.dump(artifact,private/'selected.joblib')
    write_csv(public/'validation_search.csv',rows)
    write_json(public/'selection.json',dict(run_id=run_id,selected=selected,configuration=config,fit_seconds=seconds,
               baseline_validation=base_metric,artifact_sha256=sha256(private/'selected.joblib'),
               backbone_provenance=json.loads(BACKBONE.with_suffix('.json').read_text()),
               selected_at=datetime.now().astimezone().isoformat(),test_evaluated=False))
    plot_thresholds(public,json.loads((public/'selection.json').read_text()))
    print(json.dumps(selected,indent=2),flush=True)


def evaluate(run_id):
    private,public=locations(run_id,'violence')
    if (public/'evaluation.json').exists():raise ValueError('Test already evaluated')
    selection=json.loads((public/'selection.json').read_text())
    if sha256(private/'selected.joblib')!=selection['artifact_sha256']:raise ValueError('Frozen artifact changed')
    base=joblib.load(ROOT/'models/checkpoints/violence-resnet18-v1.joblib'); new=joblib.load(private/'selected.joblib')
    static,_=load_visual(); temporal=features(('train','valid','test'))
    data=combine(static,temporal,selection['configuration']['features'])
    rows,predictions,confusion=[],{},[]
    for variant,artifact,datasets in (('baseline',base,static),('selected',new,data)):
        predictions[variant]={}; individual=[]
        for split,d in datasets.items():
            start=time.perf_counter(); scores=artifact['model'].predict_proba(d['x'])[:,1]
            metric=binary_metrics(d['y'],scores,artifact['threshold'])
            rows.append(dict(run_id=run_id,task='violence',dataset='RLVS',variant=variant,model_version=artifact['version'],
                             split=split,predict_batch_seconds=time.perf_counter()-start,**metric))
            predictions[variant][split]=(d['y'],scores)
            confusion.extend(dict(variant=variant,split=split,cell=cell,count=metric[cell]) for cell in ('tn','fp','fn','tp'))
            for sid,group,label,score in zip(d['ids'],d['groups'],d['y'],scores):
                pred=int(score>=artifact['threshold']); outcome=('TP' if pred else 'FN') if label else ('FP' if pred else 'TN')
                individual.append(dict(sample_id=sid,group_id=group,split=split,label=int(label),score=float(score),prediction=pred,outcome=outcome))
        write_csv(private/f'{variant}_predictions.csv',individual)
    history=learning_curve(new['model'],data['train'],data['valid'],new['threshold'])
    write_csv(public/'metrics_by_split.csv',rows); write_csv(public/'comparison.csv',[r for r in rows if r['split']=='test'])
    write_csv(public/'confusion_matrix.csv',confusion); write_csv(public/'learning_curve.csv',history)
    plot_results(public,rows,predictions,history)
    write_json(public/'evaluation.json',dict(run_id=run_id,task='violence',metrics=rows,
        selection_sha256=sha256(public/'selection.json'),evaluated_at=datetime.now().astimezone().isoformat(),
        loss_definition='unweighted binary cross-entropy on predicted probabilities',test_status='previously reported RLVS benchmark'))
    print(json.dumps([r for r in rows if r['split']=='test'],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','prepare-test','select','evaluate'])
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    {'prepare':prepare,'prepare-test':lambda r:prepare(r,test=True),'select':select,'evaluate':evaluate}[args.command](args.run_id)
