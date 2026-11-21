"""Adapt the frozen R3D classifier with AIRTLab100; the old external set is now training data."""
import argparse
import hashlib
import time

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from ml.experiment_metrics import binary_metrics, threshold_rows, write_csv, write_json, plot_results
from ml.local_adaptation import load_frozen, read
from ml.temporal_violence import features
from ml.temporal_visual import BACKBONE, SAMPLING, TemporalEncoder, sample_windows
from ml.toxic_baseline import ROOT, sha256


def airtlab_features(private, rows):
    digest=sha256(BACKBONE);encoder=None;matrix=[]
    for i,row in enumerate(rows,1):
        key=hashlib.sha256((row['sha256']+SAMPLING+digest+'fp32-no-tf32').encode()).hexdigest()
        path=private/'features'/f'{key}.npy'
        if path.exists():
            feature=np.load(path,allow_pickle=False)
        else:
            source=ROOT/row['path']
            if sha256(source)!=row['sha256']: raise ValueError('AIRTLab video changed')
            if encoder is None:encoder=TemporalEncoder(device='cuda' if torch.cuda.is_available() else 'cpu')
            feature=encoder.encode(sample_windows(source))
            path.parent.mkdir(parents=True,exist_ok=True);np.save(path,feature,allow_pickle=False)
        if feature.shape!=(1024,) or not np.isfinite(feature).all(): raise ValueError('Bad temporal features')
        matrix.append(feature)
        if i%20==0:print(f'AIRTLab training features {i}/{len(rows)}',flush=True)
    return dict(x=np.array(matrix),y=np.array([r['training_label'] for r in rows]),
                ids=[r['sha256'] for r in rows],groups=[r['group_id'] for r in rows])


def select(run_id):
    private,public,protocol,manifest=load_frozen(run_id)
    private,public=private/'violence',public/'violence'
    if (public/'selection.json').exists():raise ValueError('Already selected')
    if sha256(ROOT/'data/splits/rlvs-v1.json')!=protocol['rlvs_split_sha256']:raise ValueError('RLVS split changed')
    baseline_path=ROOT/'models/checkpoints/violence-r3d18-v3.joblib'
    if sha256(baseline_path)!=protocol['violence_baseline_sha256']:raise ValueError('Baseline changed')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=joblib.load(baseline_path)
    data=features(('train','valid'));air=airtlab_features(private,manifest['violence'])
    x=np.concatenate([data['train']['x'],air['x']]);y=np.concatenate([data['train']['y'],air['y']])
    base_metric=binary_metrics(data['valid']['y'],base['model'].predict_proba(data['valid']['x'])[:,1],base['threshold'])
    candidates,models,fit_log=[],{},[]
    for c in protocol['violence_training']['configurations']:
        started=time.perf_counter()
        model=make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight='balanced',max_iter=2000,random_state=42))
        model.fit(x,y);models[c]=model
        scores=model.predict_proba(data['valid']['x'])[:,1]
        candidates.extend(dict(C=c,**m) for m in threshold_rows(data['valid']['y'],scores))
        fit_log.append(dict(C=c,fit_seconds=time.perf_counter()-started,train_n=len(y),iterations=int(model[-1].n_iter_[0]),
                            train_log_loss=binary_metrics(y,model.predict_proba(x)[:,1],.5)['log_loss']))
        print(f'Violence fit C={c}, n={len(y)}, seconds={fit_log[-1]["fit_seconds"]:.2f}',flush=True)
    eligible=[r for r in candidates if r['fp']<=base_metric['fp'] and r['recall']>=base_metric['recall']]
    chosen=max(eligible or candidates,key=lambda r:(r['macro_f1'] or 0,r['f1'] or 0,-r['fp'],-abs(r['threshold']-.41),-r['C']))
    artifact=dict(base,model=models[chosen['C']],threshold=chosen['threshold'],
                  version=f'violence-temporal-{run_id}',configuration=dict(features='temporal',C=chosen['C']),
                  sklearn_version=sklearn.__version__)
    private.mkdir(parents=True,exist_ok=True);joblib.dump(artifact,private/'selected.joblib')
    write_csv(public/'validation_search.csv',candidates);write_csv(public/'fit_history.csv',fit_log)
    write_json(public/'selection.json',dict(selected=chosen,eligible=bool(eligible),baseline_validation=base_metric,
        artifact_sha256=sha256(private/'selected.joblib'),protocol_sha256=sha256(public.parent/'protocol.json'),
        train_rlvs=len(data['train']['y']),train_airtlab=len(air['y']),test_evaluated=False,
        historical_external_set_now_training=True,deployment='candidate_only'))


def evaluate(run_id):
    private,public,protocol,manifest=load_frozen(run_id)
    private,public=private/'violence',public/'violence'
    if (public/'evaluation.json').exists():raise ValueError('Test already evaluated')
    selection=read(public/'selection.json')
    if sha256(private/'selected.joblib')!=selection['artifact_sha256']:raise ValueError('Candidate changed')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=features(('train','valid','test'));air=airtlab_features(private,manifest['violence'])
    metrics,training,predictions,individual=[],[],{},[]
    for variant,path in (('baseline',ROOT/'models/checkpoints/violence-r3d18-v3.joblib'),('selected',private/'selected.joblib')):
        artifact=joblib.load(path);threshold=artifact['threshold'];predictions[variant]={}
        for split,d in data.items():
            scores=artifact['model'].predict_proba(d['x'])[:,1]
            predictions[variant][split]=(d['y'],scores)
            metrics.append(dict(task='violence',dataset='RLVS',variant=variant,split=split,**binary_metrics(d['y'],scores,threshold)))
            individual.extend(dict(sample_id=sid,split=split,variant=variant,label=int(y),score=float(p))
                              for sid,y,p in zip(d['ids'],d['y'],scores))
        air_scores=artifact['model'].predict_proba(air['x'])[:,1]
        training.append(dict(task='violence',dataset='AIRTLab100-repurposed-train',variant=variant,split='train',
                             **binary_metrics(air['y'],air_scores,threshold)))
    write_csv(private/'predictions.csv',individual)
    write_csv(public/'metrics_by_split.csv',metrics)
    write_csv(public/'airtlab_training_only.csv',training)
    plot_results(public,metrics,predictions,[])
    write_json(public/'evaluation.json',dict(metrics=metrics,airtlab_training_only=training,
        selection_sha256=sha256(public/'selection.json'),
        limitation='RLVS previously reported benchmark. AIRTLab is now fitting data, never external test for this candidate.',
        backbone='Frozen R3D18; only the standard scaler and logistic classifier were fitted',deployment='candidate_only'))
    print([r for r in metrics if r['split']=='test'],flush=True)
    print(training,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['select','evaluate']);parser.add_argument('--run-id',required=True)
    args=parser.parse_args();{'select':select,'evaluate':evaluate}[args.command](args.run_id)
