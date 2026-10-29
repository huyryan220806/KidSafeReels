"""Verify saved candidates and CPU inference without modifying active model artifacts."""
import argparse
import hashlib
import shutil

import joblib
import numpy as np

from ml.experiment_metrics import write_json
from ml.local_adaptation import load_frozen, read
from ml.temporal_visual import BACKBONE, SAMPLING
from ml.toxic_baseline import ROOT, sha256


def verify(run_id):
    from ml.visual import ViolenceClassifier
    private,public,protocol,manifest=load_frozen(run_id)
    for name,digest in protocol['model_files'].items():
        if sha256(ROOT/'models/checkpoints/toxic-phobert-v3'/name)!=digest:
            raise ValueError('Active Toxic baseline changed')
    if sha256(ROOT/'models/checkpoints/violence-r3d18-v3.joblib')!=protocol['violence_baseline_sha256']:
        raise ValueError('Active Violence baseline changed')
    toxic=read(public/'toxic/selection.json');violence=read(public/'violence/selection.json')
    for name,digest in toxic['files'].items():
        if sha256(private/'toxic/selected'/name)!=digest:raise ValueError('Toxic checkpoint changed')
    model_path=private/'violence/selected.joblib'
    if sha256(model_path)!=violence['artifact_sha256']:raise ValueError('Violence checkpoint changed')
    artifact=joblib.load(model_path);digest=sha256(BACKBONE)
    if digest!=artifact['temporal_backbone_sha256']:raise ValueError('Backbone mismatch')
    destination=model_path.parent/artifact['temporal_backbone']
    if not destination.exists():shutil.copy2(BACKBONE,destination)
    if sha256(destination)!=digest:raise ValueError('Candidate backbone copy mismatch')
    samples=[]
    for row in manifest['violence']:
        key=hashlib.sha256((row['sha256']+SAMPLING+digest+'fp32-no-tf32').encode()).hexdigest()
        feature=np.load(private/'violence/features'/f'{key}.npy',allow_pickle=False)
        score=float(artifact['model'].predict_proba(feature[None])[0,1])
        samples.append((abs(score-artifact['threshold']),row,score))
    # Stress the ten training examples nearest the already-frozen threshold.
    chosen=sorted(samples,key=lambda item:(item[0],item[1]['video_id']))[:10]
    model=ViolenceClassifier(model_path);checks=[]
    for _,row,reference in chosen:
        path=ROOT/row['path']
        if sha256(path)!=row['sha256']:raise ValueError('Audit clip changed')
        score=model.score_video(path)
        passed=abs(score-reference)<1e-4 and (score>=model.threshold)==(reference>=model.threshold)
        checks.append(dict(sample_sha256=row['sha256'],gpu_cached_score=reference,cpu_score=score,
                           absolute_difference=abs(score-reference),passed=passed))
    payload=dict(active_baselines_unchanged=True,toxic_files_verified=len(toxic['files']),
                 violence_files_verified=2,violence_cpu_checks=checks,
                 scope='Ten AIRTLab TRAIN clips near frozen threshold; technical parity only, not a test-accuracy estimate',
                 passed=all(r['passed'] for r in checks),
                 max_absolute_difference=max(r['absolute_difference'] for r in checks),
                 deployment='Both candidates saved separately; active app retains v3 + hybrid')
    write_json(public/'artifact_cpu_verification.json',payload)
    print({k:v for k,v in payload.items() if k!='violence_cpu_checks'},flush=True)
    if not payload['passed']:raise ValueError('CPU parity failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True)
    verify(parser.parse_args().run_id)
