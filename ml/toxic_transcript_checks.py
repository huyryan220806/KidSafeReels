"""Diagnostics of the already fitted transcript classifier at its frozen threshold."""
import argparse
import csv
import hashlib
import json

import joblib
import numpy as np

from ml.experiment_metrics import binary_metrics, write_json
from ml.toxic_baseline import DATA, read_split
from ml.toxic_reviewed import ROOT, WRITTEN, read, sha
from ml.toxic_transcript_trial import load_samples


def check(run_id):
    public, private = ROOT/'reports'/run_id, ROOT/'runs'/run_id
    checkpoint = private/'transcript-classifier.joblib'
    if sha(checkpoint) != read(public/'result.json')['model_sha256']:
        raise ValueError('Frozen checkpoint changed')
    model = joblib.load(checkpoint)
    # Match the original ViCTSD cleaning and order without accessing written test.
    train_texts = {text for text, _ in read_split(DATA/'ViCTSD_train.csv')}
    valid_labels = {}
    for text, label in read_split(DATA/'ViCTSD_valid.csv'):
        valid_labels.setdefault(text, set()).add(label)
    samples = [(text, next(iter(labels))) for text, labels in valid_labels.items()
               if len(labels)==1 and text not in train_texts]
    texts, labels = zip(*samples)
    ids = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
    original = read(WRITTEN/'tokens-valid.json')
    if (hashlib.sha256(json.dumps(ids).encode()).hexdigest()!=original['signature']
            or list(labels)!=[row['labels'] for row in original['rows']]):
        raise ValueError('Written validation alignment failed')
    scores = model.predict_proba(texts)[:, 1]
    written = binary_metrics(labels, scores, .5)
    with (private/'predictions.csv').open(encoding='utf-8-sig') as stream:
        pred = {r['video_id']:r for r in csv.DictReader(stream) if r['split']=='valid' and r['method']=='text'}
    valid = [s for s in load_samples() if s['split']=='valid']
    if set(pred)!={s['id'] for s in valid}:
        raise ValueError('Missing validation predictions')
    paired = {}
    for field in ('baseline','selected','rule'):
        changes = {'fixed_fp':0,'new_fp':0,'fixed_fn':0,'new_fn':0}
        for sample in valid:
            candidate = pred[sample['id']]['hybrid_flag']=='True'
            old = sample[field]
            if old!=candidate:
                outcome = ('fixed_fn' if candidate else 'new_fn') if sample['label'] else ('new_fp' if candidate else 'fixed_fp')
                changes[outcome] += 1
        paired[field] = changes
    result = dict(written_valid_model_only=written, speech_validation_changes=paired,
                  model_sha256=sha(checkpoint), n_features=len(model.steps[0][1].vocabulary_))
    write_json(public/'transfer-and-pairs.json', result)
    print(result)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='toxic-transcript-2026-11-20')
    check(parser.parse_args().run_id)
