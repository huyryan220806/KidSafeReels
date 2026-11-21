"""CPU trial on Vietnamese speech. Run: python -m ml.toxic_transcript_trial"""
import argparse
import hashlib
import re
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.experiment_metrics import binary_metrics, write_csv, write_json
from ml.hybrid_evaluation import decision_metrics
from ml.toxic_baseline import normalize
from ml.toxic_reviewed import PRIVATE, PUBLIC, ROOT, load, read, sha
from ml.toxic_rules import ToxicRules, lexical_decision


def features(samples, method):
    if method == 'text':
        return [sample['text'] for sample in samples]
    return [[np.log1p(len(s['text'])), np.log1p(len(s['text'].split())), np.log1p(s['segments'])] for s in samples]


def fit_and_score(train, evaluate, method):
    eligible = [s for s in train if s['text']]
    covered = [s for s in evaluate if s['text']]
    transform = (TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2,
                                 sublinear_tf=True, lowercase=False)
                 if method == 'text' else StandardScaler())
    model = make_pipeline(transform, LogisticRegression(C=1, solver='liblinear', random_state=42))
    model.fit(features(eligible, method), [s['label'] for s in eligible])
    scores = model.predict_proba(features(covered, method))[:, 1] if covered else []
    return model, {s['id']: float(p) for s, p in zip(covered, scores, strict=True)}


def load_samples():
    _, manifest = load()
    rules = ToxicRules()
    selection_hash = sha(PUBLIC / 'selection.json')
    rows = []
    for row in manifest['speech']:
        if row['split'] not in ('train', 'valid'):
            continue
        sid = row['video_id']
        path = PRIVATE / 'asr' / f'{sid}.json'
        trace = read(path)
        if trace['sha256'] != row['sha256']:
            raise ValueError('ASR/media mismatch')
        segments = [s for s in trace['segments'] if normalize(s['text'])]
        flag = any(lexical_decision(rules.match(s['text']), s)['flagged'] for s in segments)
        sample = dict(id=sid, split=row['split'], group=row['training_group'], channel=row['source_channel_id'],
                      label=row['training_label'], text=normalize(' '.join(s['text'] for s in segments)),
                      segments=len(segments), rule=flag)
        for variant in ('baseline', 'selected'):
            cache = read(PRIVATE / 'cpu' / variant / f'{sid}.json')
            if (cache['asr_sha256'], cache['selection_sha256'], cache['video_id'], cache['label'], cache['split'],
                cache['rule_flag'], cache['variant']) != (sha(path), selection_hash, sid, sample['label'],
                                                          sample['split'], flag, variant):
                raise ValueError(f'Comparator mismatch for {sid}')
            sample[variant] = cache['hybrid_flag']
        rows.append(sample)
    return rows


def fold_scores(train, method):
    y = [s['label'] for s in train]
    groups = [s['group'] for s in train]
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for fit_ids, eval_ids in folds.split(train, y, groups):
        fit = [train[i] for i in fit_ids]
        evaluate = [train[i] for i in eval_ids]
        if {s['group'] for s in fit} & {s['group'] for s in evaluate}:
            raise ValueError('Source group leakage')
        _, part = fit_and_score(fit, evaluate, method)
        if scores.keys() & part.keys():
            raise ValueError('Duplicate out-of-fold score')
        scores.update(part)
    return scores


def evaluate_rows(samples, scores, split, method):
    expected = {s['id'] for s in samples if s['text']}
    if set(scores) != expected:
        raise ValueError('Score coverage differs from ASR coverage')
    labels = [s['label'] for s in samples]
    model_flags = [scores.get(s['id']) is not None and scores[s['id']] >= .5 for s in samples]
    flags = [s['rule'] or f for s, f in zip(samples, model_flags)]
    metrics = []
    for name, decisions in [(method, model_flags), (method + '+rule', flags)]:
        metrics.append(dict(split=split, method=name, missing_asr=len(samples)-len(expected),
                            **decision_metrics(labels, decisions)))
    covered = [s for s in samples if s['id'] in expected]
    prob = binary_metrics([s['label'] for s in covered], [scores[s['id']] for s in covered], .5)
    return metrics, dict(split=split, method=method, **prob), flags


def run(run_id):
    if not re.fullmatch('[a-z0-9-]+', run_id):
        raise ValueError('Invalid run ID')
    public, private = ROOT/'reports'/run_id, ROOT/'runs'/run_id
    if public.exists() or private.exists():
        raise ValueError('Choose a new run ID to reproduce this completed trial')
    samples = load_samples()
    train = [s for s in samples if s['split']=='train']
    valid = [s for s in samples if s['split']=='valid']
    if {s['group'] for s in train} & {s['group'] for s in valid}:
        raise ValueError('Source group leakage')
    private.mkdir(parents=True)
    write_json(public/'protocol.json', dict(input_protocol=sha(PUBLIC/'protocol.json'), code_sha256=sha(__file__),
        sklearn=sklearn.__version__, fit='TRAIN only, nonempty Whisper Vietnamese transcripts, one row per clip',
        features='NFC/casefold; char_wb 3-5, min_df 2, sublinear TF-IDF; L2 logistic liblinear C=1',
        threshold=.5, combination='logical OR with deployed lexical rules on original ASR segments',
        control='identical LR on standardized log1p(characters, whitespace tokens, ASR segment count)',
        missing_asr='null model score, count as no Toxic flag (review)', oof='5 stratified folds by training_group, seed 42',
        gate='Hybrid must strictly reduce FP vs both prior validation hybrids without increasing FN',
        validation='fixed prior split; repeated validation; seen/unseen TRAIN channel subgroup diagnostic',
        test_inference=False, deployment='offline candidate'))
    metrics, probabilities, predictions = [], [], []
    train_channels = {s['channel'] for s in train}
    subgroups = []
    for method in ('text', 'length'):
        model, validation_scores = fit_and_score(train, valid, method)
        if method=='text':
            model_path = private/'transcript-classifier.joblib'
            joblib.dump(model, model_path)
            restored = joblib.load(model_path)
            covered = [s for s in valid if s['text']]
            assert np.array_equal(restored.predict_proba(features(covered, method))[:, 1],
                                  [validation_scores[s['id']] for s in covered])
        for split, part, scores in [('train_oof', train, fold_scores(train, method)),
                                    ('valid', valid, validation_scores)]:
            decisions, prob, flags = evaluate_rows(part, scores, split, method)
            metrics.extend(decisions)
            probabilities.append(prob)
            for sample, flag in zip(part, flags):
                predictions.append(dict(split=split, method=method, video_id=sample['id'], label=sample['label'],
                                        score=scores.get(sample['id']), hybrid_flag=bool(flag)))
            if split=='valid':
                for seen in (False, True):
                    pairs = [(s, f) for s, f in zip(part, flags) if (s['channel'] in train_channels)==seen]
                    if pairs:
                        subgroups.append(dict(method=method+'+rule', train_channel_seen=seen,
                            **decision_metrics([s['label'] for s, f in pairs], [f for s, f in pairs])))
    for split, part in [('valid', valid), ('train_oof', train)]:
        for field in (('baseline', 'selected', 'rule') if split=='valid' else ('rule',)):
            metrics.append(dict(split=split, method=field, missing_asr=sum(not s['text'] for s in part),
                                **decision_metrics([s['label'] for s in part], [s[field] for s in part])))
    val = {r['method']:r for r in metrics if r['split']=='valid'}
    candidate = val['text+rule']
    passed = all(candidate['fp']<val[m]['fp'] and candidate['fn']<=val[m]['fn'] for m in ('baseline','selected'))
    weight = model_path.stat().st_size
    write_csv(public/'decision_metrics.csv', metrics)
    write_csv(public/'model_metrics.csv', probabilities)
    write_csv(public/'channel_slices.csv', subgroups)
    write_csv(private/'predictions.csv', predictions)
    write_json(public/'result.json', dict(validation_gate=passed, model_bytes=weight, model_sha256=sha(model_path),
        train_clips=len(train), valid_clips=len(valid), test_inference=False, deployed=False))
    print({k:(v['fp'], v['fn'], round(v['f1'],4)) for k,v in val.items()})


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='toxic-transcript-2026-11-20')
    run(parser.parse_args().run_id)
