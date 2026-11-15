"""Follow up measured audit findings: OOV tokens, ASR timing and loss contribution.

Read-only source; uses dev ASR and validation snapshots, never test content.
"""
import argparse
from collections import Counter

import numpy as np

from ml.audit_validation import drift
from ml.experiment_metrics import write_csv, write_json
from ml.phobert_audit import DEFAULT_RUN, csv_rows, verify
from ml.toxic_baseline import normalize
from ml.toxic_reviewed import BASE, PRIVATE as SOURCE, WRITTEN, read, sha
from ml.toxic_rules import ToxicRules


def probe(run_id):
    from transformers import AutoTokenizer
    from pyvi import ViTokenizer
    private,public,_=verify(run_id)
    tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True);rules=ToxicRules()
    oov=[];timing=[];missing=[];counts=Counter()
    for row in read(SOURCE/'manifest.json')['speech']:
        if row['split']=='test':continue
        trace=read(SOURCE/'asr'/f"{row['video_id']}.json")
        if not trace['segments']:
            missing.append(dict(video_id=row['video_id'],split=row['split'],label=row['training_label'],
                annotated_language=row['annotation']['language'],asr_language=trace.get('language'),
                language_probability=trace.get('language_probability')))
        for i,s in enumerate(trace['segments']):
            if not normalize(s['text']):continue
            text=ViTokenizer.tokenize(normalize(s['text']))
            tokens=tokenizer.tokenize(text);full=tokenizer.encode(text,add_special_tokens=True)
            clipped=tokenizer.encode(text,add_special_tokens=True,truncation=True,max_length=192)
            counts['dev_segments']+=1;counts['full_tokens_including_special']+=len(full)
            counts['unknown_tokens']+=full.count(tokenizer.unk_token_id)
            if tokenizer.unk_token_id in full:
                counts['unknown_segments']+=1
                oov.append(dict(video_id=row['video_id'],segment_index=i,split=row['split'],raw=s['text'],
                    word_segmented=text,bpe_tokens=tokens,token_ids=full,
                    unknown_bpe_tokens=[t for t in tokens if tokenizer.convert_tokens_to_ids(t)==tokenizer.unk_token_id]))
            terms=lambda t:{m['id'] for m in rules.match(t.replace('_',' ')) if m['action']=='flag'}
            before=terms(text);full_terms=terms(tokenizer.decode(full,skip_special_tokens=True))
            clipped_terms=terms(tokenizer.decode(clipped,skip_special_tokens=True))
            counts['lexicon_terms_lost_before_truncation']+=len(before-full_terms)
            counts['lexicon_terms_lost_only_after_truncation']+=len(full_terms-clipped_terms)
            if not 0<=s['start']<=s['end']<=row['duration_seconds']+.05:
                timing.append(dict(video_id=row['video_id'],split=row['split'],segment_index=i,
                    start=s['start'],end=s['end'],duration=row['duration_seconds'],end_overrun=s['end']-row['duration_seconds']))
    write_json(private/'oov-details.json',oov);write_json(private/'timing-issues.json',timing)
    write_json(private/'missing-asr-details.json',missing)
    first=csv_rows(private/'validation/replay_epoch1/clips.csv');last=csv_rows(private/'validation/replay_epoch5/clips.csv')
    group_rows=[]
    for epoch in (1,5):
        rows=csv_rows(private/f'validation/replay_epoch{epoch}/clips.csv')
        for group in ('TN','FP','FN','TP'):
            subset=[r for r in rows if ('TP' if r['prediction'] else 'FN')==group and r['label']==1] if group in ('TP','FN') else [r for r in rows if ('FP' if r['prediction'] else 'TN')==group and r['label']==0]
            if not subset:continue
            y=np.array([r['label'] for r in subset]);p=np.array([r['probability'] for r in subset])
            loss=-np.log(np.clip(np.where(y,p,1-p),np.finfo(float).eps,1))
            group_rows.append(dict(epoch=epoch,outcome=group,n=len(subset),mean_probability=float(p.mean()),
                mean_log_loss=float(loss.mean()),contribution_to_overall_loss=float(loss.sum()/len(rows))))
    write_csv(public/'loss-by-outcome.csv',group_rows)
    segment_changes=[];boundary=[]
    for epoch in range(6):
        path=private/f'validation/replay_epoch{epoch}'
        rows=csv_rows(path/'clips.csv');p=np.array([r['probability'] for r in rows])
        boundary.append(dict(epoch=epoch,min_abs_margin_to_half=float(np.min(np.abs(p-.5))),
            exact_half=int((p==.5).sum()),max_score_below_half=float(p[p<.5].max()),
            min_score_at_or_above_half=float(p[p>=.5].min())))
        if epoch==0:continue
        prior=private/f'validation/replay_epoch{epoch-1}'
        with np.load(prior/'validation.npz',allow_pickle=False) as a,np.load(path/'validation.npz',allow_pickle=False) as b:
            if not np.array_equal(a['segment_ids'],b['segment_ids']):raise ValueError('Segment order drift')
            changes=(a['segment_predictions']!=b['segment_predictions'])
            old=csv_rows(prior/'clips.csv')
            segment_changes.append(dict(before_epoch=epoch-1,after_epoch=epoch,segments=len(changes),
                segment_prediction_flips=int(changes.sum()),segment_flip_rate=float(changes.mean()),
                max_segment_logit_delta=float(np.abs(a['segment_logits']-b['segment_logits']).max()),
                winning_segment_changed=sum(x['winning_segment_id']!=y['winning_segment_id'] for x,y in zip(old,rows))))
    write_csv(public/'segment-drift.csv',segment_changes);write_csv(public/'threshold-margins.csv',boundary)
    written=[]
    for split in ('train','valid'):
        samples=read(WRITTEN/f'tokens-{split}.json')['rows']
        written.append(dict(dataset='ViCTSD',split=split,n=len(samples),labels=dict(Counter(r['labels'] for r in samples))))
    write_json(public/'written-counts.json',written)
    result=dict(source_sha256=sha(__file__),preprocessing=dict(counts),timing_segments=len(timing),
        timing_videos=len({r['video_id'] for r in timing}),
        max_end_overrun_seconds=max(r['end_overrun'] for r in timing),
        timing_caveat='Raw ASR endpoints versus clip duration; production already clamps evidence timestamps, text remains unchanged',
        missing_asr=len(missing),missing_language_counts=dict(Counter(str(r['asr_language']) for r in missing)),
        unknown_token_caveat='Token IDs inspected; an OOV subword is not proof of a validation plateau or a gain from tokenizer replacement',
        epoch1_to_5=drift(first,last),test_accessed=False,
        audit_threshold_scope='No new threshold search; comparisons fixed at 0.5. Exact replay still executes the original validation grid as part of the unchanged trainer.')
    write_json(public/'followup-probes.json',result)
    print(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',default=DEFAULT_RUN)
    probe(parser.parse_args().run_id)
