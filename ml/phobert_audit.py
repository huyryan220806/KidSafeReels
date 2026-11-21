"""Standalone audit of the existing speech PhoBERT trainer; never infer on test.

python -m ml.phobert_audit all --device cuda
Outputs are additive. Test ASR is accessed only for integrity/duplicate counts.
Replay calls the unchanged trainer in a scoped sandbox of module paths, not a new trainer.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np

from ml.audit_validation import aligned, drift, metrics, record
from ml.experiment_metrics import clean_svg, write_csv, write_json
from ml.toxic_baseline import normalize
from ml.toxic_reviewed import BASE, ROOT, PRIVATE as SOURCE, PUBLIC as SOURCE_REPORT, WRITTEN, load, read, sha

DEFAULT_RUN='phobert-audit-2026-11-15'
CODE_FILES=['ml/toxic_reviewed.py','ml/toxic_reviewed_train.py','ml/speech_mil.py',
    'ml/deep_toxic.py','ml/deep_inference.py','ml/toxic_baseline.py','ml/experiment_metrics.py',
    'ml/local_adaptation.py','ml/speech_intake.py','ml/toxic_precision.py','ml/inference.py',
    'ml/asr_recovery.py','ml/toxic_rules.py','ml/audit_validation.py','ml/phobert_audit.py']


def locations(run_id):
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in run_id):
        raise ValueError('Use a lowercase run ID')
    return ROOT/'runs'/run_id, ROOT/'reports'/run_id


def digest_json(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def overlaps(rows, field):
    values=defaultdict(list)
    for row in rows:
        value=row.get(field)
        if value is not None and value!='': values[str(value)].append(row)
    shared=[dict(value=k,splits=sorted({r['split'] for r in group}),
        video_ids=[r['video_id'] for r in group]) for k,group in values.items() if len({r['split'] for r in group})>1]
    return dict(field=field,unique_nonmissing=len(values),
        missing=sum(r.get(field) in (None,'') for r in rows),cross_split_values=len(shared),
        pairs={f'{a}__{b}':sum(a in r['splits'] and b in r['splits'] for r in shared)
               for a,b in [('train','valid'),('train','test'),('valid','test')]}),shared


def duplicate_audit(items):
    groups=defaultdict(list)
    for item in items:
        text=normalize(item['text'])
        if text: groups[hashlib.sha256(text.encode()).hexdigest()].append(item)
    repeated=[dict(text_sha256=h,items=[{k:v for k,v in r.items() if k!='text'} for r in g],
                   cross_split=len({r['split'] for r in g})>1,
                   clip_label_disagreement=len({r['label'] for r in g})>1)
              for h,g in groups.items() if len(g)>1]
    return dict(nonempty_records=sum(len(g) for g in groups.values()),unique_texts=len(groups),
        duplicate_groups=len(repeated),duplicate_excess=sum(len(g['items'])-1 for g in repeated),
        cross_split_groups=sum(g['cross_split'] for g in repeated),
        clip_label_disagreement_groups=sum(g['clip_label_disagreement'] for g in repeated),
        cross_split_label_disagreement_groups=sum(g['cross_split'] and g['clip_label_disagreement'] for g in repeated)),repeated


def source_inventory():
    paths=[SOURCE/'manifest.json',SOURCE/'annotations-original.json',SOURCE_REPORT/'protocol.json',
           SOURCE_REPORT/'training_history.csv',SOURCE/'toxic/resume.pt',SOURCE/'toxic/best.pt',
           SOURCE/'toxic/baseline.npz',SOURCE/'toxic/selected-dev.npz',SOURCE/'toxic/dev-ids.json',
           SOURCE_REPORT/'selection.json',ROOT/'models/artifacts.json']
    paths += list((SOURCE/'asr').glob('*.json'))
    manifest=read(SOURCE/'manifest.json')['speech']
    paths += [SOURCE/'cpu'/variant/f"{r['video_id']}.json" for variant in ('baseline','selected')
              for r in manifest if r['split']!='test']
    paths += [WRITTEN/'tokens-train.json',WRITTEN/'tokens-valid.json',ROOT/'ml/resources/toxic_lexicon_v1.json']
    paths += [p for d in (BASE,SOURCE/'toxic/selected') for p in d.iterdir() if p.is_file()]
    paths += [ROOT/p for p in CODE_FILES]
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}


def cache_audit(rows):
    """Inspect development caches by IDs, never load test predictions."""
    selection=sha(SOURCE_REPORT/'selection.json');ids=read(SOURCE/'toxic/dev-ids.json')
    mismatches=[];count=0;max_delta=0.;order={}
    for variant,name in [('baseline','baseline.npz'),('selected','selected-dev.npz')]:
        directory=BASE if variant=='baseline' else SOURCE/'toxic/selected'
        threshold=read(directory/'kidsafe-config.json')['threshold']
        with np.load(SOURCE/'toxic'/name,allow_pickle=False) as cache:
            for split in ('train','valid'):
                selected=[r for r in rows if r['split']==split]
                covered=[]
                for row in selected:
                    trace=read(SOURCE/'asr'/f"{row['video_id']}.json")
                    result=read(SOURCE/'cpu'/variant/f"{row['video_id']}.json");count+=1
                    if trace['segments']:covered.append(row['video_id'])
                    checks=dict(video_id=result['video_id']==row['video_id'],split=result['split']==split,
                        label=result['label']==row['training_label'],variant=result['variant']==variant,
                        threshold=result['threshold']==threshold,selection=result['selection_sha256']==selection,
                        asr=result['asr_sha256']==sha(SOURCE/'asr'/f"{row['video_id']}.json"),
                        coverage=(result['score'] is not None)==bool(trace['segments']),
                        model_decision=result['model_flag']==(result['score'] is not None and result['score']>=threshold),
                        hybrid=result['hybrid_flag']==(result['model_flag'] or result['rule_flag']))
                    if not all(checks.values()):mismatches.append(dict(video_id=row['video_id'],variant=variant,checks=checks))
                order[f'{variant}_{split}']=covered==ids[split] and len(cache[split])==len(covered)
                if not order[f'{variant}_{split}']:raise ValueError('Source NPZ order/count mismatch')
                cpu=[read(SOURCE/'cpu'/variant/f'{sid}.json')['score'] for sid in ids[split]]
                max_delta=max(max_delta,float(np.max(np.abs(np.asarray(cpu)-cache[split]))))
    return dict(dev_cache_records_checked=count,mismatched_records=len(mismatches),
        npz_order_matches=order,max_cpu_vs_gpu_cache_delta=max_delta,test_predictions_opened=False),mismatches


def verify(run_id):
    private,public=locations(run_id)
    protocol=read(public/'protocol.json')
    if sha(private/'source-inventory.json')!=protocol['source_inventory_sha256']:
        raise ValueError('Source inventory changed')
    for path,h in read(private/'source-inventory.json').items():
        if sha(ROOT/path)!=h: raise ValueError(f'Audit source changed: {path}')
    return private,public,protocol


def data(run_id):
    from ml.local_adaptation import SPEECH, split_speech
    from ml.toxic_rules import ToxicRules
    from transformers import AutoTokenizer
    from pyvi import ViTokenizer
    private,public=locations(run_id)
    if private.exists() or public.exists(): raise ValueError('Audit exists; resume its next stage or choose a new run ID')
    source_protocol,manifest=load(); rows=manifest['speech']
    private.mkdir(parents=True);public.mkdir(parents=True)
    inventory=source_inventory();write_json(private/'source-inventory.json',inventory)
    write_json(public/'protocol.json',dict(run_id=run_id,created_utc=datetime.now(timezone.utc).isoformat(),
        source_run='toxic-human-2026-11-13',source_inventory_sha256=sha(private/'source-inventory.json'),
        inference_splits=['valid'],test_usage='Only label/count/hash and normalized ASR duplicate integrity audit; no inference, metrics or tuning',
        primary_threshold=.5,threshold_search=False,replay='Unchanged existing train function; isolated paths; epoch recorder only',
        original_immutable=True,runtime_changed=False))
    for name in CODE_FILES:
        target=private/'source-code'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    split_rows=[];overlap_details={}
    for field in ['video_id','original_video_id','sha256','source_group_id','training_group',
                  'speaker_id','source_channel_id','query_speaker_group']:
        info,detail=overlaps(rows,field);split_rows.append(info);overlap_details[field]=detail
    original=read(SPEECH/'manifest.json')['rows'];pairs=read(SPEECH/'duplicate_review.json')['pairs']
    exposed=read(SPEECH/'development-exposure-user-failures-2026-10-28.json')['excluded_from_future_independent_test']
    regenerated=split_speech(original,pairs,exposed)
    current={r['video_id']:r for r in rows}
    split_differences=[r['video_id'] for r in regenerated if any(r[k]!=current[r['video_id']][k] for k in ('split','training_group'))]
    tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True);rules=ToxicRules()
    counts=[];clip_texts=[];segment_texts=[];preprocessing=[];issues=[]
    media_errors=[];asr_errors=[];mapping_errors=[]
    for split in ('train','valid','test'):
        part=[r for r in rows if r['split']==split];covered=[];segment_count=0
        for row in part:
            if sha(ROOT/row['path'])!=row['sha256']:media_errors.append(row['video_id'])
            trace=read(SOURCE/'asr'/f"{row['video_id']}.json")
            if trace['sha256']!=row['sha256']:asr_errors.append(row['video_id'])
            if type(row['training_label']) is not int or row['training_label'] not in (0,1) or row['training_label']!=row['annotation']['toxic_language']:
                mapping_errors.append(row['video_id'])
            segments=[s for s in trace['segments'] if normalize(s['text'])]
            if segments: covered.append(row)
            segment_count+=len(segments)
            clip_texts.append(dict(video_id=row['video_id'],split=split,label=row['training_label'],text=' '.join(s['text'] for s in segments)))
            for index,s in enumerate(segments):
                sid=f"{row['video_id']}:asr:{index:04d}"
                segment_texts.append(dict(video_id=row['video_id'],segment_id=sid,split=split,label=row['training_label'],text=s['text']))
                if split=='test':continue
                normal=normalize(s['text']);segmented=ViTokenizer.tokenize(normal)
                full=tokenizer(segmented,add_special_tokens=True,truncation=False)['input_ids']
                clipped=tokenizer(segmented,add_special_tokens=True,truncation=True,max_length=192)['input_ids']
                decoded=tokenizer.decode(clipped,skip_special_tokens=True).replace('_',' ')
                terms=lambda text:{m['id'] for m in rules.match(text) if m['action']=='flag'}
                raw=terms(s['text']);norm=terms(normal);word=terms(segmented.replace('_',' '));kept=terms(decoded)
                audit=dict(video_id=row['video_id'],segment_id=sid,split=split,video_label=row['training_label'],
                    raw_chars=len(s['text']),normalized_chars=len(normal),full_tokens=len(full),used_tokens=len(clipped),
                    truncated=len(full)>192,lost_after_normalization=len(raw-norm),lost_after_word_segmentation=len(norm-word),
                    lost_after_tokenization_truncation=len(word-kept),raw_toxic_terms=len(raw),
                    label_scope='video only; no human segment gold',timing_valid=0<=s['start']<=s['end']<=row['duration_seconds']+.05)
                preprocessing.append(audit)
                if audit['truncated'] or any(audit[k] for k in ('lost_after_normalization','lost_after_word_segmentation','lost_after_tokenization_truncation')):
                    issues.append(dict(audit,raw=s['text'],normalized=normal,word_segmented=segmented,decoded=decoded,
                        lost_rule_ids=sorted(word-kept)))
        counts.append(dict(split=split,videos=len(part),unique_video_ids=len({r['video_id'] for r in part}),
            source_videos=len({r['source_group_id'] for r in part}),training_groups=len({r['training_group'] for r in part}),
            negative=sum(r['training_label']==0 for r in part),positive=sum(r['training_label']==1 for r in part),
            asr_covered_videos=len(covered),missing_asr=len(part)-len(covered),segments=segment_count,
            eligible_negative=sum(r['training_label']==0 for r in covered),eligible_positive=sum(r['training_label']==1 for r in covered)))
    dup_clip,clip_details=duplicate_audit(clip_texts);dup_segment,segment_details=duplicate_audit(segment_texts)
    caches,cache_issues=cache_audit(rows)
    write_json(public/'cache-audit.json',caches);write_json(private/'cache-issues.json',cache_issues)
    write_json(private/'overlap-details.json',overlap_details)
    write_json(private/'duplicate-details.json',dict(clips=clip_details,segments=segment_details))
    write_json(private/'preprocessing-issues.json',issues);write_csv(private/'preprocessing.csv',preprocessing)
    write_csv(public/'counts.csv',counts)
    write_csv(public/'overlaps.csv',[{k:(json.dumps(v) if isinstance(v,dict) else v) for k,v in r.items()} for r in split_rows])
    result=dict(counts=counts,overlaps=split_rows,split_regeneration_mismatches=len(split_differences),
        media_hash_mismatches=len(media_errors),asr_media_hash_mismatches=len(asr_errors),label_mapping_errors=len(mapping_errors),
        changed_from_source_labels=sum(r['training_label']!=r['previous_weak_label'] for r in rows),
        legacy_human_label_null=sum(r['human_label'] is None for r in rows),
        effective_label_field='training_label == annotation.toxic_language',speaker_identity_verified=False,
        duplicate_clips=dup_clip,duplicate_segments=dup_segment,
        duplicate_caveat='Identical ASR text can be generic filler/hallucination; segment clip-label disagreement is not segment-label conflict',
        preprocessing=dict(segments=len(preprocessing),max_length=192,
            truncated=sum(r['truncated'] for r in preprocessing),max_full_tokens=max(r['full_tokens'] for r in preprocessing),
            lost_after_normalization=sum(r['lost_after_normalization'] for r in preprocessing),
            lost_after_word_segmentation=sum(r['lost_after_word_segmentation'] for r in preprocessing),
            lost_after_tokenization_truncation=sum(r['lost_after_tokenization_truncation'] for r in preprocessing),
            invalid_segment_timing=sum(not r['timing_valid'] for r in preprocessing)),
        human_full_transcripts=sum(bool(r['annotation']['transcript_verified']) for r in rows),wer=None,cer=None,
        historical_checkpoints_available=[0,2,5],historical_checkpoints_missing=[1,3,4],
        source_unchanged=True)
    write_json(public/'data-audit.json',result)
    write_json(public/'progress.json',dict(stage='data_complete',next='checkpoints'))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def validation_inputs():
    from transformers import AutoTokenizer
    from ml.speech_mil import encode_clips
    _,manifest=load()
    tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    clips,coverage=encode_clips(SOURCE,manifest,tokenizer,('valid',))
    ids=[x['row']['video_id'] for x in clips['valid']]
    if ids!=read(SOURCE/'toxic/dev-ids.json')['valid']:
        raise ValueError('Validation order differs from source cache')
    return clips['valid']


def configure(device):
    import torch
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Use .venv-train or --device cpu')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def checkpoints(run_id,device):
    import torch
    from transformers import AutoModelForSequenceClassification
    private,public,_=verify(run_id);configure(device);clips=validation_inputs()
    snapshots=[]
    for name,epoch,directory,cache in [('original_epoch0',0,BASE,'baseline.npz'),
                                     ('original_epoch2',2,SOURCE/'toxic/selected','selected-dev.npz'),
                                     ('original_epoch5',5,BASE,None)]:
        destination=private/'validation'/name
        if (destination/'metadata.json').exists():snapshots.append(read(destination/'metadata.json'));continue
        model=AutoModelForSequenceClassification.from_pretrained(directory,local_files_only=True).to(device)
        if epoch==5:
            state=torch.load(SOURCE/'toxic/resume.pt',map_location='cpu',weights_only=True)
            if state['epoch']!=5:raise ValueError('Expected historical final epoch 5')
            model.load_state_dict(state['model'],strict=True);del state
        expected=None
        if cache:
            with np.load(SOURCE/'toxic'/cache,allow_pickle=False) as saved:expected=saved['valid'].copy()
        result=record(model,clips,destination,device=device,origin='original checkpoint, new validation export',
            checkpoint=str((SOURCE/'toxic/resume.pt' if epoch==5 else directory/'model.safetensors').relative_to(ROOT)),epoch=epoch,expected=expected)
        snapshots.append(result);del model
        if device=='cuda':torch.cuda.empty_cache()
        print(f'Exported historical epoch {epoch}: {result["metrics"]}',flush=True)
    write_json(public/'checkpoint-audit.json',snapshots)
    write_json(public/'progress.json',dict(stage='historical_checkpoints_complete',next='replay'))


def replay(run_id):
    import torch
    import ml.toxic_reviewed_train as trainer
    private,public,_=verify(run_id);configure('cuda')
    if (public/'replay-comparison.json').exists(): raise ValueError('Replay audit complete, preserve results')
    source_protocol,manifest=load()
    replay_private=private/'replay';replay_public=private/'replay-reports'
    replay_private.mkdir(exist_ok=True);replay_public.mkdir(exist_ok=True)
    if (replay_public/'selection.json').exists():
        raise ValueError('Replay trainer complete; use report, do not train again')
    manifest={'speech':[r for r in manifest['speech'] if r['split']!='test']}
    for row in manifest['speech']:
        target=replay_private/'asr'/f"{row['video_id']}.json";target.parent.mkdir(exist_ok=True)
        source=SOURCE/'asr'/target.name
        if target.exists() and sha(target)!=sha(source):raise ValueError('Replay ASR changed')
        if not target.exists():shutil.copyfile(source,target)
    write_json(replay_public/'protocol.json',source_protocol)
    schedule=[];state_path=replay_private/'toxic/resume.pt';start=1
    if state_path.exists():
        state=torch.load(state_path,map_location='cpu',weights_only=True);start=state['epoch']+1;del state
    if not (replay_private/'toxic/baseline.npz').exists():schedule.append(0)
    schedule.extend(range(start,6));schedule.append('selected')
    original_predict=trainer.predict
    def observed_predict(model,clips,device):
        values=original_predict(model,clips,device)
        if clips and all(x['row']['split']=='valid' for x in clips):
            epoch=schedule.pop(0);name=f'replay_epoch{epoch}' if isinstance(epoch,int) else 'replay_selected'
            checkpoint=BASE/'model.safetensors' if epoch==0 else private/'epoch-checkpoints'/f'{name}.pt'
            if epoch!=0:
                checkpoint.parent.mkdir(exist_ok=True);trainer.atomic_save(model.state_dict(),checkpoint)
            record(model,clips,private/'validation'/name,device=device,origin='controlled replay; not original historical output',
                checkpoint=str(checkpoint.relative_to(ROOT)),epoch=epoch,expected=values)
            print(f'Audit recorded {name}',flush=True)
        return values
    # Scope redirection to the audit process; original files and production code are unchanged.
    with patch.object(trainer,'PRIVATE',replay_private),patch.object(trainer,'PUBLIC',replay_public),\
         patch.object(trainer,'RUN',run_id+'-replay'),patch.object(trainer,'load',lambda:(source_protocol,manifest)),\
         patch.object(trainer,'predict',observed_predict):
        trainer.train()
    if schedule:raise ValueError('Not all expected epochs were recorded')
    compare_replay(run_id)


def csv_rows(path):
    import csv
    with path.open(encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream))
    for r in rows:
        for k in ('label','prediction'):r[k]=int(r[k])
        r['probability']=float(r['probability'])
    return rows


def compare_replay(run_id):
    import torch
    from safetensors.torch import load_file
    private,public,_=verify(run_id)
    differences=[]
    for epoch in (0,2,5):
        before=csv_rows(private/'validation'/f'original_epoch{epoch}'/'clips.csv')
        after=csv_rows(private/'validation'/f'replay_epoch{epoch}'/'clips.csv')
        differences.append(dict(epoch=epoch,**drift(before,after)))
    tensor_comparison=[]
    for epoch in (2,5):
        original=load_file(str(SOURCE/'toxic/selected/model.safetensors')) if epoch==2 else torch.load(SOURCE/'toxic/resume.pt',map_location='cpu',weights_only=True)['model']
        replayed=torch.load(private/'epoch-checkpoints'/f'replay_epoch{epoch}.pt',map_location='cpu',weights_only=True)
        if original.keys()!=replayed.keys():raise ValueError('Checkpoint tensor names differ')
        changed=[];max_delta=0.
        for name in original:
            if not torch.equal(original[name],replayed[name]):
                changed.append(name);max_delta=max(max_delta,float((original[name]-replayed[name]).abs().max()))
        tensor_comparison.append(dict(epoch=epoch,tensors=len(original),changed_tensors=len(changed),max_weight_delta=max_delta))
        del original,replayed
    write_json(public/'replay-comparison.json',dict(probabilities=differences,weights=tensor_comparison,
        missing_original_epochs=[1,3,4],warning='Even exact retained-checkpoint matches do not turn replay files into historical logs'))


def report(run_id):
    import csv
    private,public,_=verify(run_id)
    if not (public/'replay-comparison.json').exists() and (private/'replay-reports/selection.json').exists():compare_replay(run_id)
    records=[];metadata={}
    for folder in sorted((private/'validation').iterdir()):
        if not (folder/'metadata.json').exists():continue
        info=read(folder/'metadata.json');metadata[folder.name]=info
        records.append(dict(snapshot=folder.name,origin=info['origin'],epoch=info['epoch'],**info['metrics']))
    write_csv(public/'validation-metrics.csv',records)
    changes=[]
    for a,b in [('original_epoch0','original_epoch2'),('original_epoch2','original_epoch5')]+[(f'replay_epoch{i}',f'replay_epoch{i+1}') for i in range(5)]:
        if a in metadata and b in metadata:
            changes.append(dict(before=a,after=b,**drift(csv_rows(private/'validation'/a/'clips.csv'),csv_rows(private/'validation'/b/'clips.csv'))))
    write_csv(public/'probability-drift.csv',changes)
    history=list(csv.DictReader((SOURCE_REPORT/'training_history.csv').open(encoding='utf-8-sig')))
    comparisons=[]
    for row in history:
        if row['split']!='valid':continue
        epoch=int(row['epoch']);m=metadata.get(f'replay_epoch{epoch}')
        if not m:continue
        current=m['metrics']
        comparisons.append(dict(epoch=epoch,history_accuracy=float(row['accuracy']),audit_accuracy=current['accuracy'],
            history_binary_f1=float(row['f1']),audit_binary_f1=current['toxic_f1'],
            history_loss=float(row['log_loss']),audit_loss=current['log_loss'],
            loss_delta=current['log_loss']-float(row['log_loss']),
            history_ap=float(row['average_precision']),audit_ap=current['average_precision']))
    write_csv(public/'history-comparison.csv',comparisons)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[metadata[f'replay_epoch{i}']['metrics'] for i in range(1,6)]
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for key in ('log_loss',): axes[0].plot(range(1,6),[r[key] for r in rows],'o-',label=key)
    for key in ('accuracy','toxic_f1','macro_f1','average_precision'):
        axes[1].plot(range(1,6),[r[key] for r in rows],'o-',label=key)
    for key in ('tn','fp','fn','tp'):axes[2].plot(range(1,6),[r[key] for r in rows],'o-',label=key)
    for ax in axes:ax.set_xlabel('Epoch tái chạy');ax.legend();ax.grid(alpha=.2)
    axes[0].set_title('Validation log-loss');axes[1].set_title('Chỉ số phân loại và AP');axes[2].set_title('Confusion matrix (số clip)')
    fig.suptitle('Audit validation 28 clip · ngưỡng 0,5 · tái chạy riêng, không chấm test')
    for ext in ('png','svg'):fig.savefig(public/f'validation-audit.{ext}',dpi=160)
    clean_svg(public/'validation-audit.svg');plt.close(fig)
    write_json(public/'progress.json',dict(stage='audit_computed',snapshots=len(records),
        test_inference=False,source_inputs_verified_unchanged=True))
    write_markdown(public,records,changes)
    print('Audit metrics:',[(r['snapshot'],r['accuracy'],r['toxic_f1'],r['average_precision']) for r in records],flush=True)


def write_markdown(public,records,changes):
    data=read(public/'data-audit.json');cache=read(public/'cache-audit.json')
    lines=['# Audit PhoBERT lời nói — kết quả tự sinh','',
        'Pipeline và dữ liệu gốc được giữ nguyên. Chỉ inference validation; test chỉ dùng kiểm count/hash/trùng ASR, không dùng tuning.',
        '', '## Số lượng và nhãn','',
        '| Split | Video | Nguồn video | Nhóm train | Nhãn 0 / 1 | Có ASR | Segments |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in data['counts']:
        lines.append(f"| {r['split']} | {r['videos']} | {r['source_videos']} | {r['training_groups']} | {r['negative']} / {r['positive']} | {r['asr_covered_videos']} | {r['segments']} |")
    lines+=['','## Kiểm chứng split, nhãn và cache','',
        f"- Tái tạo split từ code gốc: {data['split_regeneration_mismatches']} sai khác; media SHA sai: {data['media_hash_mismatches']}; mapping nhãn sai: {data['label_mapping_errors']}."]
    for r in data['overlaps']:
        lines.append(f"- `{r['field']}`: {r['unique_nonmissing']} giá trị, {r['missing']} mục thiếu, {r['cross_split_values']} giá trị xuất hiện ở nhiều split.")
    lines += ['- `speaker_id` chưa được xác minh. Kênh và từ khóa tên người nói là proxy; không chứng minh speaker-disjoint.',
        f"- Cache dev: {cache['dev_cache_records_checked']} bản ghi kiểm theo ID/nhãn/split/ngưỡng/hash, {cache['mismatched_records']} bản ghi lệch; thứ tự NPZ khớp.",
        '- Nhãn dùng thực tế: `training_label == annotation.toxic_language`. `human_label`/`label_origin` cũ còn giữ để truy vết; không dùng làm target.',
        '', '## Trùng transcript và preprocessing','',
        f"- ASR toàn clip: {data['duplicate_clips']}",f"- ASR từng đoạn: {data['duplicate_segments']}",
        '- Bất đồng nhãn clip trên cùng ASR không tự chứng minh gán nhãn sai; đoạn bình thường có thể nằm trong clip Toxic. ASR boilerplate cũng có thể trùng.',
        f"- Preprocessing train/validation: {data['preprocessing']}",
        '- NFC + casefold + gộp khoảng trắng → PyVi 0.1.1 → tokenizer PhoBERT; giới hạn 192 token gồm special tokens. Không dịch sang tiếng Anh.',
        '- Không có transcript đầy đủ được người nghe xác minh; chưa tính WER/CER. Không gán nhãn video thành nhãn gold từng đoạn.',
        '', '## Logits và metrics validation','',
        'P/R/F1 dưới đây là lớp Toxic=1. Macro-F1 lấy trung bình 2 lớp, weighted-F1 theo support. AP tính từ điểm liên tục. Ngưỡng cố định 0,5; gộp max ở cấp video.',
        '', '| Snapshot | Loss | Accuracy | Toxic P | Toxic R | Toxic F1 | Macro-F1 | Weighted-F1 | AP | TN/FP/FN/TP |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in records:
        values=' | '.join(f"{r[k]:.6f}" for k in ['log_loss','accuracy','toxic_precision','toxic_recall','toxic_f1','macro_f1','weighted_f1','average_precision'])
        lines.append(f"| {r['snapshot']} | {values} | {r['tn']}/{r['fp']}/{r['fn']}/{r['tp']} |")
    lines+=['','## Probability drift','',
        '| Từ → đến | Mean absolute drift | Max drift | Flips | Flip rate |',
        '|---|---:|---:|---:|---:|']
    for r in changes:
        lines.append(f"| {r['before']} → {r['after']} | {r['mean_absolute_probability_drift']:.8f} | {r['max_absolute_probability_drift']:.8f} | {r['prediction_flips']} | {r['flip_rate']:.4%} |")
    lines+=['','![Validation audit](validation-audit.png)','',
        '## Những gì xác minh được và giới hạn','',
        '- Run cũ có checkpoint epoch 2 và resume epoch 5; epoch 1/3/4 không còn checkpoint riêng. `replay_*` là lần tái chạy riêng bằng chính hàm train, không giả là log lịch sử.',
        '- Kiểm checkpoint/shape `[segments, 2]`, `model.eval()` và `torch.inference_mode()` thực tế nằm trong `checkpoint-audit.json` và metadata private. So trọng số/điểm lần tái chạy với checkpoint gốc tại `replay-comparison.json`.',
        '- Loss speech = BCEWithLogits trên max(z1-z0) theo video, thêm 0,5 CE từ ViCTSD train. Gradient speech chỉ đi qua đoạn thắng max; tác hại đến chất lượng là giả thuyết cần thí nghiệm.',
        '- Chỉ số history được tính trên toàn validation sau epoch, không lấy trung bình F1 từng batch; `f1` trong history là binary Toxic, không phải macro.',
        '- `.5` của history khác ngưỡng model active `.78` và checkpoint chọn `.80`; chỉ số hybrid gồm bộ luật khác model-only.',
        '- Chưa có bằng chứng cache đóng băng gây plateau. Kiểm cache hiện tại không chứng minh mọi lần chạy cũ đều không lỗi.',
        '', '## R0–R7 — kế hoạch, chưa tự triển khai các thay đổi model','',
        '| Mã | Bằng chứng/giả thuyết | Thí nghiệm tối thiểu | Cách đánh giá / điều kiện |',
        '|---|---|---|---|']
    plan=[
        ('R0','History thiếu logits mỗi epoch','Audit/tái chạy với checkpoint + logits/ID đầy đủ; đã thực hiện','Tái hiện metrics, drift và checkpoint; không thay production'),
        ('R1','Source-disjoint nhưng speaker/channel proxy giao nhau','Xác minh speaker; tạo manifest chẩn đoán phụ từ TRAIN theo speaker/nguồn, giữ nguyên manifest gốc','Group-CV chỉ TRAIN; báo số nhóm/nhãn, không dùng test để chia lại hoặc chọn'),
        ('R2','Bỏ sót có ASR thiếu/sai; chưa có transcript chuẩn','Đối chiếu một nhóm TRAIN bằng transcript người xác minh; so ASR hiện hành với một thay đổi cố định','WER/CER chỉ khi có reference; coverage, FP/FN; không thêm prompt ép từ Toxic'),
        ('R3','Supervision là cấp clip, không phải từng đoạn','Xác minh đoạn evidence trên TRAIN rồi thử loss phụ cho các đoạn đã xác minh','Không gán mọi đoạn của clip Toxic thành 1; giữ cùng split, so với R0'),
        ('R4','Max pooling chỉ truyền gradient qua một đoạn mỗi clip','Thử mean BCE trên đoạn của clip âm đã kiểm ASR, giữ max-MIL cho clip dương','Một thay đổi loss; cùng seed/LR/epoch/replay; báo FP, FN, AP và recall ViCTSD'),
        ('R5','Ngữ cảnh thử 15/11 giảm FP nhưng tăng FN so checkpoint 13/11','Nếu R2/R3 đủ dữ liệu, train và infer cùng cửa sổ ngữ cảnh cố định','Không tự phát hành từ inference-only; giữ Recall, báo truncation'),
        ('R6','Loss/AP đổi nhưng quyết định tại 0,5 có thể chưa qua ngưỡng','Sau khi khóa trọng số, báo điểm vận hành và calibration trên dev; giữ .5 làm mốc','Chọn ngưỡng chỉ train/validation theo FP/Recall đã định; không tuning test'),
        ('R7','Validation dùng nhiều lần, thiếu test speaker độc lập mới','Khóa toàn bộ pipeline rồi đánh giá một lần trên tập mới có nguồn/quyền dùng và nhãn xác minh','Báo support/coverage/CI, FP/FN, CPU latency; không chọn lại model bằng test')]
    for r in plan:lines.append('| '+' | '.join(r)+' |')
    lines+=['','## Artifact','',
        '- Public: `counts.csv`, `overlaps.csv`, `validation-metrics.csv`, `probability-drift.csv`, `history-comparison.csv`, JSON và đồ thị.',
        '- Private: `validation/<snapshot>/segments.csv`, `clips.csv`, `validation.npz`, `metadata.json`; checkpoint từng epoch, transcript/trùng/tiền xử lý ở cùng thư mục run.',
        '- Không có logits test hoặc model mới được cài vào app.']
    (public/'AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_csv(public/'experiment-plan.csv',[dict(id=r[0],evidence=r[1],experiment=r[2],evaluation=r[3]) for r in plan])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['data','checkpoints','replay','report','all'])
    parser.add_argument('--run-id',default=DEFAULT_RUN)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    if args.command=='all':
        if args.device!='cuda':raise ValueError('Full replay requires --device cuda; data/checkpoints/report support CPU')
        data(args.run_id);checkpoints(args.run_id,args.device);replay(args.run_id);report(args.run_id)
    elif args.command=='data':data(args.run_id)
    elif args.command=='checkpoints':checkpoints(args.run_id,args.device)
    elif args.command=='replay':replay(args.run_id)
    else:report(args.run_id)


if __name__=='__main__':main()
