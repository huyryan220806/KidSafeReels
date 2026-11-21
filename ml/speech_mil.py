"""Weak clip-label adaptation: max pooling over ASR segments, never label every segment positive."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml.deep_toxic import atomic_save, plot_history
from ml.experiment_metrics import binary_metrics, threshold_rows, write_csv, write_json, plot_results
from ml.hybrid_evaluation import decision_metrics
from ml.local_adaptation import load_frozen, read
from ml.toxic_baseline import ROOT, normalize, sha256
from ml.toxic_rules import ToxicRules, lexical_decision

BASE = ROOT/'models/checkpoints/toxic-phobert-v3'


def clip_logit(logits):
    if logits.ndim != 2 or logits.shape[1] != 2 or not len(logits):
        raise ValueError('Expected nonempty two-class segment logits')
    return (logits[:,1]-logits[:,0]).max()


def encode_clips(private, manifest, tokenizer, splits):
    from pyvi import ViTokenizer
    rows, coverage = {}, []
    for split in splits:
        rows[split] = []
        selected = [r for r in manifest['speech'] if r['split']==split]
        for row in selected:
            trace = read(private/'asr'/f"{row['video_id']}.json")
            if trace['sha256'] != row['sha256']:
                raise ValueError('ASR/source mismatch')
            segments = [s for s in trace['segments'] if normalize(s['text'])]
            if not segments:
                continue
            text = [ViTokenizer.tokenize(normalize(s['text'])) for s in segments]
            tokens = tokenizer(text,truncation=True,max_length=192,padding=True,return_tensors='pt')
            rows[split].append(dict(row=row,tokens=tokens,segments=segments))
        coverage.append(dict(split=split,total_clips=len(selected),asr_eligible=len(rows[split]),
                             missing_asr=len(selected)-len(rows[split]),
                             negative=sum(x['row']['training_label']==0 for x in rows[split]),
                             positive=sum(x['row']['training_label']==1 for x in rows[split]),
                             segments=sum(len(x['segments']) for x in rows[split])))
        if {x['row']['training_label'] for x in rows[split]} != {0,1}:
            raise ValueError('ASR eligible subset missing a class')
    return rows, coverage


def forward_clip(model, tokens, device):
    return clip_logit(model(**{k:v.to(device) for k,v in tokens.items()}).logits)


def predict(model, clips, device):
    model.eval()
    with torch.inference_mode():
        return np.array([float(torch.sigmoid(forward_clip(model,x['tokens'],device).float())) for x in clips])


def labels(clips):
    return np.array([x['row']['training_label'] for x in clips])


def operating_point(y, scores, baseline):
    grid = threshold_rows(y,scores)
    eligible = [r for r in grid if r['precision'] is not None and
                r['precision'] >= (baseline['precision'] or 0) and r['recall'] >= baseline['recall']]
    best = max(eligible or grid,key=lambda r:(r['macro_f1'] or 0,r['f1'] or 0,-r['log_loss'],-abs(r['threshold']-.5)))
    return dict(best,eligible=bool(eligible))


def train(run_id):
    private, public, protocol, manifest = load_frozen(run_id)
    private, public = private/'toxic', public/'toxic'
    if (public/'selection.json').exists():
        raise ValueError('Model selection already frozen')
    for name,digest in protocol['model_files'].items():
        if sha256(BASE/name) != digest:
            raise ValueError('Initial model changed')
    if not torch.cuda.is_available():
        raise RuntimeError('Use isolated CUDA environment for adaptation')
    torch.set_num_threads(4); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    device='cuda'
    tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    clips,coverage=encode_clips(private.parent,manifest,tokenizer,('train','valid'))
    write_csv(public/'coverage.csv',coverage)
    model=AutoModelForSequenceClassification.from_pretrained(BASE,local_files_only=True).to(device)
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith(('roberta.encoder.layer.10.','roberta.encoder.layer.11.','classifier.')))
    optimizer=torch.optim.AdamW([
        dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('classifier.')],lr=5e-6),
        dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('classifier.')],lr=2e-5)],weight_decay=.01)
    baseline_threshold=read(BASE/'kidsafe-config.json')['threshold']
    baseline_path=private/'baseline-train-valid.npz'
    if not baseline_path.exists():
        private.mkdir(parents=True,exist_ok=True)
        np.savez(baseline_path,**{s:predict(model,clips[s],device) for s in clips})
    with np.load(baseline_path,allow_pickle=False) as cache:
        baseline={s:cache[s] for s in clips}
    base_valid=binary_metrics(labels(clips['valid']),baseline['valid'],baseline_threshold)
    scaler=torch.amp.GradScaler('cuda')
    start,best,history=1,None,[]
    resume=private/'resume.pt'
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=True)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        start=state['epoch']+1;best=state['best'];history=state['history']
    for epoch in range(start,protocol['toxic_training']['epochs']+1):
        started=time.perf_counter(); model.train();optimizer.zero_grad(set_to_none=True)
        order=np.random.default_rng(42+epoch).permutation(len(clips['train']))
        total_loss=0.
        for step,index in enumerate(order):
            item=clips['train'][index]
            target=torch.tensor(float(item['row']['training_label']),device=device)
            with torch.autocast('cuda',dtype=torch.float16):
                logit=forward_clip(model,item['tokens'],device)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logit.float(),target)
            group_size=min(8,len(order)-(step//8)*8)
            scaler.scale(loss/group_size).backward();total_loss+=float(loss.detach())
            if (step+1)%8==0 or step+1==len(order):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            if (step+1)%40==0:
                print(f'MIL epoch {epoch}/5 clip {step+1}/{len(order)} loss {total_loss/(step+1):.4f}',flush=True)
        scores={s:predict(model,clips[s],device) for s in clips}
        point=operating_point(labels(clips['valid']),scores['valid'],base_valid)
        candidate=dict(epoch=epoch,**point)
        rank=lambda x:(x['eligible'],x['macro_f1'] or 0,x['f1'] or 0,-x['log_loss'],-x['epoch'])
        if best is None or rank(candidate)>rank(best):
            best=candidate;atomic_save(model.state_dict(),private/'best.pt')
        for split in clips:
            history.append(dict(epoch=epoch,split=split,optimizer_clip_loss=total_loss/len(order),
                                elapsed_seconds=time.perf_counter()-started,
                                **binary_metrics(labels(clips[split]),scores[split],.5)))
        write_csv(public/'training_history.csv',history)
        write_json(public/'progress.json',dict(completed_epoch=epoch,total_epochs=5,best=best))
        atomic_save(dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                         torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),best=best,history=history),resume)
        print(f'MIL epoch {epoch}: validation F1={point["f1"]:.4f}, threshold={point["threshold"]:.2f}, best={best["epoch"]}',flush=True)
    model.load_state_dict(torch.load(private/'best.pt',map_location='cpu',weights_only=True))
    target=private/'selected';model.save_pretrained(target,safe_serialization=True);tokenizer.save_pretrained(target)
    write_json(target/'kidsafe-config.json',dict(version=f'toxic-phobert-mil-{run_id}',threshold=best['threshold'],max_length=192,
                                                segmentation='pyvi-0.1.1',normalization='NFC-casefold-whitespace-v1'))
    write_json(public/'selection.json',dict(selected=best,baseline_validation=base_valid,
        files={p.name:sha256(p) for p in target.iterdir() if p.is_file()},
        protocol_sha256=sha256(public.parent/'protocol.json'),manifest_sha256=protocol['manifest_sha256'],
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        selected_utc=datetime.now(timezone.utc).isoformat(),test_evaluated=False,
        label_status='user_reported_weak_clip_labels',deployment='candidate_only'))
    plot_history(public,history)


def evaluate(run_id):
    private,public,protocol,manifest=load_frozen(run_id)
    private,public=private/'toxic',public/'toxic'
    if (public/'evaluation.json').exists(): raise ValueError('Test already evaluated')
    selection=read(public/'selection.json');target=private/'selected'
    for name,digest in selection['files'].items():
        if sha256(target/name)!=digest: raise ValueError('Selected artifact changed')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    clips,coverage=encode_clips(private.parent,manifest,tokenizer,('train','valid','test'))
    write_csv(public/'coverage.csv',coverage)
    metrics,hybrid_metrics,predictions,individual=[],[],{},[]
    rules=ToxicRules()
    for variant,directory in (('baseline',BASE),('selected',target)):
        model=AutoModelForSequenceClassification.from_pretrained(directory,local_files_only=True).to(device)
        threshold=read(directory/'kidsafe-config.json')['threshold'];predictions[variant]={}
        for split,items in clips.items():
            scores=predict(model,items,device);y=labels(items)
            predictions[variant][split]=(y,scores)
            metrics.append(dict(task='toxic',dataset='speech200-user-reported-labels-ASR-GPU',variant=variant,split=split,
                                **binary_metrics(y,scores,threshold)))
            lexical=np.array([any(lexical_decision(rules.match(s['text']),s)['flagged'] for s in x['segments']) for x in items])
            for method,pred in (('model_only',scores>=threshold),('hybrid',(scores>=threshold)|lexical)):
                hybrid_metrics.append(dict(variant=variant,split=split,method=method,**decision_metrics(y,pred)))
            for item,score,rule in zip(items,scores,lexical):
                individual.append(dict(video_id=item['row']['video_id'],split=split,variant=variant,
                                       label=item['row']['training_label'],score=float(score),rule_flag=bool(rule),
                                       model_flag=bool(score>=threshold),hybrid_flag=bool(score>=threshold or rule)))
        del model
        if device=='cuda':torch.cuda.empty_cache()
    write_csv(private/'predictions.csv',individual)
    write_csv(public/'metrics_by_split.csv',metrics)
    write_csv(public/'hybrid_decisions.csv',hybrid_metrics)
    plot_results(public,metrics,predictions,[],label_description='Nhãn yếu người cung cấp',
                 scope_note='Toxic — nhãn yếu cấp clip, ASR GPU; chưa phải chất lượng app CPU')
    write_json(public/'evaluation.json',dict(metrics=metrics,hybrid_decisions=hybrid_metrics,coverage=coverage,
        selected_artifact_sha256=selection['files']['model.safetensors'],
        selection_sha256=sha256(public/'selection.json'),label_status='weak clip labels, not adjudicated gold',
        asr='CUDA FP16; CPU production pipeline not evaluated here',
        missing_asr='Excluded from probability metrics; review/null at inference, see coverage',
        deployment='candidate_only; active app unchanged'))
    print([r for r in metrics if r['split']=='test'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['train','evaluate']);parser.add_argument('--run-id',required=True)
    args=parser.parse_args();{'train':train,'evaluate':evaluate}[args.command](args.run_id)
