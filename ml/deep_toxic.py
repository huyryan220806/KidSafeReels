"""Fine-tune Vietnamese PhoBERT with epoch checkpoints and validation-only selection."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import time

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from ml.experiment_metrics import binary_metrics, threshold_rows, write_csv, write_json, plot_results, clean_svg
from ml.model_experiments import load_toxic, locations
from ml.toxic_baseline import ROOT, sha256

PRETRAINED = ROOT / 'models/checkpoints/pretrained/phobert-base'
EPOCHS = 6
MAX_LENGTH = 192
SEED = 42


def segment_text(text):
    from pyvi import ViTokenizer
    return ViTokenizer.tokenize(text)


def atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + '.partial')
    torch.save(payload, temporary)
    temporary.replace(path)


def encoded_data(data, tokenizer, private, needed):
    result, audit = {}, {}
    for split in needed:
        part = data[split]
        signature = hashlib.sha256(json.dumps(part['ids']).encode()).hexdigest()
        path = private / f'tokens-{split}.json'
        if path.exists():
            cached = json.loads(path.read_text(encoding='utf-8'))
            if cached['signature'] != signature or cached['max_length'] != MAX_LENGTH:
                raise ValueError('Token cache differs from frozen data/configuration')
        else:
            segmented = [segment_text(text) for text in part['x']]
            tokens = tokenizer(segmented, truncation=False, padding=False)
            rows, truncated = [], 0
            for i, ids in enumerate(tokens['input_ids']):
                truncated += len(ids) > MAX_LENGTH
                clipped = ids[:MAX_LENGTH]
                if len(ids) > MAX_LENGTH:
                    clipped[-1] = tokenizer.eos_token_id
                rows.append(dict(input_ids=clipped, attention_mask=[1]*len(clipped), labels=int(part['y'][i])))
            cached = dict(signature=signature, max_length=MAX_LENGTH, truncated=truncated, rows=rows)
            write_json(path, cached)
        result[split] = cached['rows']
        audit[split] = dict(n=len(cached['rows']), truncated=cached['truncated'], text_ids_sha256=signature)
    return result, audit


def loader(rows, tokenizer, batch_size, shuffle=False, epoch=0):
    def collate(items):
        labels = torch.tensor([item['labels'] for item in items])
        inputs = tokenizer.pad([{k:v for k,v in item.items() if k!='labels'} for item in items],
                               padding=True, pad_to_multiple_of=8, return_tensors='pt')
        return dict(inputs, labels=labels)
    generator = torch.Generator().manual_seed(SEED + epoch)
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      num_workers=0, generator=generator)


def predict(model, rows, tokenizer, device, batch_size=16):
    model.eval()
    values = []
    with torch.inference_mode():
        for batch in loader(rows, tokenizer, batch_size):
            batch.pop('labels')
            logits = model(**{k:v.to(device) for k,v in batch.items()}).logits
            values.extend(torch.softmax(logits.float(), dim=-1)[:,1].cpu().tolist())
    return np.asarray(values)


def operating_point(y, scores, baseline):
    rows = threshold_rows(y, scores)
    eligible = [row for row in rows if row['precision'] is not None and row['precision']>=baseline['precision']
                and row['recall']>=baseline['recall']]
    best = max(eligible or rows, key=lambda r:(r['macro_f1'] or 0, r['f1'] or 0, -abs(r['threshold']-.5)))
    return best, bool(eligible), rows


def plot_history(public, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for split in ('train','valid'):
        part=[r for r in history if r['split']==split]
        for ax, key in zip(axes,('log_loss','accuracy','f1')):
            ax.plot([r['epoch'] for r in part],[r[key] for r in part],'o-',label=split)
            ax.set(xlabel='Epoch thực tế',ylabel=key); ax.grid(alpha=.2); ax.legend()
    fig.suptitle('PhoBERT — số đo cuối mỗi epoch; ngưỡng cố định 0,5 trong đồ thị')
    for suffix in ('png','svg'):
        fig.savefig(public/f'training_history.{suffix}',dpi=160)
    clean_svg(public/'training_history.svg'); plt.close(fig)


def train(run_id):
    private,public=locations(run_id,'toxic')
    private.mkdir(parents=True,exist_ok=True)
    if (public/'selection.json').exists():
        raise ValueError('Selection frozen; do not repeat training for this run')
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('This training command requires the isolated CUDA environment')
    device='cuda'
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    provenance=json.loads((PRETRAINED/'provenance.json').read_text(encoding='utf-8'))
    for entry in provenance['files']:
        if sha256(PRETRAINED/entry['path'])!=entry['sha256']:
            raise ValueError('Pretrained checkpoint checksum mismatch')
    data,audit=load_toxic()
    baseline=joblib.load(ROOT/'models/checkpoints/toxic-tfidf-v1.joblib')
    base_valid=binary_metrics(data['valid']['y'],baseline['model'].predict_proba(data['valid']['x'])[:,1],baseline['threshold'])
    protocol=dict(run_id=run_id,task='toxic',seed=SEED,epochs=EPOCHS,batch_size=8,gradient_accumulation=2,
        learning_rate=2e-5,weight_decay=.01,warmup_fraction=.1,max_length=MAX_LENGTH,
        loss='CrossEntropyLoss weights [1,sqrt(train_negative/train_positive)]',
        selection='max validation macro-F1 subject to precision and recall >= v1; epochs and thresholds on validation only',
        test_policy='Previously reported benchmark, not a new independent target test; no test inference before selection',
        training='all PhoBERT layers, fp16 AMP, gradient clipping 1.0',segmentation='PyVi 0.1.1 on existing normalized text',
        pretrained=provenance,baseline_validation=base_valid,
        data_audit=audit,environment=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0)))
    if not (public/'protocol.json').exists():
        write_json(public/'protocol.json',dict(protocol,started_at=datetime.now().astimezone().isoformat()))
    else:
        existing=json.loads((public/'protocol.json').read_text(encoding='utf-8'))
        if any(existing.get(k)!=v for k,v in protocol.items()):
            raise ValueError('Resume protocol mismatch')
    tokenizer=AutoTokenizer.from_pretrained(PRETRAINED,local_files_only=True)
    encoded,token_audit=encoded_data(data,tokenizer,private,('train','valid'))
    write_json(public/'tokenization_audit.json',token_audit)
    model=AutoModelForSequenceClassification.from_pretrained(PRETRAINED,num_labels=2,local_files_only=True)
    model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    steps_per_epoch=(len(encoded['train'])+15)//16
    scheduler=get_linear_schedule_with_warmup(optimizer,int(steps_per_epoch*EPOCHS*.1),steps_per_epoch*EPOCHS)
    scaler=torch.amp.GradScaler('cuda')
    positive=float(np.sum(data['train']['y'])); negative=len(data['train']['y'])-positive
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor([1.,(negative/positive)**.5],device=device))
    start_epoch,best,history=1,None,[]
    resume=private/'resume.pt'
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=True)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        start_epoch=state['epoch']+1; best=state['best']; history=state['history']
        print(f'Resume after completed epoch {state["epoch"]}',flush=True)
    for epoch in range(start_epoch,EPOCHS+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        batches=loader(encoded['train'],tokenizer,8,shuffle=True,epoch=epoch)
        started=time.perf_counter(); total_loss,total_count=0.,0
        for step,batch in enumerate(batches,1):
            labels=batch.pop('labels').to(device)
            batch={k:v.to(device) for k,v in batch.items()}
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(**batch).logits
                loss=criterion(logits,labels)
            total_loss+=float(loss.detach())*len(labels); total_count+=len(labels)
            # The last accumulation group may contain only one minibatch.
            group_size=2 if step<=len(batches)-(len(batches)%2) else 1
            scaler.scale(loss/group_size).backward()
            if step%2==0 or step==len(batches):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            if step%100==0:
                print(f'epoch {epoch}/{EPOCHS} step {step}/{len(batches)} loss={total_loss/total_count:.4f} elapsed={time.perf_counter()-started:.1f}s',flush=True)
        fit_seconds=time.perf_counter()-started
        scores={s:predict(model,encoded[s],tokenizer,device) for s in ('train','valid')}
        point,eligible,_=operating_point(data['valid']['y'],scores['valid'],base_valid)
        candidate=dict(epoch=epoch,eligible=eligible,**point)
        rank=lambda row:(row['eligible'],row['macro_f1'],row['f1'],-row['epoch'])
        if best is None or rank(candidate)>rank(best):
            best=candidate
            atomic_save(model.state_dict(),private/'best.pt')
            np.savez(private/'best-validation.npz',labels=data['valid']['y'],scores=scores['valid'])
        for split in ('train','valid'):
            history.append(dict(epoch=epoch,split=split,fit_seconds=fit_seconds,
                optimizer_weighted_loss=total_loss/total_count,**binary_metrics(data[split]['y'],scores[split],.5)))
        write_csv(public/'training_history.csv',history)
        write_json(public/'training_progress.json',dict(completed_epoch=epoch,total_epochs=EPOCHS,best=best,
                   updated_at=datetime.now().astimezone().isoformat()))
        atomic_save(dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),best=best,history=history),resume)
        print(json.dumps(dict(epoch=epoch,fit_seconds=fit_seconds,validation=candidate,best_epoch=best['epoch'])),flush=True)
    model.load_state_dict(torch.load(private/'best.pt',map_location='cpu',weights_only=True))
    target=private/'selected'
    model.save_pretrained(target,safe_serialization=True); tokenizer.save_pretrained(target)
    write_json(target/'kidsafe-config.json',dict(version=f'toxic-phobert-{run_id}',threshold=best['threshold'],
               max_length=MAX_LENGTH,segmentation='pyvi-0.1.1',normalization='NFC-casefold-whitespace-v1'))
    files=[dict(path=p.name,sha256=sha256(p),bytes=p.stat().st_size) for p in sorted(target.iterdir()) if p.is_file()]
    write_json(public/'selection.json',dict(run_id=run_id,selected=best,files=files,
        protocol_sha256=sha256(public/'protocol.json'),selected_at=datetime.now().astimezone().isoformat(),
        artifact_directory=f'runs/{run_id}/toxic/selected',test_evaluated=False))
    plot_history(public,history)


def evaluate(run_id):
    private,public=locations(run_id,'toxic')
    if (public/'evaluation.json').exists():
        raise ValueError('Frozen test already evaluated')
    selection=json.loads((public/'selection.json').read_text(encoding='utf-8'))
    target=private/'selected'
    for entry in selection['files']:
        if sha256(target/entry['path'])!=entry['sha256']:
            raise ValueError('Frozen model changed')
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=AutoModelForSequenceClassification.from_pretrained(target,local_files_only=True).to(device)
    tokenizer=AutoTokenizer.from_pretrained(target,local_files_only=True)
    data,_=load_toxic(); encoded,audit=encoded_data(data,tokenizer,private,('train','valid','test'))
    write_json(public/'tokenization_audit.json',audit)
    baseline=joblib.load(ROOT/'models/checkpoints/toxic-tfidf-v1.joblib')
    config=json.loads((target/'kidsafe-config.json').read_text(encoding='utf-8'))
    rows,predictions,confusion=[],{},[]
    for variant in ('baseline','selected'):
        threshold=baseline['threshold'] if variant=='baseline' else config['threshold']
        predictions[variant]={}; individual=[]
        for split in ('train','valid','test'):
            d=data[split]; started=time.perf_counter()
            scores=(baseline['model'].predict_proba(d['x'])[:,1] if variant=='baseline' else predict(model,encoded[split],tokenizer,device))
            metric=binary_metrics(d['y'],scores,threshold)
            rows.append(dict(run_id=run_id,task='toxic',dataset='ViCTSD',variant=variant,
                model_version=baseline['version'] if variant=='baseline' else config['version'],split=split,
                predict_batch_seconds=time.perf_counter()-started,**metric))
            predictions[variant][split]=(d['y'],scores)
            confusion.extend(dict(variant=variant,split=split,cell=cell,count=metric[cell]) for cell in ('tn','fp','fn','tp'))
            for sid,group,label,score in zip(d['ids'],d['groups'],d['y'],scores):
                pred=int(score>=threshold)
                outcome=('TP' if pred else 'FN') if label else ('FP' if pred else 'TN')
                individual.append(dict(sample_id=sid,group_id=group,split=split,label=int(label),score=float(score),prediction=pred,outcome=outcome))
        write_csv(private/f'{variant}_predictions.csv',individual)
    write_csv(public/'metrics_by_split.csv',rows)
    write_csv(public/'comparison.csv',[r for r in rows if r['split']=='test'])
    write_csv(public/'confusion_matrix.csv',confusion)
    plot_results(public,rows,predictions,[])
    write_json(public/'evaluation.json',dict(run_id=run_id,task='toxic',metrics=rows,
        selection_sha256=sha256(public/'selection.json'),evaluated_at=datetime.now().astimezone().isoformat(),
        loss_definition='unweighted binary cross-entropy on predicted probabilities; epoch optimizer loss is separate',
        test_status='previously reported benchmark; not unseen target-video test'))
    print(json.dumps([r for r in rows if r['split']=='test'],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['train','evaluate'])
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    {'train':train,'evaluate':evaluate}[args.command](args.run_id)
