"""Read-only validation recording and independently checked binary metrics."""
import hashlib

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from ml.experiment_metrics import write_csv, write_json


def metrics(labels, probabilities, threshold=.5):
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or not len(y) or p.shape != y.shape or not set(y).issubset({0, 1}):
        raise ValueError('Expected nonempty aligned binary labels and probabilities')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not 0 <= threshold <= 1:
        raise ValueError('Invalid probabilities/threshold')
    pred = p >= threshold
    tn, fp = int(((y == 0) & ~pred).sum()), int(((y == 0) & pred).sum())
    fn, tp = int(((y == 1) & ~pred).sum()), int(((y == 1) & pred).sum())
    div = lambda a,b: a/b if b else 0.
    f1, f0 = div(2*tp,2*tp+fp+fn), div(2*tn,2*tn+fp+fn)
    return dict(n=len(y), negative=tn+fp, positive=tp+fn, threshold=threshold,
        tn=tn, fp=fp, fn=fn, tp=tp, accuracy=(tp+tn)/len(y),
        toxic_precision=div(tp,tp+fp), toxic_recall=div(tp,tp+fn), toxic_f1=f1,
        macro_f1=(f1+f0)/2, weighted_f1=((tn+fp)*f0+(tp+fn)*f1)/len(y),
        log_loss=float(log_loss(y,np.column_stack([1-p,p]),labels=[0,1])),
        average_precision=float(average_precision_score(y,p)) if (y==1).any() else None,
        roc_auc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,
        undefined_class_policy='zero division -> 0 for P/R/F1; undefined AP/AUC -> null')


def aligned(rows, ids):
    lookup = {r['video_id']:r for r in rows}
    if len(lookup)!=len(rows) or len(ids)!=len(set(ids)) or set(ids)!=set(lookup):
        raise ValueError('Missing, extra or duplicate sample IDs')
    return [lookup[sid] for sid in ids]


def drift(before, after, threshold=.5):
    ids=[r['video_id'] for r in before]
    right=aligned(after,ids); aligned(before,ids)
    y=np.array([r['label'] for r in before])
    if not np.array_equal(y,[r['label'] for r in right]):
        raise ValueError('Label mismatch between checkpoints')
    a=np.array([r['probability'] for r in before]); b=np.array([r['probability'] for r in right])
    metrics(y,a,threshold); metrics(y,b,threshold)
    changed=(a>=threshold)!=(b>=threshold)
    true_delta=np.where(y==1,b-a,a-b)
    return dict(n=len(y),threshold=threshold,mean_absolute_probability_drift=float(np.abs(b-a).mean()),
        max_absolute_probability_drift=float(np.abs(b-a).max()),
        mean_signed_probability_drift=float((b-a).mean()),
        prediction_flips=int(changed.sum()),flip_rate=float(changed.mean()),
        true_label_probability_increased=int((true_delta>1e-8).sum()),
        true_label_probability_decreased=int((true_delta < -1e-8).sum()),
        negative_mean_delta=float((b-a)[y==0].mean()) if (y==0).any() else None,
        positive_mean_delta=float((b-a)[y==1].mean()) if (y==1).any() else None)


def record(model, clips, directory, *, device, origin, checkpoint, epoch, expected=None):
    """Record raw two-class segment logits; only video labels are human truth."""
    import torch
    if directory.exists():
        raise ValueError(f'Record exists, preserve it: {directory}')
    if any(x['row']['split']!='valid' for x in clips):
        raise ValueError('This recorder accepts validation only')
    model.eval()
    segment_rows=[]; clip_rows=[]; all_logits=[]; shapes=[]; repeat_delta=0.
    max_softmax_sigmoid_delta=0.; disabled_grad=True
    with torch.inference_mode():
        for item in clips:
            output=model(**{k:v.to(device) for k,v in item['tokens'].items()}).logits
            if output.ndim!=2 or output.shape!=(len(item['segments']),2):
                raise ValueError('Expected [number of segments, 2] logits')
            disabled_grad &= not torch.is_grad_enabled() and not output.requires_grad
            logits=output.float()
            probability=torch.sigmoid(logits[:,1]-logits[:,0])
            softmax=torch.softmax(logits,dim=-1)[:,1]
            max_softmax_sigmoid_delta=max(max_softmax_sigmoid_delta,float((softmax-probability).abs().max()))
            if not clip_rows:
                repeated=model(**{k:v.to(device) for k,v in item['tokens'].items()}).logits.float()
                repeat_delta=float((repeated-logits).abs().max())
            raw=logits.cpu().numpy(); probs=probability.cpu().numpy(); winner=int(probs.argmax())
            row=item['row']; shapes.append(list(raw.shape))
            for index,(segment,vector,p) in enumerate(zip(item['segments'],raw,probs)):
                sid=f"{row['video_id']}:asr:{index:04d}"
                segment_rows.append(dict(segment_id=sid,video_id=row['video_id'],segment_index=index,
                    start_seconds=segment['start'],end_seconds=segment['end'],
                    text_sha256=hashlib.sha256(segment['text'].encode()).hexdigest(),
                    video_label=row['training_label'],segment_label=None,
                    logit_0=float(vector[0]),logit_1=float(vector[1]),probability=float(p),
                    prediction=int(p>=.5),threshold=.5,is_max_segment=index==winner))
            clip_rows.append(dict(video_id=row['video_id'],label=row['training_label'],
                winning_segment_id=f"{row['video_id']}:asr:{winner:04d}",
                logit_0=float(raw[winner,0]),logit_1=float(raw[winner,1]),
                log_odds=float(raw[winner,1]-raw[winner,0]),probability=float(probs[winner]),
                prediction=int(probs[winner]>=.5),threshold=.5,segments=len(raw)))
            all_logits.extend(raw)
    actual=np.array([r['probability'] for r in clip_rows]); labels=[r['label'] for r in clip_rows]
    directory.mkdir(parents=True)
    write_csv(directory/'segments.csv',segment_rows); write_csv(directory/'clips.csv',clip_rows)
    np.savez_compressed(directory/'validation.npz',
        segment_ids=np.array([r['segment_id'] for r in segment_rows]),
        segment_video_ids=np.array([r['video_id'] for r in segment_rows]),
        segment_logits=np.asarray(all_logits),
        segment_probabilities=np.array([[1-r['probability'],r['probability']] for r in segment_rows]),
        segment_predictions=np.array([r['prediction'] for r in segment_rows]),
        segment_labels=np.full(len(segment_rows),-1,dtype=int),
        inherited_video_labels=np.array([r['video_label'] for r in segment_rows]),
        video_ids=np.array([r['video_id'] for r in clip_rows]),labels=np.asarray(labels),
        clip_probabilities=actual,clip_predictions=(actual>=.5).astype(int),
        clip_winning_logits=np.array([[r['logit_0'],r['logit_1']] for r in clip_rows]))
    result=dict(origin=origin,checkpoint=checkpoint,epoch=epoch,device=device,
        num_labels=model.config.num_labels,id2label=model.config.id2label,
        label_semantics={'0':'non_toxic','1':'toxic'},label_source='Human video labels, not segment labels',
        segment_label_sentinel=-1,segment_count=len(segment_rows),clip_count=len(clip_rows),
        logits_shapes=shapes,model_training=model.training,gradient_disabled=disabled_grad,
        repeated_forward_max_logit_delta=repeat_delta,softmax_sigmoid_max_delta=max_softmax_sigmoid_delta,
        ordered_ids_sha256=hashlib.sha256('\n'.join(r['video_id'] for r in clip_rows).encode()).hexdigest(),
        expected_probability_max_delta=float(np.max(np.abs(actual-expected))) if expected is not None else None,
        expected_prediction_flips=int(((actual>=.5)!=(expected>=.5)).sum()) if expected is not None else None,
        metrics=metrics(labels,actual))
    write_json(directory/'metadata.json',result)
    return result
