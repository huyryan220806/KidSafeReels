"""Precision-focused Toxic MIL training; CPU ASR, frozen dev split, no test access."""
import argparse
import csv
from datetime import datetime, timezone
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml.deep_toxic import atomic_save, plot_history
from ml.experiment_metrics import binary_metrics, clean_svg, write_csv, write_json
from ml.hybrid_evaluation import decision_metrics
from ml.speech_mil import encode_clips, forward_clip, labels, predict
from ml.toxic_baseline import sha256
from ml.toxic_precision import BASE, RUN, development_rows, load, read
from ml.toxic_rules import ToxicRules, lexical_decision


def weighted_clip_loss(logit, target, negative_weight):
    weight = torch.where(target == 0, negative_weight, 1.)
    return torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), target) * weight


def hybrid_metrics(y, scores, rule_flags, threshold, missing_labels=()):
    # Missing ASR is a review decision, never a synthetic probability of zero.
    decisions = list((np.asarray(scores) >= threshold) | np.asarray(rule_flags, dtype=bool))
    return decision_metrics(list(y) + list(missing_labels), decisions + [False]*len(missing_labels))


def point_rank(point):
    return (point['eligible'], -point['hybrid_fp'], point['hybrid_recall'],
            -point['log_loss'], -point.get('epoch', 0), -abs(point['threshold']-.5))


def operating_point(y, scores, rule_flags, missing_labels, baseline_model, baseline_hybrid):
    points = []
    for threshold in np.linspace(.05, .95, 91):
        model = binary_metrics(y, scores, float(threshold))
        hybrid = hybrid_metrics(y, scores, rule_flags, threshold, missing_labels)
        eligible = (hybrid['fp'] <= baseline_hybrid['fp'] and hybrid['recall'] >= baseline_hybrid['recall']
                    and model['fp'] <= baseline_model['fp'])
        points.append(dict(**model, **{f'hybrid_{k}':v for k,v in hybrid.items()}, eligible=eligible))
    return max(points, key=point_rank), points


def components(clips, manifest):
    rules = ToxicRules(); flags, missing = {}, {}
    for split, items in clips.items():
        flags[split] = np.array([any(lexical_decision(rules.match(s['text']), s)['flagged']
                               for s in item['segments']) for item in items], dtype=bool)
        included = {x['row']['video_id'] for x in items}
        missing[split] = [r['training_label'] for r in development_rows(manifest)
                          if r['split'] == split and r['video_id'] not in included]
    return flags, missing


def train():
    private, public, protocol, manifest = load()
    target_dir = private/'toxic'; target_dir.mkdir(exist_ok=True)
    if (public/'selection.json').exists():
        raise ValueError('Selection already frozen')
    for row in development_rows(manifest):
        cache = read(private/'asr'/f"{row['video_id']}.json")
        if cache['sha256'] != row['sha256'] or cache['configuration'] != protocol['asr']:
            raise ValueError('ASR cache mismatch')
    hashes = {r['video_id']:sha256(private/'asr'/f"{r['video_id']}.json") for r in development_rows(manifest)}
    if (target_dir/'asr-hashes.json').exists() and read(target_dir/'asr-hashes.json') != hashes:
        raise ValueError('ASR changed after training began')
    write_json(target_dir/'asr-hashes.json', hashes)
    if not torch.cuda.is_available():
        raise RuntimeError('Use the isolated CUDA training environment')
    torch.set_num_threads(4); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    clips, coverage = encode_clips(private, manifest, tokenizer, ('train', 'valid'))
    write_csv(public/'coverage.csv', coverage)
    flags, missing = components(clips, manifest)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, local_files_only=True).to('cuda')
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith(('roberta.encoder.layer.10.', 'roberta.encoder.layer.11.', 'classifier.')))
    config = protocol['training']
    optimizer = torch.optim.AdamW([
        dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('classifier.')], lr=config['encoder_lr']),
        dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('classifier.')], lr=config['classifier_lr'])],
        weight_decay=config['weight_decay'])
    baseline_path = target_dir/'baseline-train-valid.npz'
    if not baseline_path.exists():
        np.savez(baseline_path, **{s:predict(model, clips[s], 'cuda') for s in clips})
    with np.load(baseline_path, allow_pickle=False) as cache:
        baseline = {s:cache[s] for s in clips}
    threshold = read(BASE/'kidsafe-config.json')['threshold']
    base_model = binary_metrics(labels(clips['valid']), baseline['valid'], threshold)
    base_hybrid = hybrid_metrics(labels(clips['valid']), baseline['valid'], flags['valid'], threshold, missing['valid'])
    write_json(public/'baseline-validation.json', dict(model_only=base_model, hybrid=base_hybrid))
    scaler = torch.amp.GradScaler('cuda')
    start, best, history, search = 1, None, [], []
    resume = target_dir/'resume.pt'
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=True)
        if state['protocol_sha256'] != sha256(public/'protocol.json'):
            raise ValueError('Resume protocol mismatch')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        start = state['epoch']+1; best = state['best']; history = state['history']; search = state['search']
    for epoch in range(start, config['epochs']+1):
        started = time.perf_counter(); model.train(); optimizer.zero_grad(set_to_none=True)
        order = np.random.default_rng(42+epoch).permutation(len(clips['train']))
        total_loss = 0.; accumulation = config['accumulate_clips']
        for step, index in enumerate(order):
            item = clips['train'][index]; target = torch.tensor(float(item['row']['training_label']), device='cuda')
            with torch.autocast('cuda', dtype=torch.float16):
                loss = weighted_clip_loss(forward_clip(model, item['tokens'], 'cuda'), target, config['negative_weight'])
            group_size = min(accumulation, len(order)-(step//accumulation)*accumulation)
            scaler.scale(loss/group_size).backward(); total_loss += float(loss.detach())
            if (step+1)%accumulation == 0 or step+1 == len(order):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            if (step+1)%40 == 0:
                print(f'Precision epoch{epoch}: {step+1}/{len(order)} loss={total_loss/(step+1):.4f}', flush=True)
        scores = {s:predict(model, clips[s], 'cuda') for s in clips}
        point, grid = operating_point(labels(clips['valid']), scores['valid'], flags['valid'], missing['valid'], base_model, base_hybrid)
        candidate = dict(epoch=epoch, **point)
        search.extend(dict(epoch=epoch, **p) for p in grid)
        if best is None or point_rank(candidate) > point_rank(best):
            best = candidate; atomic_save(model.state_dict(), target_dir/'best.pt')
        for split in clips:
            history.append(dict(epoch=epoch, split=split, optimizer_weighted_clip_loss=total_loss/len(order),
                elapsed_seconds=time.perf_counter()-started, **binary_metrics(labels(clips[split]), scores[split], .5)))
        write_csv(public/'training_history.csv', history); write_csv(public/'validation_thresholds.csv', search)
        write_json(public/'progress.json', dict(completed_epoch=epoch, total_epochs=config['epochs'], best=best))
        atomic_save(dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), best=best, history=history, search=search,
            protocol_sha256=sha256(public/'protocol.json')), resume)
        print(f'Epoch{epoch}: eligible={point["eligible"]}, hybrid FP/FN={point["hybrid_fp"]}/{point["hybrid_fn"]}, threshold={point["threshold"]:.2f}', flush=True)
    model.load_state_dict(torch.load(target_dir/'best.pt', map_location='cpu', weights_only=True))
    selected = target_dir/'selected'; model.save_pretrained(selected, safe_serialization=True); tokenizer.save_pretrained(selected)
    write_json(selected/'kidsafe-config.json', dict(version=f'toxic-phobert-precision-{RUN}', threshold=best['threshold'], max_length=192,
        segmentation='pyvi-0.1.1', normalization='NFC-casefold-whitespace-v1'))
    write_json(public/'selection.json', dict(selected=best, baseline_validation=dict(model_only=base_model, hybrid=base_hybrid),
        files={p.name:sha256(p) for p in selected.iterdir() if p.is_file()}, protocol_sha256=sha256(public/'protocol.json'),
        asr_hashes_sha256=sha256(target_dir/'asr-hashes.json'), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        selected_utc=datetime.now(timezone.utc).isoformat(), deployment='candidate_only', test_evaluated=False,
        label_status='user_reported_weak_clip_labels'))
    plot_history(public, history)
    report(clips, flags, missing, model, baseline, best['threshold'], private, public)


def report(clips, flags, missing, model, baseline, selected_threshold, private, public):
    model.eval(); rows, decisions, records = [], [], []
    for variant in ('baseline', 'selected'):
        threshold = read(BASE/'kidsafe-config.json')['threshold'] if variant == 'baseline' else selected_threshold
        for split, items in clips.items():
            scores = baseline[split] if variant == 'baseline' else predict(model, items, 'cuda')
            rows.append(dict(variant=variant, split=split, **binary_metrics(labels(items), scores, threshold)))
            decisions.append(dict(variant=variant, split=split,
                **hybrid_metrics(labels(items), scores, flags[split], threshold, missing[split])))
            for item, score, rule in zip(items, scores, flags[split]):
                records.append(dict(video_id=item['row']['video_id'], split=split, variant=variant, label=item['row']['training_label'],
                    score=float(score), rule_flag=bool(rule), model_flag=bool(score>=threshold), hybrid_flag=bool(rule or score>=threshold)))
    write_csv(public/'model_metrics.csv', rows); write_csv(public/'hybrid_metrics.csv', decisions)
    write_csv(private/'toxic/predictions.csv', records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    for ax, metric in zip(axes, ('fp', 'fn', 'f1')):
        for i, variant in enumerate(('baseline', 'selected')):
            values = [next(r for r in decisions if r['variant']==variant and r['split']==s)[metric] for s in ('train', 'valid')]
            ax.bar(np.arange(2)+i*.35, values, .35, label=variant)
        ax.set_xticks(np.arange(2)+.175, ('train', 'validation')); ax.set_title(f'Hybrid {metric}'); ax.legend()
    fig.suptitle('Toxic CPU-ASR · nhãn yếu · số đo phát triển, không phải test độc lập')
    for suffix in ('png','svg'):
        fig.savefig(public/f'development_comparison.{suffix}', dpi=160)
    clean_svg(public/'development_comparison.svg'); plt.close(fig)


def cpu_check():
    from ml.deep_inference import PhoBertClassifier
    private, public, _, manifest = load()
    if (public/'cpu-validation.json').exists():
        raise ValueError('CPU validation check already recorded')
    selected = private/'toxic/selected'; selection = read(public/'selection.json')
    for name, digest in selection['files'].items():
        if sha256(selected/name) != digest:
            raise ValueError('Selected checkpoint changed')
    hashes_path = private/'toxic/asr-hashes.json'
    if sha256(hashes_path) != selection['asr_hashes_sha256']:
        raise ValueError('Frozen ASR hash inventory changed')
    hashes = read(hashes_path)
    with (private/'toxic/predictions.csv').open(encoding='utf-8-sig') as stream:
        reference = {(r['variant'], r['video_id']):r for r in csv.DictReader(stream) if r['split']=='valid'}
    models = {'baseline':PhoBertClassifier(BASE/'kidsafe-config.json'), 'selected':PhoBertClassifier(selected/'kidsafe-config.json')}
    rules = ToxicRules(); rows = []
    for row in development_rows(manifest):
        if row['split'] != 'valid':
            continue
        asr_path = private/'asr'/f"{row['video_id']}.json"
        if sha256(asr_path) != hashes[row['video_id']]:
            raise ValueError('Validation ASR changed after selection')
        segments = read(asr_path)['segments']
        rule = any(lexical_decision(rules.match(s['text']), s)['flagged'] for s in segments)
        for variant, model in models.items():
            values = [model.predict(s['text'])['score'] for s in segments]
            score = max(values) if values else None
            rows.append(dict(video_id=row['video_id'], label=row['training_label'], variant=variant, score=score,
                model_flag=score is not None and score>=model.threshold,
                hybrid_flag=rule or (score is not None and score>=model.threshold)))
    write_csv(private/'toxic/cpu-validation-predictions.csv', rows)
    metrics, deltas, changed = [], [], []
    for row in rows:
        if row['score'] is not None:
            old = reference[row['variant'], row['video_id']]
            deltas.append(abs(row['score']-float(old['score'])))
            if row['hybrid_flag'] != (old['hybrid_flag']=='True') or row['model_flag'] != (old['model_flag']=='True'):
                changed.append(dict(video_id=row['video_id'], variant=row['variant']))
    for variant, model in models.items():
        part = [r for r in rows if r['variant']==variant]; eligible = [r for r in part if r['score'] is not None]
        metrics.append(dict(variant=variant, model_only=binary_metrics([r['label'] for r in eligible], [r['score'] for r in eligible], model.threshold),
            hybrid=decision_metrics([r['label'] for r in part], [r['hybrid_flag'] for r in part])))
    write_json(public/'cpu-validation.json', dict(metrics=metrics, scope='same development validation, not independent test',
        max_score_delta=max(deltas, default=None), changed_decisions=len(changed),
        test_evaluated=False, deployment='candidate_only', selection_sha256=sha256(public/'selection.json')))
    write_json(private/'toxic/cpu-parity.json', dict(changed_decisions=changed, max_score_delta=max(deltas, default=None)))
    print(metrics, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('command', choices=['train','cpu-check'])
    args = parser.parse_args()
    train() if args.command == 'train' else cpu_check()
