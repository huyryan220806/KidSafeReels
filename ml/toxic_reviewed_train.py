"""Human clip supervision with written-text rehearsal and dual validation gates."""
import argparse
from datetime import datetime, timezone
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml.deep_toxic import atomic_save, loader, plot_history, predict as predict_written
from ml.experiment_metrics import binary_metrics, write_csv, write_json, plot_results, clean_svg
from ml.hybrid_evaluation import decision_metrics
from ml.speech_mil import encode_clips, forward_clip, labels, predict
from ml.toxic_precision_train import hybrid_metrics
from ml.toxic_reviewed import BASE, PRIVATE, PUBLIC, ROOT, RUN, WRITTEN, load, read, sha
from ml.toxic_rules import ToxicRules, lexical_decision


def rehearsal_indices(rows, size, seed):
    if size <= 0 or size % 2:
        raise ValueError('Rehearsal batch size must be a positive even integer')
    pools = [np.array([i for i, row in enumerate(rows) if row['labels'] == label]) for label in (0, 1)]
    if any(not len(pool) for pool in pools):
        raise ValueError('Rehearsal requires both classes in TRAIN')
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(pool, size//2, replace=True) for pool in pools])


def rank(point):
    return (point['eligible'], point['speech_hybrid_f1'] or 0, -point['speech_hybrid_fp'],
            point['written_f1'] or 0, -point['speech_log_loss'], -point['epoch'],
            -abs(point['threshold']-.78))


def eligible(speech_model, speech_hybrid, written, baseline):
    return (speech_hybrid['fp'] <= baseline['hybrid']['fp']
        and speech_hybrid['fn'] <= baseline['hybrid']['fn']
        and (speech_model['f1'] or 0) >= (baseline['model']['f1'] or 0)
        and written['recall'] >= baseline['written']['recall']
        and (written['f1'] or 0) >= (baseline['written']['f1'] or 0))


def selection_grid(epoch, y, scores, flags, missing, written_y, written_scores, baseline):
    points = []
    for threshold in np.linspace(.05, .95, 91):
        model = binary_metrics(y, scores, threshold)
        hybrid = hybrid_metrics(y, scores, flags, threshold, missing)
        written = binary_metrics(written_y, written_scores, threshold)
        points.append(dict(epoch=epoch, threshold=float(threshold), eligible=eligible(model, hybrid, written, baseline),
            speech_log_loss=model['log_loss'], speech_model_f1=model['f1'], speech_model_recall=model['recall'],
            **{f'speech_hybrid_{k}': hybrid[k] for k in ('fp','fn','precision','recall','f1','accuracy')},
            **{f'written_{k}': written[k] for k in ('fp','fn','precision','recall','f1','accuracy','log_loss')}))
    return max(points, key=rank), points


def components(clips, manifest):
    rules = ToxicRules(); flags, missing = {}, {}
    for split, items in clips.items():
        flags[split] = np.array([any(lexical_decision(rules.match(s['text']), s)['flagged']
                                   for s in item['segments']) for item in items], dtype=bool)
        included = {x['row']['video_id'] for x in items}
        missing[split] = [r['training_label'] for r in manifest['speech']
                         if r['split'] == split and r['video_id'] not in included]
    return flags, missing


def configure():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def train():
    protocol, manifest = load()
    if (PUBLIC / 'selection.json').exists():
        raise ValueError('Selection frozen; use evaluate, do not retrain this run')
    if not torch.cuda.is_available():
        raise RuntimeError('Use .venv-train with CUDA')
    configure(); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    target = PRIVATE / 'toxic'; target.mkdir(exist_ok=True)
    config = protocol['training']; device = 'cuda'
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    clips, coverage = encode_clips(PRIVATE, manifest, tokenizer, ('train', 'valid'))
    write_csv(PUBLIC / 'coverage.csv', coverage)
    flags, missing = components(clips, manifest)
    written_train = read(WRITTEN / 'tokens-train.json')['rows']
    written_valid = read(WRITTEN / 'tokens-valid.json')['rows']
    written_y = [r['labels'] for r in written_valid]
    model = AutoModelForSequenceClassification.from_pretrained(BASE, local_files_only=True).to(device)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(('roberta.encoder.layer.10.', 'roberta.encoder.layer.11.', 'classifier.')))
    baseline_path = target / 'baseline.npz'
    if not baseline_path.exists():
        np.savez(baseline_path, **{s: predict(model, clips[s], device) for s in clips},
                 written=predict_written(model, written_valid, tokenizer, device))
    with np.load(baseline_path, allow_pickle=False) as cache:
        baseline_scores = {key: cache[key] for key in cache.files}
    threshold = read(BASE / 'kidsafe-config.json')['threshold']
    baseline = dict(model=binary_metrics(labels(clips['valid']), baseline_scores['valid'], threshold),
        hybrid=hybrid_metrics(labels(clips['valid']), baseline_scores['valid'], flags['valid'], threshold, missing['valid']),
        written=binary_metrics(written_y, baseline_scores['written'], threshold))
    write_json(PUBLIC / 'baseline-validation.json', baseline)
    calibrated, base_grid = selection_grid(0, labels(clips['valid']), baseline_scores['valid'], flags['valid'],
        missing['valid'], written_y, baseline_scores['written'], baseline)
    write_json(PUBLIC / 'baseline-calibration.json', calibrated)
    write_csv(PUBLIC / 'baseline-thresholds.csv', base_grid)
    optimizer = torch.optim.AdamW([
        dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('classifier.')], lr=config['encoder_lr']),
        dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('classifier.')], lr=config['classifier_lr'])],
        weight_decay=config['weight_decay'])
    scaler = torch.amp.GradScaler('cuda')
    history, written_history, search, best, start = [], [], [], None, 1
    resume = target / 'resume.pt'
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=True)
        if state['protocol_sha256'] != sha(PUBLIC / 'protocol.json'):
            raise ValueError('Resume protocol mismatch')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        history, written_history, search, best, start = state['history'], state['written_history'], state['search'], state['best'], state['epoch']+1
    for epoch in range(start, config['epochs']+1):
        started = time.perf_counter(); model.train(); optimizer.zero_grad(set_to_none=True)
        order = np.random.default_rng(42+epoch).permutation(len(clips['train']))
        speech_loss_sum = replay_loss_sum = 0.
        accumulation = config['accumulate_clips']
        for step, index in enumerate(order):
            item = clips['train'][index]
            target_label = torch.tensor(float(item['row']['training_label']), device=device)
            divisor = min(accumulation, len(order)-(step//accumulation)*accumulation)
            # Separate backward calls bound activation memory; same summed objective.
            with torch.autocast('cuda', dtype=torch.float16):
                speech_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    forward_clip(model, item['tokens'], device).float(), target_label)
            scaler.scale(speech_loss/divisor).backward()
            chosen = rehearsal_indices(written_train, config['replay_batch'], 42+epoch*10000+step)
            batch = next(iter(loader([written_train[int(i)] for i in chosen], tokenizer, config['replay_batch'])))
            with torch.autocast('cuda', dtype=torch.float16):
                replay_loss = model(**{k:v.to(device) for k,v in batch.items()}).loss
            scaler.scale(config['replay_weight']*replay_loss/divisor).backward()
            speech_loss_sum += float(speech_loss.detach()); replay_loss_sum += float(replay_loss.detach())
            if (step+1)%accumulation == 0 or step+1 == len(order):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            if (step+1)%40 == 0:
                print(f'Epoch {epoch}: {step+1}/{len(order)} speech_loss={speech_loss_sum/(step+1):.4f} replay_loss={replay_loss_sum/(step+1):.4f}', flush=True)
        scores = {s: predict(model, clips[s], device) for s in clips}
        written_scores = predict_written(model, written_valid, tokenizer, device)
        point, grid = selection_grid(epoch, labels(clips['valid']), scores['valid'], flags['valid'],
            missing['valid'], written_y, written_scores, baseline)
        search.extend(grid)
        if best is None or rank(point) > rank(best):
            best = point; atomic_save(model.state_dict(), target / 'best.pt')
        for split in clips:
            history.append(dict(epoch=epoch, split=split, optimizer_speech_loss=speech_loss_sum/len(order),
                optimizer_replay_loss=replay_loss_sum/len(order), elapsed_seconds=time.perf_counter()-started,
                **binary_metrics(labels(clips[split]), scores[split], .5)))
        written_history.append(dict(epoch=epoch, **binary_metrics(written_y, written_scores, threshold)))
        write_csv(PUBLIC / 'training_history.csv', history)
        write_csv(PUBLIC / 'written_history.csv', written_history)
        write_csv(PUBLIC / 'validation_thresholds.csv', search)
        write_json(PUBLIC / 'progress.json', dict(completed_epoch=epoch, total_epochs=config['epochs'], best=best))
        atomic_save(dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), best=best, history=history,
            written_history=written_history, search=search, protocol_sha256=sha(PUBLIC / 'protocol.json')), resume)
        print(f'Epoch {epoch} complete: eligible={point["eligible"]} threshold={point["threshold"]:.2f} speech FP/FN={point["speech_hybrid_fp"]}/{point["speech_hybrid_fn"]} written recall={point["written_recall"]:.3f}', flush=True)
    model.load_state_dict(torch.load(target / 'best.pt', map_location='cpu', weights_only=True))
    selected = target / 'selected'; model.save_pretrained(selected, safe_serialization=True); tokenizer.save_pretrained(selected)
    write_json(selected / 'kidsafe-config.json', dict(version=f'toxic-phobert-{RUN}', threshold=best['threshold'],
        max_length=192, segmentation='pyvi-0.1.1', normalization='NFC-casefold-whitespace-v1'))
    scores = {s: predict(model, clips[s], device) for s in clips}
    written_scores = predict_written(model, written_valid, tokenizer, device)
    np.savez(target / 'selected-dev.npz', **scores, written=written_scores)
    write_json(target / 'dev-ids.json', {s: [item['row']['video_id'] for item in clips[s]] for s in clips})
    write_json(PUBLIC / 'selection.json', dict(selected=best, baseline=baseline,
        baseline_calibration=calibrated, protocol_sha256=sha(PUBLIC / 'protocol.json'),
        files={p.name: sha(p) for p in selected.iterdir() if p.is_file()},
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        selected_utc=datetime.now(timezone.utc).isoformat(), test_evaluated=False, deployment='candidate_only'))
    write_csv(PUBLIC / 'written-validation.csv', [dict(variant=variant, **binary_metrics(written_y, values, t))
        for variant, values, t in [('baseline', baseline_scores['written'], threshold),
                                  ('selected', written_scores, best['threshold']),
                                  ('selected_same_threshold', written_scores, threshold)]])
    plot_history(PUBLIC, history)
    print(dict(selection=best, baseline_calibration=calibrated), flush=True)


def prepare_test(manifest):
    """Only called after model/threshold selection; reuse audio traces, never old predictions."""
    from ml.asr_recovery import repeated_asr, recover_repetition
    model = None
    for row in manifest['speech']:
        if row['split'] != 'test':
            continue
        path = PRIVATE / 'asr' / f"{row['video_id']}.json"
        if path.exists():
            record = read(path)
            if record['sha256'] != row['sha256']:
                raise ValueError('Test ASR mismatch')
            continue
        old_path = ROOT / 'runs/local-adaptation-2026-10-29/cpu_holdout' / f"{row['video_id']}.json"
        old = read(old_path)
        if old['sha256'] != row['sha256'] or sha(ROOT / row['path']) != row['sha256']:
            raise ValueError('Test media changed')
        segments = old['segments']; warning = None
        if repeated_asr(segments):
            from faster_whisper import WhisperModel
            from faster_whisper.audio import decode_audio
            if model is None:
                model = WhisperModel(str(ROOT / 'models/checkpoints/faster-whisper-small'), device='cpu',
                    compute_type='int8', cpu_threads=4, local_files_only=True)
            audio = decode_audio(str(ROOT / row['path']), sampling_rate=16000)
            raw, info = model.transcribe(audio, task='transcribe', beam_size=5, temperature=0,
                vad_filter=True, condition_on_previous_text=False)
            segments = [dict(start=s.start, end=s.end, text=s.text, avg_logprob=s.avg_logprob,
                compression_ratio=s.compression_ratio, no_speech_prob=s.no_speech_prob)
                for s in raw if s.text.strip()] if info.language == 'vi' and info.language_probability >= .5 else []
            segments, warning = recover_repetition(model, audio, segments,
                language=info.language, language_probability=info.language_probability)
        write_json(path, dict(video_id=row['video_id'], sha256=row['sha256'], segments=segments,
            source_asr_sha256=sha(old_path), recovery_warning=warning,
            selection_sha256=sha(PUBLIC / 'selection.json')))


def evaluate():
    from ml.deep_inference import PhoBertClassifier
    protocol, manifest = load(); configure()
    if (PUBLIC / 'evaluation.json').exists():
        raise ValueError('Evaluation already frozen; do not retune from test')
    selection = read(PUBLIC / 'selection.json'); selected = PRIVATE / 'toxic/selected'
    for name, digest in selection['files'].items():
        if sha(selected / name) != digest:
            raise ValueError('Selected checkpoint changed')
    eval_protocol = dict(selection_sha256=sha(PUBLIC / 'selection.json'),
        scope='Speech train/valid and reused 30-clip test; single-reviewer labels; CPU app classifier',
        policy='No tuning; missing ASR excludes probability metrics and counts as unflagged/review for decision metrics')
    if (PUBLIC / 'evaluation-protocol.json').exists() and read(PUBLIC / 'evaluation-protocol.json') != eval_protocol:
        raise ValueError('Evaluation protocol changed')
    write_json(PUBLIC / 'evaluation-protocol.json', eval_protocol)
    prepare_test(manifest)
    rules = ToxicRules(); all_rows = []; metric_rows = []; decisions = []; plots = {}
    parity_changes = []; max_delta = 0.
    dev_ids = read(PRIVATE / 'toxic/dev-ids.json')
    for variant, directory in (('baseline', BASE), ('selected', selected)):
        model = PhoBertClassifier(directory / 'kidsafe-config.json')
        reference = np.load(PRIVATE / 'toxic' / ('baseline.npz' if variant == 'baseline' else 'selected-dev.npz'), allow_pickle=False)
        plots[variant] = {}
        for split in ('train', 'valid', 'test'):
            part = []
            expected = dict(zip(dev_ids.get(split, []), reference[split])) if split != 'test' else {}
            for row in manifest['speech']:
                if row['split'] != split:
                    continue
                path = PRIVATE / 'cpu' / variant / f"{row['video_id']}.json"
                if path.exists():
                    result = read(path)
                    if result['selection_sha256'] != eval_protocol['selection_sha256'] or result['asr_sha256'] != sha(PRIVATE/'asr'/f"{row['video_id']}.json"):
                        raise ValueError('CPU result cache changed')
                else:
                    segments = read(PRIVATE / 'asr' / f"{row['video_id']}.json")['segments']
                    scores = [model.predict(s['text'])['score'] for s in segments if s['text'].strip()]
                    score = max(scores) if scores else None
                    rule = any(lexical_decision(rules.match(s['text']), s)['flagged'] for s in segments)
                    result = dict(video_id=row['video_id'], split=split, variant=variant, label=row['training_label'],
                        score=score, threshold=model.threshold, rule_flag=rule,
                        model_flag=score is not None and score >= model.threshold,
                        hybrid_flag=rule or (score is not None and score >= model.threshold),
                        selection_sha256=eval_protocol['selection_sha256'], asr_sha256=sha(PRIVATE/'asr'/f"{row['video_id']}.json"))
                    write_json(path, result)
                if result['score'] is not None and row['video_id'] in expected:
                    delta = abs(result['score']-float(expected[row['video_id']]))
                    max_delta = max(delta, max_delta)
                    if result['model_flag'] != bool(expected[row['video_id']] >= model.threshold):
                        parity_changes.append(dict(video_id=row['video_id'], variant=variant))
                part.append(result)
            covered = [r for r in part if r['score'] is not None]
            y, p = [r['label'] for r in covered], [r['score'] for r in covered]
            metric_rows.append(dict(task='toxic', variant=variant, split=split, **binary_metrics(y, p, model.threshold)))
            for method, field in (('model_only', 'model_flag'), ('rule_only', 'rule_flag'), ('hybrid', 'hybrid_flag')):
                decisions.append(dict(variant=variant, split=split, method=method,
                    missing_asr=len(part)-len(covered), **decision_metrics([r['label'] for r in part], [r[field] for r in part])))
            plots[variant][split] = (np.array(y), np.array(p)); all_rows.extend(part)
            print(f'CPU {variant} {split}: {len(part)} clips complete', flush=True)
        reference.close(); del model
    write_csv(PRIVATE / 'cpu-predictions.csv', all_rows)
    write_csv(PUBLIC / 'metrics_by_split.csv', metric_rows)
    write_csv(PUBLIC / 'decisions_by_split.csv', decisions)
    write_json(PUBLIC / 'cpu-parity.json', dict(max_score_delta=max_delta, changed_decisions=len(parity_changes), scope='All covered train/validation clips; CPU vs FP32 CUDA'))
    write_json(PRIVATE / 'cpu-parity-details.json', parity_changes)
    plot_results(PUBLIC, metric_rows, plots, [], label_description='Nhãn Huy nghe duyệt',
        scope_note='Toxic model-only · nhãn một người duyệt · test cũ tái sử dụng, không phải test độc lập mới')
    plot_decisions(decisions)
    write_json(PUBLIC / 'evaluation.json', dict(**eval_protocol, probability_metrics=metric_rows,
        decision_metrics=decisions, cpu_parity_changed=len(parity_changes), max_cpu_gpu_delta=max_delta,
        deployment='candidate_only', independent_test=False, reference_wer_cer=None))
    print([r for r in decisions if r['split'] == 'test'], flush=True)


def plot_decisions(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    for ax, metric in zip(axes, ('fp', 'fn', 'f1')):
        for i, variant in enumerate(('baseline','selected')):
            values = [next(r for r in rows if r['variant']==variant and r['split']==s and r['method']=='hybrid')[metric]
                      for s in ('train','valid','test')]
            bars = ax.bar(np.arange(3)+i*.35, values, .35, label=variant)
            ax.bar_label(bars, fmt='%.3f' if metric=='f1' else '%.0f')
        ax.set_xticks(np.arange(3)+.175, ['train','validation','test cũ']); ax.set_title('Hybrid '+metric); ax.legend()
    fig.suptitle('Toxic + bộ luật · 200 nhãn Huy · không có test độc lập mới')
    for suffix in ('png','svg'):
        fig.savefig(PUBLIC / f'hybrid_comparison.{suffix}', dpi=160)
    clean_svg(PUBLIC / 'hybrid_comparison.svg'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('train','evaluate'))
    {'train':train, 'evaluate':evaluate}[parser.parse_args().command]()
