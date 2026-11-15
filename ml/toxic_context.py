"""Development-only experiment: adjacent ASR context with frozen classifiers.

Never loads test ASR or fits thresholds. Raw predictions stay under runs/.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ml.experiment_metrics import binary_metrics, clean_svg, write_csv, write_json
from ml.hybrid_evaluation import decision_metrics
from ml.toxic_baseline import normalize
from ml.toxic_reviewed import ROOT, BASE, PRIVATE as SOURCE, read, sha
from ml.toxic_rules import ToxicRules, lexical_decision

RUN = 'toxic-context-2026-11-15'
PRIVATE = ROOT / 'runs' / RUN
PUBLIC = ROOT / 'reports' / RUN
MODELS = {'baseline': BASE, 'human': SOURCE / 'toxic/selected'}
MAX_LENGTH = 192
MAX_GAP = 2.0


def development_rows(manifest):
    """Check group boundaries, then exclude test before accessing its content."""
    groups, hashes, ids = {}, {}, set()
    for row in manifest['speech']:
        if row['split'] not in ('train', 'valid', 'test') or row['video_id'] in ids:
            raise ValueError('Invalid split or duplicate ID')
        ids.add(row['video_id'])
        for registry, key in ((groups, row['training_group']), (hashes, row['sha256'])):
            if key in registry and registry[key] != row['split']:
                raise ValueError('Group/media crosses split boundary')
            registry[key] = row['split']
    return [r for r in manifest['speech'] if r['split'] in ('train', 'valid')]


def center_window(before, center, after, budget):
    """Keep the center first; allocate spare tokens to closest left/right context."""
    if budget < 1 or not center:
        raise ValueError('A nonempty center and positive budget are required')
    center = list(center[:budget])
    spare = budget - len(center)
    left = min(len(before), spare // 2)
    right = min(len(after), spare - left)
    left = min(len(before), spare - right)
    return (list(before[-left:]) if left else []) + center + list(after[:right])


def passes_gate(candidate, reference):
    return (candidate['fp'] < reference['fp'] and candidate['fn'] <= reference['fn']
            and candidate['f1'] >= reference['f1'])


def encode_windows(tokenizer, segments):
    from pyvi import ViTokenizer
    pieces = [tokenizer.encode(ViTokenizer.tokenize(normalize(s['text'])),
                               add_special_tokens=False) for s in segments]
    budget = MAX_LENGTH - tokenizer.num_special_tokens_to_add(pair=False)
    windows, audit = [], Counter()
    for i, center in enumerate(pieces):
        before = pieces[i-1] if i and segments[i]['start'] - segments[i-1]['end'] <= MAX_GAP else []
        after = pieces[i+1] if i+1 < len(pieces) and segments[i+1]['start'] - segments[i]['end'] <= MAX_GAP else []
        tokens = center_window(before, center, after, budget)
        windows.append(tokenizer.build_inputs_with_special_tokens(tokens))
        audit['windows'] += 1
        audit['center_truncated'] += len(center) > budget
        audit['context_added'] += len(tokens) > min(len(center), budget)
    isolated = [tokenizer.build_inputs_with_special_tokens(p[:budget]) for p in pieces]
    return {'isolated': isolated, 'context': windows}, dict(audit)


def input_inventory(rows):
    paths = [SOURCE / 'manifest.json', ROOT / 'models/artifacts.json',
             ROOT / 'ml/resources/toxic_lexicon_v1.json']
    paths += [SOURCE / 'asr' / f"{r['video_id']}.json" for r in rows]
    paths += [SOURCE / 'cpu' / variant / f"{r['video_id']}.json"
              for variant in ('baseline', 'selected') for r in rows]
    paths += [p for directory in MODELS.values() for p in directory.iterdir() if p.is_file()]
    paths += [ROOT / name for name in ('ml/toxic_context.py', 'ml/toxic_rules.py',
              'ml/toxic_baseline.py', 'ml/hybrid_evaluation.py', 'ml/experiment_metrics.py')]
    return {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in paths}


def prepare():
    if PRIVATE.exists() or PUBLIC.exists():
        raise ValueError('Run exists; resume score, preserve the original protocol')
    rows = development_rows(read(SOURCE / 'manifest.json'))
    inventory = input_inventory(rows)
    PRIVATE.mkdir(parents=True); PUBLIC.mkdir(parents=True)
    write_json(PRIVATE / 'manifest.json', {'speech': rows})
    write_json(PRIVATE / 'inputs.json', inventory)
    protocol = dict(run_id=RUN, frozen_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis='Adjacent ASR context can reduce isolated-segment false alarms',
        splits=dict(Counter(r['split'] for r in rows)),
        checkpoints={k: {'directory': str(v.relative_to(ROOT)),
            'model_sha256': sha(v/'model.safetensors'), 'threshold': read(v/'kidsafe-config.json')['threshold']}
            for k,v in MODELS.items()},
        methods=['isolated', 'context'], context='Previous + center + next ASR segment, within 2-second gaps',
        token_policy='Center-first 192-token budget incl special tokens; split spare left/right; no cross-clip context',
        threshold_search=False, fit=False, test_scored=False,
        validation_reused=True, independent_test=False,
        gate='Strictly fewer validation hybrid FP, no higher FN and no lower F1 versus BOTH isolated checkpoints',
        ranking='Eligible first; fewer validation FP; fewer FN; higher F1; baseline checkpoint on ties',
        rule_policy='Unchanged rules on original isolated ASR segments; do not label context windows',
        inference='CUDA FP32 for dev experiment, no autocast/TF32; CPU check only selected eligible candidate',
        deployment='offline_candidate_only; no runtime activation from reused validation',
        manifest_sha256=sha(PRIVATE/'manifest.json'), inventory_sha256=sha(PRIVATE/'inputs.json'))
    write_json(PUBLIC/'protocol.json', protocol)
    write_json(PUBLIC/'progress.json', {'stage': 'prepared', 'completed_models': []})
    print('Frozen dev-only protocol:', protocol['splits'], flush=True)


def load():
    protocol = read(PUBLIC/'protocol.json')
    if sha(PRIVATE/'manifest.json') != protocol['manifest_sha256'] or sha(PRIVATE/'inputs.json') != protocol['inventory_sha256']:
        raise ValueError('Frozen input metadata changed')
    for path, digest in read(PRIVATE/'inputs.json').items():
        if sha(ROOT/path) != digest:
            raise ValueError(f'Frozen input changed: {path}')
    rows = development_rows(read(PRIVATE/'manifest.json'))
    if any(r['split'] == 'test' for r in rows) or len(rows) != sum(protocol['splits'].values()):
        raise ValueError('Development inventory changed')
    return protocol, rows


def infer(model, tokenizer, windows, device):
    import torch
    if not windows:
        return []
    result = []
    with torch.inference_mode():
        for begin in range(0, len(windows), 8):
            batch = tokenizer.pad({'input_ids': windows[begin:begin+8]}, padding=True, return_tensors='pt')
            probabilities = torch.softmax(model(**{k:v.to(device) for k,v in batch.items()}).logits.float(), dim=-1)[:,1]
            result.extend(probabilities.cpu().tolist())
    return result


def score():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    protocol, rows = load()
    if (PUBLIC/'evaluation.json').exists():
        raise ValueError('Experiment complete; do not rescore or retune')
    if not torch.cuda.is_available():
        raise RuntimeError('Use .venv-train for the recorded CUDA experiment')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rules = ToxicRules(); records = []; counts = Counter()
    for variant, directory in MODELS.items():
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(directory, local_files_only=True).to('cuda').eval()
        threshold = protocol['checkpoints'][variant]['threshold']
        for i, row in enumerate(rows):
            path = PRIVATE/'predictions'/variant/f"{row['video_id']}.json"
            if path.exists():
                result = read(path)
                if result['protocol_sha256'] != sha(PUBLIC/'protocol.json'):
                    raise ValueError('Prediction cache protocol mismatch')
            else:
                trace = read(SOURCE/'asr'/f"{row['video_id']}.json")
                if trace['sha256'] != row['sha256']:
                    raise ValueError('ASR/media hash mismatch')
                segments = [s for s in trace['segments'] if normalize(s['text'])]
                windows, audit = encode_windows(tokenizer, segments)
                probabilities = {method: infer(model,tokenizer,tokens,'cuda') for method,tokens in windows.items()}
                rule = any(lexical_decision(rules.match(s['text']),s)['flagged'] for s in segments)
                result = dict(video_id=row['video_id'], split=row['split'], label=row['training_label'],
                    variant=variant, threshold=threshold, rule_flag=rule, audit=audit,
                    segment_scores=probabilities, protocol_sha256=sha(PUBLIC/'protocol.json'))
                write_json(path,result)
            for method, probabilities in result['segment_scores'].items():
                value = max(probabilities) if probabilities else None
                flag = value is not None and value >= threshold
                records.append(dict(video_id=row['video_id'], split=row['split'], label=row['training_label'],
                    variant=variant, method=method, threshold=threshold, score=value,
                    rule_flag=result['rule_flag'], model_flag=flag, hybrid_flag=result['rule_flag'] or flag))
            if variant == 'baseline': counts.update(result['audit'])
            if (i+1)%40 == 0: print(f'{variant}: {i+1}/{len(rows)} dev clips', flush=True)
        del model; torch.cuda.empty_cache()
        write_json(PUBLIC/'progress.json', {'stage':'scoring', 'last_completed_model':variant})
    write_csv(PRIVATE/'predictions.csv',records)
    metrics=[]
    for variant in MODELS:
        for method in ('isolated','context'):
            for split in ('train','valid'):
                part=[r for r in records if r['variant']==variant and r['method']==method and r['split']==split]
                covered=[r for r in part if r['score'] is not None]
                probability=binary_metrics([r['label'] for r in covered],[r['score'] for r in covered],part[0]['threshold'])
                metrics.append(dict(variant=variant,method=method,split=split,missing_asr=len(part)-len(covered),
                    **decision_metrics([r['label'] for r in part],[r['hybrid_flag'] for r in part]),
                    model_log_loss=probability['log_loss'], model_covered_n=len(covered), model_f1=probability['f1']))
    refs=[r for r in metrics if r['split']=='valid' and r['method']=='isolated']
    candidates=[dict(r,eligible=all(passes_gate(r,ref) for ref in refs))
                for r in metrics if r['split']=='valid' and r['method']=='context']
    winner=max(candidates,key=lambda r:(r['eligible'],-r['fp'],-r['fn'],r['f1'],r['variant']=='baseline'))
    # Check isolated scoring against the previous CPU run; this is not a test evaluation.
    parity=[]
    for r in records:
        if r['method']!='isolated': continue
        old=read(SOURCE/'cpu'/('baseline' if r['variant']=='baseline' else 'selected')/f"{r['video_id']}.json")
        parity.append(dict(changed=r['hybrid_flag']!=old['hybrid_flag'],
                           delta=abs(r['score']-old['score']) if r['score'] is not None else 0.))
    write_csv(PUBLIC/'metrics.csv',metrics)
    write_json(PUBLIC/'selection.json',dict(candidates=candidates,selected=winner if winner['eligible'] else None,
        protocol_sha256=sha(PUBLIC/'protocol.json'),test_scored=False,deployment='offline_candidate_only'))
    result=dict(metrics=metrics,token_audit=dict(counts),test_scored=False,training_performed=False,
        independent_test=False,isolated_cpu_reference_changed=sum(r['changed'] for r in parity),
        isolated_cpu_reference_max_delta=max(r['delta'] for r in parity),candidate_eligible=winner['eligible'],
        runtime_changed=False,violence_changed=False)
    write_json(PUBLIC/'evaluation.json',result)
    plot(metrics)
    write_json(PUBLIC/'progress.json',dict(stage='complete',test_scored=False,candidate_eligible=winner['eligible']))
    print('Validation:',[r for r in metrics if r['split']=='valid'],flush=True)


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    subset=[r for r in rows if r['split']=='valid']
    names=[('v3' if r['variant']=='baseline' else '13/11')+'\n'+('từng đoạn' if r['method']=='isolated' else 'thêm ngữ cảnh') for r in subset]
    for ax,key in zip(axes,('fp','fn','f1')):
        bars=ax.bar(names,[r[key] for r in subset],color=['#7098b8','#176978','#b5a0ce','#725698'])
        ax.bar_label(bars,fmt='%.3f' if key=='f1' else '%.0f')
        ax.set_title({'fp':'Cảnh báo nhầm (FP)','fn':'Bỏ sót (FN)','f1':'F1 hybrid'}[key]);ax.margins(y=.2)
    fig.suptitle('Validation 28 clip đã dùng trước · ngưỡng giữ nguyên · chưa chấm test / chưa đổi app')
    for ext in ('png','svg'): fig.savefig(PUBLIC/f'context_comparison.{ext}',dpi=160)
    clean_svg(PUBLIC/'context_comparison.svg');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','score'))
    {'prepare':prepare,'score':score}[parser.parse_args().command]()
