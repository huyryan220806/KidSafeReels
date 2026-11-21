"""Post-selection holdout check with the app's CPU INT8 ASR and CPU text inference."""
import argparse
import time

import numpy as np

from ml.deep_inference import PhoBertClassifier
from ml.experiment_metrics import binary_metrics, write_csv, write_json
from ml.hybrid_evaluation import decision_metrics
from ml.local_adaptation import load_frozen, read
from ml.toxic_baseline import ROOT, sha256
from ml.toxic_rules import ToxicRules, lexical_decision


def evaluate_cpu(run_id):
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    private, public, protocol, manifest = load_frozen(run_id)
    public=public/'toxic';selection=read(public/'selection.json')
    target=private/'toxic/selected'
    if (public/'cpu_holdout.json').exists():raise ValueError('CPU holdout already evaluated')
    for name,digest in selection['files'].items():
        if sha256(target/name)!=digest:raise ValueError('Candidate changed')
    if sha256(ROOT/'models/checkpoints/faster-whisper-small/model.bin')!=protocol['asr']['model_sha256']:
        raise ValueError('ASR model changed')
    # Lock scope before CPU inference; neither thresholds nor rules are retuned.
    write_json(public/'cpu_holdout_protocol.json',dict(
        selected_sha256=sha256(public/'selection.json'),scope='Toxic branch only, frozen speech test groups',
        label_status='user-reported weak clip labels',asr='Whisper small CPU INT8 beam5, Vietnamese gate, VAD',
        methods=['model_only','hybrid'],missing_asr='null, excluded probability metrics; review in decision metrics',
        policy='post-selection deployment check; no tuning based on these results'))
    asr=WhisperModel(str(ROOT/'models/checkpoints/faster-whisper-small'),device='cpu',compute_type='int8',cpu_threads=4,local_files_only=True)
    classifiers={'baseline':PhoBertClassifier(ROOT/'models/checkpoints/toxic-phobert-v3/kidsafe-config.json'),
                 'selected':PhoBertClassifier(target/'kidsafe-config.json')}
    rules=ToxicRules();rows=[];test=[r for r in manifest['speech'] if r['split']=='test'];started=time.perf_counter()
    for i,row in enumerate(test,1):
        out=private/'cpu_holdout'/f"{row['video_id']}.json"
        if out.exists():
            saved=read(out)
            if saved['sha256']!=row['sha256'] or saved['selected_sha256']!=selection['files']['model.safetensors']:
                raise ValueError('CPU cache mismatch')
            rows.append(saved)
            continue
        path=ROOT/row['path']
        if sha256(path)!=row['sha256']:raise ValueError('Holdout file changed')
        seg,info=asr.transcribe(decode_audio(str(path),sampling_rate=16000),task='transcribe',beam_size=5,
                              temperature=0,vad_filter=True,condition_on_previous_text=False)
        segments=[dict(start=s.start,end=s.end,text=s.text,avg_logprob=s.avg_logprob,compression_ratio=s.compression_ratio)
                  for s in seg if s.text.strip()] if info.language=='vi' and info.language_probability>=.5 else []
        rule_flag=any(lexical_decision(rules.match(s['text']),s)['flagged'] for s in segments)
        record=dict(video_id=row['video_id'],sha256=row['sha256'],label=row['training_label'],segments=segments,
                    selected_sha256=selection['files']['model.safetensors'],rule_flag=rule_flag)
        for variant,model in classifiers.items():
            values=[model.predict(s['text'])['score'] for s in segments]
            score=max(values) if values else None
            record[variant]=dict(score=score,flagged=score is not None and score>=model.threshold,
                                 hybrid_flag=rule_flag or (score is not None and score>=model.threshold))
        write_json(out,record);rows.append(record)
        status=dict(completed=i,total=len(test),elapsed_seconds=time.perf_counter()-started)
        write_json(public/'cpu_holdout_progress.json',status)
        print(f'CPU holdout {i}/{len(test)}',flush=True)
    probability,decisions=[],[]
    for variant,model in classifiers.items():
        eligible=[r for r in rows if r[variant]['score'] is not None]
        probability.append(dict(variant=variant,**binary_metrics([r['label'] for r in eligible],
                            [r[variant]['score'] for r in eligible],model.threshold)))
        for method,key in (('model_only','flagged'),('hybrid','hybrid_flag')):
            decisions.append(dict(variant=variant,method=method,**decision_metrics([r['label'] for r in rows],
                                                                                  [r[variant][key] for r in rows])))
    write_csv(public/'cpu_holdout_probabilities.csv',probability)
    write_csv(public/'cpu_holdout_decisions.csv',decisions)
    write_json(public/'cpu_holdout.json',dict(probability_metrics=probability,decision_metrics=decisions,
        total=len(rows),missing_asr=sum(not r['segments'] for r in rows),
        limitation='Weak clip labels, provisional source groups; only Toxic branch, not total app accuracy',
        selection_sha256=sha256(public/'selection.json'),deployment='candidate_only'))
    print(probability,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True)
    evaluate_cpu(parser.parse_args().run_id)
