"""Describe observed dev errors, without inventing ASR reference transcripts."""
from collections import Counter
import json

from ml.experiment_metrics import write_csv, write_json
from ml.toxic_context import PRIVATE, PUBLIC, SOURCE, load, read, sha
from ml.toxic_rules import ToxicRules, lexical_decision


def audit():
    _, rows = load()
    rules = ToxicRules(); details=[]; summary=Counter(); evidence_counts=Counter()
    for row in rows:
        trace=read(SOURCE/'asr'/f"{row['video_id']}.json")
        segments=[s for s in trace['segments'] if s['text'].strip()]
        predicted=read(SOURCE/'cpu/selected'/f"{row['video_id']}.json")
        if bool(predicted['hybrid_flag']) == bool(row['training_label']):
            continue
        error='FP' if predicted['hybrid_flag'] else 'FN'
        raw_matches=[m for s in segments for m in rules.match(s['text']) if m['action']=='flag']
        if error=='FP':
            category='rule_flag_disagrees_with_clip_label' if predicted['rule_flag'] else 'classifier_only_false_alarm'
        elif not segments:
            category='missing_asr'
        elif raw_matches:
            category='keyword_present_but_audio_gate_rejects'
        else:
            category='below_model_threshold_and_no_rule_keyword'
        summary[(row['split'],error,category)]+=1
        evidence_audit=[]
        for evidence in row['annotation']['evidence']:
            matches=[m for m in rules.match(evidence['reason']) if m['action']=='flag']
            if not matches: continue
            overlap=[s for s in segments if s['end']>evidence['start_seconds'] and s['start']<evidence['end_seconds']]
            observed={m['id'] for s in overlap for m in rules.match(s['text']) if m['action']=='flag'}
            absent=not any(m['id'] in observed for m in matches)
            evidence_audit.append(dict(evidence=evidence,asr_overlap=overlap,
                any_same_rule_term_in_overlap=not absent))
            evidence_counts[(row['split'],error,'absent' if absent else 'present')]+=1
        details.append(dict(video_id=row['video_id'],split=row['split'],label=row['training_label'],
            error=error,category=category,score=predicted['score'],rule_flag=predicted['rule_flag'],
            human_evidence=row['annotation']['evidence'],asr=segments,evidence_alignment=evidence_audit,
            caution='Observed text/label disagreement only; no new human listening or verified full transcript'))
    write_json(PRIVATE/'development-audit-details.json',details)
    write_csv(PRIVATE/'development-audit.csv',[dict(video_id=r['video_id'],split=r['split'],label=r['label'],
        error=r['error'],category=r['category'],score=r['score'],rule_flag=r['rule_flag'],
        human_evidence=json.dumps(r['human_evidence'],ensure_ascii=False),
        asr_text=' '.join(s['text'] for s in r['asr'])) for r in details])
    aggregate=[dict(split=s,error=e,category=c,clips=n) for (s,e,c),n in sorted(summary.items())]
    write_csv(PUBLIC/'development-audit.csv',aggregate)
    write_json(PUBLIC/'development-audit.json',dict(
        reference='13 November selected checkpoint + unchanged rules',test_included=False,
        human_labels_changed=False,asr_reference_verified=False,wer=None,cer=None,
        source_sha256=sha(__file__),categories=aggregate,
        evidence_rule_term_alignment=[dict(split=s,error=e,term_in_time_overlap=status,evidence_items=n)
            for (s,e,status),n in sorted(evidence_counts.items())],
        interpretation='Exact lexicon-term/time overlap is a diagnostic signal, not WER or proof of ASR correctness',
        private_detail_sha256=sha(PRIVATE/'development-audit-details.json')))
    print(aggregate)


if __name__=='__main__':
    audit()
