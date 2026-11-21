"""Export aggregate comparisons and a private review queue from completed evaluations."""
import argparse

from ml.experiment_metrics import clean_svg, write_csv, write_json
from ml.local_adaptation import load_frozen, read


def export(run_id):
    private,public,protocol,manifest=load_frozen(run_id)
    cpu=read(public/'toxic/cpu_holdout.json');violence=read(public/'violence/evaluation.json')
    rows=[]
    for r in cpu['decision_metrics']:
        rows.append(dict(task='toxic',dataset='speech30-weak-labels',split='test',scope='CPU ASR; review counted as not flagged',
                         log_loss=None,**r))
    for r in cpu['probability_metrics']:
        rows.append(dict(task='toxic',dataset='speech29-weak-labels-ASR-eligible',split='test',method='model_only',
                         scope='CPU ASR; missing ASR excluded from probability metrics',**r))
    for r in violence['metrics']:
        if r['split']=='test':
            rows.append(dict(method='model_only',scope='RLVS previously reported test benchmark',**r))
    write_csv(public/'summary.csv',rows)
    write_json(public/'deployment_decision.json',dict(
        activate=False,active_model='kidsafe-cpu-v3 + toxic-hybrid-v1-56b3a177c7b2',
        toxic='CPU weak-label holdout: hybrid FP6 ->8 despite FN6 ->2; not sufficient evidence to replace the app',
        violence='No candidate met validation FP/recall constraints; RLVS test FN5 ->14, F1 declined',
        test_policy='Do not retune on these observed holdouts; later experiments require a new protocol and clear test-exposure status',
        checkpoints='Candidates retained locally, existing models not overwritten'))
    lookup={r['video_id']:r for r in manifest['speech']}
    review=[]
    for path in sorted((private/'cpu_holdout').glob('review-*.json')):
        r=read(path);pred=r['selected']['hybrid_flag'];meta=lookup[r['video_id']]
        if bool(r['label'])==bool(pred):continue
        review.append(dict(video_id=r['video_id'],original_video_id=meta['original_video_id'],path=meta['path'],
                           weak_label=r['label'],outcome_against_weak_label='FN' if r['label'] else 'FP',
                           model_score=r['selected']['score'],rule_flag=r['rule_flag'],human_label='',
                           human_transcript='',reviewer_notes='',review_status='post_prediction_review_not_blind'))
    write_csv(private/'cpu_error_review.csv',review)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    chosen=[r for r in cpu['decision_metrics'] if r['method']=='hybrid']
    x=[0,1];names=['v3 + luật','Ứng viên + luật']
    for offset,key,label,color in ((-.18,'fp','Báo nhầm','#ce765c'),(.18,'fn','Bỏ sót','#7196b4')):
        bars=axes[0].bar([v+offset for v in x],[r[key] for r in chosen],.36,label=label,color=color)
        axes[0].bar_label(bars)
    axes[0].set(xticks=x,xticklabels=names,ylim=(0,11),title='Toxic CPU · 30 clip nhãn yếu')
    axes[0].legend()
    test=[r for r in violence['metrics'] if r['split']=='test']
    for offset,key,label,color in ((-.18,'fp','Báo nhầm','#ce765c'),(.18,'fn','Bỏ sót','#7196b4')):
        bars=axes[1].bar([v+offset for v in x],[r[key] for r in test],.36,label=label,color=color)
        axes[1].bar_label(bars)
    axes[1].set(xticks=x,xticklabels=['v3','Ứng viên'],ylim=(0,18),title='Violence · RLVS test 386 clip')
    axes[1].legend()
    fig.suptitle('Đánh đổi sau huấn luyện — giữ app hiện tại, lưu ứng viên riêng')
    for ext in ('png','svg'):
        target=public/f'error_tradeoffs.{ext}';fig.savefig(target,dpi=180)
        if ext=='svg':clean_svg(target)
    plt.close(fig)
    print(f'Exported summary and {len(review)} private weak-label disagreements',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True)
    export(parser.parse_args().run_id)
