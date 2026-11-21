"""Draw the two report figures from measured CSVs with the project's ML environment."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/toxic-transcript-2026-11-20'


def load_csv(path):
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def style(ax):
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#b9c9c9')
    ax.tick_params(length=0, pad=6)
    ax.set_axisbelow(True)


def main():
    teal, clay, gray = '#08786f', '#ac4b29', '#586e82'
    records = {r['method']:r for r in load_csv(OUT/'decision_metrics.csv') if r['split']=='valid'}
    methods = ['rule','baseline','selected','text+rule']
    fig, ax = plt.subplots(figsize=(8,2.8), layout='constrained')
    pos = np.arange(len(methods))
    for delta, key, label, color in [(-.18,'fp','Báo nhầm',clay),(.18,'fn','Bỏ sót',teal)]:
        bars = ax.barh(pos+delta, [int(records[m][key]) for m in methods],.32,color=color,label=label)
        ax.bar_label(bars,padding=5,fontsize=10)
    ax.set_yticks(pos,['Luật từ / cụm từ','PhoBERT v3 + luật','PhoBERT 13/11 + luật','Ký tự + luật'])
    ax.set_xticks(range(5)); ax.set_xlim(0,4.5); ax.invert_yaxis()
    ax.grid(axis='x',alpha=.15); ax.legend(loc='lower right',frameon=False,ncol=2)
    ax.set_xlabel('Số clip trên tập validation'); style(ax)
    fig.savefig(OUT/'errors.png',dpi=180); plt.close(fig)

    rows = [r for r in load_csv(ROOT/'reports/toxic-human-2026-11-13/training_history.csv') if r['split']=='valid']
    training = [r for r in load_csv(ROOT/'reports/toxic-human-2026-11-13/training_history.csv') if r['split']=='train']
    epochs = [int(r['epoch']) for r in rows]
    fig, axes = plt.subplots(1,2,figsize=(8.4,2.8),layout='constrained')
    for values,label,color in [(training,'Train',gray),(rows,'Validation',teal)]:
        axes[0].plot(epochs,[float(r['log_loss']) for r in values],'o-',label=label,color=color)
    axes[0].set_title('Log-loss cấp clip'); axes[0].set_ylim(.45,1.4)
    for k,label,color in [('average_precision','AP',teal),('accuracy','Accuracy',gray),('f1','F1 Toxic',clay)]:
        axes[1].plot(epochs,[float(r[k]) for r in rows],'o-',label=label,color=color)
    axes[1].set_title('Chỉ số validation'); axes[1].set_ylim(.48,.91)
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.set_xticks(epochs); ax.legend(frameon=False,ncol=3,fontsize=8,loc='upper center')
        ax.grid(axis='y',alpha=.15); style(ax)
    fig.savefig(OUT/'learning.png',dpi=180); plt.close(fig)


if __name__=='__main__':
    main()
