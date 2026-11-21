"""Plot recorded RLVS/AIRTLab results without additional inference."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ml.experiment_metrics import clean_svg
from ml.toxic_baseline import ROOT


def plot(run_id):
    public = ROOT / "reports" / run_id
    rlvs = json.loads((public / "violence/evaluation.json").read_text(encoding="utf-8"))["metrics"]
    external = json.loads((public / "airtlab_evaluation.json").read_text(encoding="utf-8"))["metrics"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
    for ax, key in zip(axes, ("accuracy", "macro_f1", "recall")):
        for i, variant in enumerate(("baseline", "selected")):
            values = [next(r[key] for r in rlvs if r["variant"] == variant and r["split"] == "test"),
                      next(r[key] for r in external if r["variant"] == variant)]
            bars = ax.bar(np.arange(2)+(i-.5)*.35, values, .35, label="CPU v2" if i == 0 else "R3D-18 mới")
            ax.bar_label(bars, fmt="%.3f", fontsize=9)
        ax.set(xticks=[0,1], xticklabels=["RLVS test\nn=386", "AIRTLab nguồn\nn=100"], ylim=(0,1.13), title=key)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(loc="lower left")
    fig.suptitle("Violence: cải thiện benchmark, nhưng chuyển bối cảnh còn yếu")
    for suffix in ("png", "svg"):
        fig.savefig(public / f"external_generalization.{suffix}", dpi=160)
    clean_svg(public / "external_generalization.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    plot(parser.parse_args().run_id)
