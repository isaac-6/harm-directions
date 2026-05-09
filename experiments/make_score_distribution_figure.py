#!/usr/bin/env python3
"""
make_score_distribution_figure.py
----------------------------------
Score distributions for Qwen2.5-0.5B-Instruct and Llama-3.2-1B-Instruct,
under w_LDA and w_opt, with 1% FPR threshold marked.  (Figure 5 in paper)

2x2 grid: rows are models, columns are strategies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B-Instruct"),
    ("meta-llama/Llama-3.2-1B-Instruct", "Llama-3.2-1B-Instruct"),
]
STRATEGIES = [
    ("mean_diff", r"$\mathbf{w}_{\mathrm{LDA}}$"),
    ("soft_auc", r"$\mathbf{w}_{\mathrm{opt}}$"),
]


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def load_scores(artifacts_dir: Path, model_id: str, strategy: str) -> pd.DataFrame:
    p = artifacts_dir / model_slug(model_id) / "score_distributions.csv"
    df = pd.read_csv(p)
    return df[df["strategy"] == strategy]


def plot_one(ax, scores_df: pd.DataFrame, strategy_label: str, model_label: str) -> None:
    harm = scores_df[scores_df["split"] == "harmful"]["score"].to_numpy()
    benign = scores_df[scores_df["split"] == "benign"]["score"].to_numpy()

    # Sign correction (same convention as rest of paper)
    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.zeros(len(benign)), np.ones(len(harm))])
    s = np.concatenate([benign, harm])
    raw = roc_auc_score(y, s)
    if raw < 0.5:
        harm, benign = -harm, -benign

    # 1% FPR threshold
    thresh = np.percentile(benign, 99)

    # Histograms
    bins = np.linspace(min(harm.min(), benign.min()), max(harm.max(), benign.max()), 50)
    ax.hist(benign, bins=bins, alpha=0.55, color="#4477AA", label="benign", density=True)
    ax.hist(harm, bins=bins, alpha=0.55, color="#CC3311", label="harmful", density=True)
    ax.axvline(thresh, linestyle="--", color="black", linewidth=1.0, alpha=0.7)

    ax.set_xlabel("score")
    ax.set_ylabel("density")
    ax.set_title(f"{model_label}, {strategy_label}", fontsize=10, pad=6)
    ax.grid(True, linestyle="-", linewidth=0.3, alpha=0.35)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("results/artifacts"))
    parser.add_argument("--out-path", type=Path, default=Path("figures/fig_score_dist"))
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), dpi=150)

    for i, (model_id, model_label) in enumerate(MODELS):
        for j, (strategy, strategy_label) in enumerate(STRATEGIES):
            df = load_scores(args.artifacts_dir, model_id, strategy)
            plot_one(axes[i, j], df, strategy_label, model_label)

    # Single legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], linestyle="--", color="black", linewidth=1.0, alpha=0.7))
    labels.append("1% FPR threshold")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {out}.pdf and .png")


if __name__ == "__main__":
    main()
