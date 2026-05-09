#!/usr/bin/env python3
"""
make_auroc_vs_tpr_figure.py
---------------------------
Two-panel scatter: AUROC vs TPR@1%FPR, with stratified bootstrap 95% CI whiskers.
Left panel: w_LDA; right panel: w_opt.

Each point: one model. Colour by family, shape by alignment variant.

Reads:  results/artifacts/all_bootstrap_cis.csv
Writes: figures/fig_auroc_vs_tpr.pdf, figures/fig_auroc_vs_tpr.png (Figure 1 in paper)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

FAMILY_COLOURS = {
    "Qwen2.5": "#4477AA",
    "Qwen3.5": "#228833",
    "Llama-3.2": "#CC3311",
    "Gemma-3": "#AA3377",
}

VARIANT_MARKERS = {
    "base": "o",
    "instruct": "s",
    "abliterated": "^",
}


def model_family_variant(model_id: str) -> tuple[str, str]:
    if "Qwen2.5" in model_id:
        family = "Qwen2.5"
    elif "Qwen3.5" in model_id or "Qwen3-" in model_id:
        family = "Qwen3.5"
    elif "Llama-3.2" in model_id:
        family = "Llama-3.2"
    elif "gemma-3" in model_id.lower():
        family = "Gemma-3"
    else:
        raise ValueError(f"Unknown family: {model_id}")

    lower = model_id.lower()
    if "abliterated" in lower:
        variant = "abliterated"
    elif "instruct" in lower or ("it" in lower and "gemma" in lower):
        variant = "instruct"
    elif (
        model_id.endswith("-pt")
        or lower.endswith("-base")
        or lower == "qwen/qwen2.5-0.5b"
        or lower == "meta-llama/llama-3.2-1b"
    ):
        variant = "base"
    else:
        variant = "base"  # fallback

    return family, variant


def plot_panel(ax, df: pd.DataFrame, strategy: str, title: str) -> None:
    sub = df[df["strategy"] == strategy]
    for _, r in sub.iterrows():
        family, variant = model_family_variant(r["model"])
        colour = FAMILY_COLOURS[family]
        marker = VARIANT_MARKERS[variant]

        x = r["auroc_median"]
        y = r["tpr_median"]
        xerr = [[x - r["auroc_lo"]], [r["auroc_hi"] - x]]
        yerr = [[y - r["tpr_lo"]], [r["tpr_hi"] - y]]
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt=marker,
            color=colour,
            ecolor=colour,
            capsize=2.5,
            elinewidth=1.0,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.6,
            alpha=0.9,
        )

    ax.set_xlabel("Effective AUROC")
    ax.set_ylabel("TPR @ 1% FPR")
    ax.set_xlim(0.93, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title, fontsize=10, pad=8)
    ax.grid(True, linestyle="-", linewidth=0.3, alpha=0.35)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def build_legend(fig, ax):
    """Compact legend: families on one line, variants on another."""
    handles = []
    labels = []
    for family, colour in FAMILY_COLOURS.items():
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="o",
                color=colour,
                linestyle="",
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )
        )
        labels.append(family)
    for variant, marker in VARIANT_MARKERS.items():
        handles.append(
            plt.Line2D(
                [],
                [],
                marker=marker,
                color="gray",
                linestyle="",
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )
        )
        labels.append(variant)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        handletextpad=0.3,
        columnspacing=1.2,
        bbox_to_anchor=(0.5, -0.02),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-csv", type=Path, default=Path("results/artifacts/all_bootstrap_cis.csv")
    )
    parser.add_argument("--out-path", type=Path, default=Path("figures/fig_auroc_vs_tpr"))
    args = parser.parse_args()

    df = pd.read_csv(args.bootstrap_csv)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150, sharey=True)
    plot_panel(axes[0], df, "mean_diff", r"$\mathbf{w}_{\mathrm{LDA}}$")
    plot_panel(axes[1], df, "soft_auc", r"$\mathbf{w}_{\mathrm{opt}}$")
    build_legend(fig, axes[0])

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = args.out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {out}.pdf and .png")


if __name__ == "__main__":
    main()
