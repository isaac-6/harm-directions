#!/usr/bin/env python3
"""
make_layer_profile_figure.py
-----------------------------
Generate Figure 5: per-layer effective AUROC for all strategies and all
12 models, arranged as a 4x3 grid (families x alignment variants).

Reads:  results/artifacts/<slug>/layer_sweep.csv
Writes: figures/fig_layer_profiles.pdf, .png     (Figure 4 in paper)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

FAMILY_ORDER = ["Qwen2.5", "Qwen3.5", "Llama-3.2", "Gemma-3"]
VARIANT_ORDER = ["base", "instruct", "abliterated"]

MODEL_METADATA: dict[str, tuple[str, str, str]] = {
    "Qwen/Qwen2.5-0.5B": ("Qwen2.5", "base", "Qwen2.5-0.5B"),
    "Qwen/Qwen2.5-0.5B-Instruct": ("Qwen2.5", "instruct", "Qwen2.5-0.5B"),
    "huihui-ai/Qwen2.5-0.5B-Instruct-abliterated": ("Qwen2.5", "abliterated", "Qwen2.5-0.5B"),
    "Qwen/Qwen3.5-0.8B-Base": ("Qwen3.5", "base", "Qwen3.5-0.8B"),
    "Qwen/Qwen3.5-0.8B": ("Qwen3.5", "instruct", "Qwen3.5-0.8B"),
    "huihui-ai/Huihui-Qwen3.5-0.8B-abliterated": ("Qwen3.5", "abliterated", "Qwen3.5-0.8B"),
    "meta-llama/Llama-3.2-1B": ("Llama-3.2", "base", "Llama-3.2-1B"),
    "meta-llama/Llama-3.2-1B-Instruct": ("Llama-3.2", "instruct", "Llama-3.2-1B"),
    "huihui-ai/Llama-3.2-1B-Instruct-abliterated": ("Llama-3.2", "abliterated", "Llama-3.2-1B"),
    "google/gemma-3-1b-pt": ("Gemma-3", "base", "Gemma-3-1B"),
    "google/gemma-3-1b-it": ("Gemma-3", "instruct", "Gemma-3-1B"),
    "huihui-ai/gemma-3-1b-it-abliterated": ("Gemma-3", "abliterated", "Gemma-3-1B"),
}

# Strategy styling: Paul Tol muted palette for categorical lines, colour-blind safe.
STRATEGY_STYLE = {
    "mean_diff": {
        "label": r"$\mathbf{w}_{\mathrm{LDA}}$",
        "colour": "#332288",  # dark blue
        "linestyle": "-",
        "linewidth": 1.6,
        "zorder": 5,
    },
    "soft_auc": {
        "label": r"$\mathbf{w}_{\mathrm{opt}}$",
        "colour": "#117733",  # dark green
        "linestyle": "-",
        "linewidth": 1.6,
        "zorder": 5,
    },
    "theta_two_class": {
        "label": r"$\theta$ two-class",
        "colour": "#CC6677",  # rose
        "linestyle": "-",
        "linewidth": 1.4,
        "zorder": 4,
    },
    "pc1_normative": {
        "label": r"$\mathbf{w}_{\mathrm{PC1}}$",
        "colour": "#88CCEE",  # light cyan
        "linestyle": "--",
        "linewidth": 1.2,
        "zorder": 3,
    },
    "theta_normative": {
        "label": r"$\theta$-norm",
        "colour": "#DDCC77",  # sand
        "linestyle": "--",
        "linewidth": 1.2,
        "zorder": 3,
    },
    "random": {
        "label": "Random",
        "colour": "#999999",  # gray
        "linestyle": ":",
        "linewidth": 1.0,
        "zorder": 2,
    },
}

STRATEGY_ORDER = list(STRATEGY_STYLE.keys())  # legend order


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def load_layer_sweep(artifacts_dir: Path, model_id: str) -> pd.DataFrame | None:
    p = artifacts_dir / model_slug(model_id) / "layer_sweep.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def build_grid(artifacts_dir: Path) -> dict[tuple[str, str], pd.DataFrame]:
    """(family, variant) -> layer_sweep DataFrame."""
    grid: dict[tuple[str, str], pd.DataFrame] = {}
    for model_id, (family, variant, _) in MODEL_METADATA.items():
        df = load_layer_sweep(artifacts_dir, model_id)
        if df is not None:
            grid[(family, variant)] = df
    return grid


def plot(artifacts_dir: Path, out_path: Path) -> None:
    grid = build_grid(artifacts_dir)

    n_families = len(FAMILY_ORDER)
    n_variants = len(VARIANT_ORDER)

    fig, axes = plt.subplots(
        n_families,
        n_variants,
        figsize=(11, 10),
        dpi=150,
        sharex=False,  # layer counts differ by family
        sharey=True,
    )

    for i, family in enumerate(FAMILY_ORDER):
        for j, variant in enumerate(VARIANT_ORDER):
            ax = axes[i, j]
            key = (family, variant)
            if key not in grid:
                ax.text(
                    0.5,
                    0.5,
                    "(no data)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="0.5",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            df = grid[key]
            for strategy in STRATEGY_ORDER:
                sub = df[df["strategy"] == strategy].sort_values("layer")
                if sub.empty:
                    continue
                style = STRATEGY_STYLE[strategy]
                ax.plot(
                    sub["layer"],
                    sub["auroc_eff"],
                    color=style["colour"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    zorder=style["zorder"],
                    label=style["label"] if (i == 0 and j == 0) else None,
                )

            # Per-panel styling
            ax.set_ylim(0.45, 1.02)
            ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax.axhline(0.5, color="0.85", linewidth=0.5, zorder=1)
            ax.grid(True, linestyle="-", linewidth=0.3, alpha=0.3, zorder=0)
            ax.set_axisbelow(True)

            for side in ["top", "right"]:
                ax.spines[side].set_visible(False)

            # Column titles: only on top row
            if i == 0:
                ax.set_title(variant, fontsize=11, pad=6)

            # Row labels: only on leftmost column
            if j == 0:
                ax.set_ylabel(family, fontsize=11, rotation=90, labelpad=8)

            # X label: only on bottom row
            if i == n_families - 1:
                ax.set_xlabel("Layer", fontsize=9)

            ax.tick_params(axis="both", labelsize=8)

    # Single legend at top centre, above all panels
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(STRATEGY_ORDER),
        frameon=True,
        framealpha=1.0,
        edgecolor="0.7",
        fontsize=9,
        bbox_to_anchor=(0.5, 1.01),
        handletextpad=0.5,
        columnspacing=1.4,
    )

    # Shared y-axis label
    fig.supylabel("Effective AUROC", fontsize=10, x=0.01)

    fig.tight_layout(rect=[0.02, 0, 1, 0.97])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}.pdf and .png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("results/artifacts"))
    parser.add_argument("--out-path", type=Path, default=Path("figures/fig_layer_profiles"))
    args = parser.parse_args()

    plot(args.artifacts_dir, args.out_path)


if __name__ == "__main__":
    main()
