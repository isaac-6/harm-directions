#!/usr/bin/env python3
"""
make_heatmaps.py
----------------
Generate publication-quality AUROC and TPR@1%FPR heatmaps from the
per-model results CSVs in results/.

Produces both PDF (for LaTeX) and PNG (for README preview).

Reads:  results/artifacts/all_bootstrap_cis.csv
Writes: figures/fig_auroc_heatmap.pdf, .png  (Figure 2 in paper)
        figures/fig_tpr_heatmap.pdf, .png    (Figure 3 in paper)

Usage
-----
    python scripts/make_heatmaps.py
    python scripts/make_heatmaps.py --out-dir figures
    python scripts/make_heatmaps.py --cmap Blues
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

# ---------------------------------------------------------------------------
# Model and strategy metadata
# ---------------------------------------------------------------------------

# Display order: by family, then variant (base, instruct, abliterated).
FAMILY_ORDER = ["Qwen2.5", "Qwen3.5", "Llama-3.2", "Gemma-3"]
VARIANT_ORDER = {"base": 0, "instruct": 1, "abliterated": 2}

# Model ID → (family, variant, short_label)
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

# Strategy display order and labels. Keys match the 'strategy' column in CSVs.
STRATEGY_ORDER = [
    "pc1_normative",
    "theta_normative",
    "mean_diff",
    "soft_auc",
    "theta_two_class",
    "random",
    "perplexity",
]
STRATEGY_LABELS = {
    "pc1_normative": r"$\mathbf{w}_{\mathrm{PC1}}$",
    "theta_normative": r"$\theta$-norm (zero-shot)",
    "mean_diff": r"$\mathbf{w}_{\mathrm{LDA}}$",
    "soft_auc": r"$\mathbf{w}_{\mathrm{opt}}$",
    "theta_two_class": r"$\theta$ two-class",
    "random": "Random",
    "perplexity": "Perplexity",
}


# ---------------------------------------------------------------------------
# Data loading and aggregation
# ---------------------------------------------------------------------------

# def load_all_results(results_dir: Path) -> pd.DataFrame:
#     """Load and concatenate all per-model _results.csv files."""
#     csvs = sorted(results_dir.glob("*_results.csv"))
#     if not csvs:
#         raise FileNotFoundError(
#             f"No _results.csv files found in {results_dir}"
#         )
#     frames = [pd.read_csv(p) for p in csvs]
#     return pd.concat(frames, ignore_index=True)


# def aggregate_per_model(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Collapse harmful x benign source cells to a single value per (model, strategy)
#     by taking the mean across evaluation-source pairs.
#     """
#     return (
#         df.groupby(["model", "strategy"], as_index=False)
#         .agg(
#             eff_auroc=("eff_auroc", "mean"),
#             tpr=("tpr_1pct_fpr", "mean"),
#         )
#     )


def load_bootstrap_medians(bootstrap_csv: Path) -> pd.DataFrame:
    """Load bootstrap CI file and rename columns to match downstream usage."""
    if not bootstrap_csv.exists():
        raise FileNotFoundError(
            f"Bootstrap CI file not found: {bootstrap_csv}. Run compute_bootstrap_ci.py first."
        )
    df = pd.read_csv(bootstrap_csv)
    return df.rename(
        columns={
            "auroc_median": "eff_auroc",
            "tpr_median": "tpr",
        }
    )


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------


def ordered_models(df: pd.DataFrame) -> list[str]:
    """Return model IDs in (family, variant) display order."""
    present = [m for m in df["model"].unique() if m in MODEL_METADATA]
    return sorted(
        present,
        key=lambda m: (
            FAMILY_ORDER.index(MODEL_METADATA[m][0]),
            VARIANT_ORDER[MODEL_METADATA[m][1]],
        ),
    )


def ordered_strategies(df: pd.DataFrame) -> list[str]:
    """Return strategies in display order, filtering to those present in data."""
    present = set(df["strategy"].unique())
    return [s for s in STRATEGY_ORDER if s in present]


def build_matrix(
    df: pd.DataFrame,
    metric: str,
    strategies: list[str],
    models: list[str],
) -> np.ndarray:
    """Return an (n_strategies, n_models) matrix of values."""
    mat = np.full((len(strategies), len(models)), np.nan)
    for si, strat in enumerate(strategies):
        for mi, model in enumerate(models):
            cell = df[(df["strategy"] == strat) & (df["model"] == model)]
            if not cell.empty:
                mat[si, mi] = cell[metric].iloc[0]
    return mat


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def format_model_labels(models: list[str]) -> list[str]:
    """Two-line labels: short name on top, variant underneath."""
    return [f"{MODEL_METADATA[m][2]}\n{MODEL_METADATA[m][1]}" for m in models]


def format_strategy_labels(strategies: list[str]) -> list[str]:
    return [STRATEGY_LABELS.get(s, s) for s in strategies]


def luminance(rgb: tuple[float, float, float]) -> float:
    """Relative luminance (WCAG), used to choose black or white text on a cell."""
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    cbar_label: str,
    out_path: Path,
    vmin: float,
    vmax: float,
    cmap: str,
    fmt: str = ".3f",
) -> None:
    """Render and save a single heatmap."""
    n_rows, n_cols = matrix.shape

    # Figure sizing: width scales with n_cols, height with n_rows. Tuned for
    # 12 models x 7 strategies ≈ (10, 4.5) inches.
    fig_w = max(6.0, 0.75 * n_cols + 2.0)
    fig_h = max(3.5, 0.5 * n_rows + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

    cmap_obj = plt.get_cmap(cmap)
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap_obj, norm=norm)

    # Family separator lines (Qwen2.5 | Qwen3.5 | Llama-3.2 | Gemma-3)
    # Each family has 3 variants so separators are at columns 3, 6, 9.
    for col in [3, 6, 9]:
        if col < n_cols:
            ax.axvline(col - 0.5, color="white", linewidth=1.5)

    # Axis ticks and labels
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(row_labels)
    ax.tick_params(axis="both", which="both", length=0)  # remove tick marks

    # Cell annotations
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            rgba = cmap_obj(norm(val))
            text_colour = "white" if luminance(rgba[:3]) < 0.5 else "black"
            ax.text(
                j,
                i,
                format(val, fmt),
                ha="center",
                va="center",
                color=text_colour,
                fontsize=9,
            )

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.015)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Family grouping labels above the heatmap (optional, small)
    # Positioned above the column labels, centered per 3-model group.
    family_positions = {"Qwen2.5": 1, "Qwen3.5": 4, "Llama-3.2": 7, "Gemma-3": 10}
    for family, pos in family_positions.items():
        if pos < n_cols:
            ax.text(
                pos,
                -0.9,
                family,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                transform=ax.transData,
            )

    ax.set_ylim(n_rows - 0.5, -1.2)  # make room for family labels above

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save both PDF (for LaTeX) and PNG (for GitHub preview)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path.with_suffix('.pdf')} and {out_path.with_suffix('.png')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-csv",
        type=Path,
        default=Path("results/artifacts/all_bootstrap_cis.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures"),
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="Blues",
        help="Sequential matplotlib colourmap. Try: viridis, cividis, mako, Blues.",
    )
    parser.add_argument(
        "--auroc-vmin",
        type=float,
        default=0.5,
        help="Lower bound for AUROC colour range (default: 0.5 = chance).",
    )
    args = parser.parse_args()

    # # Load and aggregate
    # raw = load_all_results(args.results_dir)
    # agg = aggregate_per_model(raw)

    # Load bootstrap medians (canonical numbers for the paper)
    agg = load_bootstrap_medians(args.bootstrap_csv)

    strategies = ordered_strategies(agg)
    models = ordered_models(agg)
    row_labels = format_strategy_labels(strategies)
    col_labels = format_model_labels(models)

    # AUROC heatmap
    auroc_mat = build_matrix(agg, "eff_auroc", strategies, models)
    plot_heatmap(
        auroc_mat,
        row_labels=row_labels,
        col_labels=col_labels,
        cbar_label="Effective AUROC",
        out_path=args.out_dir / "fig_auroc_heatmap",
        vmin=args.auroc_vmin,
        vmax=1.0,
        cmap=args.cmap,
    )

    # TPR heatmap
    tpr_mat = build_matrix(agg, "tpr", strategies, models)
    plot_heatmap(
        tpr_mat,
        row_labels=row_labels,
        col_labels=col_labels,
        cbar_label="TPR @ 1% FPR",
        out_path=args.out_dir / "fig_tpr_heatmap",
        vmin=0.0,
        vmax=1.0,
        cmap=args.cmap,
    )


if __name__ == "__main__":
    main()
