#!/usr/bin/env python3
"""
Generate the full cross-variant transfer matrix table (LaTeX).

This is the detailed 3x3 transfer matrix table (paper Table 10).
Each family gets a 3-row block showing how each variant's direction
transfers to each variant's eval data. A separate scaling block holds
the Qwen3.5 larger-size rows.

Reads:
  results/cross_variant_transfer.csv

Writes:
  results/transfer_table_full.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Main experiment families and their display labels.
MAIN_FAMILIES: list[tuple[str, str]] = [
    ("Llama-3.2", "Llama-3.2"),
    ("Qwen2.5", "Qwen2.5"),
    ("Qwen3.5", "Qwen3.5$^\\dagger$"),  # dagger for the layer footnote
    ("Gemma-3", "Gemma-3"),
]

# Scaling extension families. Presented separately below the main block.
SCALE_FAMILIES: list[tuple[str, str]] = [
    ("Qwen3.5-2B", "Qwen3.5-2B"),
    ("Qwen3.5-4B", "Qwen3.5-4B"),
    ("Qwen3.5-9B", "Qwen3.5-9B"),
]

VARIANTS = ["base", "instruct", "abliterated"]
VARIANT_LABELS = {"base": "base", "instruct": "instruct", "abliterated": "abliterated"}

TPR_ITALIC_THRESHOLD = 0.25


def load_transfer(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "cross_variant_transfer.csv"
    df = pd.read_csv(path, sep=None, engine="python")
    required = {
        "family",
        "source_direction",
        "target_data",
        "eff_auroc",
        "tpr_corrected",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in transfer CSV: {sorted(missing)}")
    return df


def lookup(df: pd.DataFrame, family: str, src: str, tgt: str, metric: str) -> float:
    mask = (df["family"] == family) & (df["source_direction"] == src) & (df["target_data"] == tgt)
    rows = df.loc[mask, metric]
    if len(rows) == 0:
        raise KeyError(f"Missing row family={family}, source={src}, target={tgt}")
    return float(rows.iloc[0])


def fmt_cell(
    auc: float,
    tpr: float,
    bold: bool,
    italic_tpr_threshold: float = TPR_ITALIC_THRESHOLD,
) -> tuple[str, str]:
    """Return (auc_cell, tpr_cell) formatted strings."""
    auc_str = f"{auc:.3f}"
    tpr_str = f"{tpr:.3f}"
    if bold:
        auc_str = f"\\textbf{{{auc_str}}}"
        tpr_str = f"\\textbf{{{tpr_str}}}"
    elif tpr < italic_tpr_threshold:
        tpr_str = f"\\textit{{{tpr_str}}}"
    return auc_str, tpr_str


def build_family_block(
    df: pd.DataFrame,
    family_key: str,
    display_name: str,
) -> str:
    """Build a 3-row \\multirow block for one family."""
    rows = []
    for i, src in enumerate(VARIANTS):
        cells: list[str] = []
        # First cell: multirow family label on first row, empty on others
        first = f"\\multirow{{3}}{{*}}{{{display_name}}}" if i == 0 else ""
        # Second cell: source variant label
        source_label = VARIANT_LABELS[src]
        # Then the 6 data cells (3 targets x AUC+TPR)
        for tgt in VARIANTS:
            auc = lookup(df, family_key, src, tgt, "eff_auroc")
            tpr = lookup(df, family_key, src, tgt, "tpr_corrected")
            is_diagonal = src == tgt
            auc_str, tpr_str = fmt_cell(auc, tpr, bold=is_diagonal)
            cells.extend([auc_str, tpr_str])
        line_parts = [first, source_label, *cells]
        rows.append(" & ".join(line_parts) + r" \\")
    return "\n  ".join(rows)


def build_table(df: pd.DataFrame) -> str:
    main_blocks = [build_family_block(df, key, label) for key, label in MAIN_FAMILIES]
    scale_blocks = [build_family_block(df, key, label) for key, label in SCALE_FAMILIES]

    # Join with \midrule separators between blocks
    main_section = "\n\\midrule\n  ".join(main_blocks)
    scale_section = "\n\\midrule\n  ".join(scale_blocks)

    # Scaling extension visual separator
    scale_separator = (
        r"\midrule" + "\n"
        r"\multicolumn{8}{l}{\emph{Qwen3.5 scaling extension (same pipeline, larger sizes)}} \\"
        + "\n"
        r"\midrule" + "\n  "
    )

    return (
        r"""\begin{table}[htb]
\centering\small
\caption{Full cross-variant transfer matrix: AUROC and TPR@1\%FPR.
  Each cell shows AUROC\,/\,TPR when scoring the target variant's
  evaluation data with the source variant's $\wmd$ direction,
  fitted at the base model's validation-selected layer.
  Diagonal entries (bold) use each variant's own direction.
  Italicised TPR values ($<0.25$) indicate near-zero operational
  detectability despite adequate AUROC\@.
  The lower block extends the analysis to Qwen3.5 at 2B, 4B, and 9B
  parameters; see Table~\ref{tab:transfer_scale} for the summary.
  $^\dagger$For Qwen3.5 (0.8B), layer\,10 (base-selected) outperforms the
  instruct and abliterated variants' own optimal layer\,(22);
  diagonal AUROC for those variants therefore exceeds
  Table~\ref{tab:full_results_2}.}
\label{tab:transfer_full}
\begin{tabular}{llcccccc}
\toprule
 & & \multicolumn{2}{c}{Base} & \multicolumn{2}{c}{Instruct} & \multicolumn{2}{c}{Ablit.} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
 & Source & AUC & TPR & AUC & TPR & AUC & TPR \\
\midrule
  """
        + main_section
        + "\n"
        + scale_separator
        + scale_section
        + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="transfer_table_full.tex",
    )
    args = parser.parse_args()

    df = load_transfer(args.results_dir)
    table_tex = build_table(df)

    out_path = args.results_dir / args.output_name
    out_path.write_text(table_tex, encoding="utf-8")

    print(f"Wrote {out_path}\n")
    print("Row summary for verification:")
    all_families = [f for f, _ in MAIN_FAMILIES] + [f for f, _ in SCALE_FAMILIES]
    for family in all_families:
        print(f"\n  {family}")
        for src in VARIANTS:
            row = []
            for tgt in VARIANTS:
                auc = lookup(df, family, src, tgt, "eff_auroc")
                tpr = lookup(df, family, src, tgt, "tpr_corrected")
                marker = " *" if src == tgt else "  "
                row.append(f"{auc:.3f}/{tpr:.3f}{marker}")
            print(f"    {src:>12}:  " + "  ".join(row))


if __name__ == "__main__":
    main()
