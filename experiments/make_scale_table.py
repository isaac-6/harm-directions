#!/usr/bin/env python3
"""
Generate the Qwen3.5 scaling transfer table (LaTeX).

Sibling of make_transfer_table.py. Same input CSVs, same column structure,
restricted to Qwen3.5 at multiple sizes. Used to show that the cross-variant
transfer pattern holds across an 11x parameter range within one family.

Reads:
  results/cross_variant_transfer.csv  -- per-(family, source, target) AUROC/TPR
  results/cross_variant_angles.csv    -- pairwise direction angles per family

Writes:
  results/transfer_table_scale.tex    -- LaTeX table, ready to \\input

Expects families named: "Qwen3.5", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B".
"Qwen3.5" (without a size suffix) is treated as the 0.8B baseline used in
the main experiments; the Size column will render it as "0.8B".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Ordered by size. First entry is the main-experiment 0.8B family.
# The tuples are (family_key_in_csv, size_label_for_table).
SCALE_ORDER: list[tuple[str, str]] = [
    ("Qwen3.5", "0.8B"),
    ("Qwen3.5-2B", "2B"),
    ("Qwen3.5-4B", "4B"),
    ("Qwen3.5-9B", "9B"),
]

SOURCE = "base"
TARGETS = ["instruct", "abliterated"]


def load_inputs(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    transfer_path = results_dir / "cross_variant_transfer.csv"
    angles_path = results_dir / "cross_variant_angles.csv"

    transfer = pd.read_csv(transfer_path, sep=None, engine="python")
    angles = pd.read_csv(angles_path, sep=None, engine="python")

    required_transfer_cols = {
        "family",
        "source_direction",
        "target_data",
        "eff_auroc",
        "tpr_corrected",
    }
    required_angle_cols = {"family", "direction_a", "direction_b", "angle_deg"}

    missing_t = required_transfer_cols - set(transfer.columns)
    missing_a = required_angle_cols - set(angles.columns)
    if missing_t:
        raise ValueError(f"Missing columns in transfer CSV: {sorted(missing_t)}")
    if missing_a:
        raise ValueError(f"Missing columns in angles CSV: {sorted(missing_a)}")

    return transfer, angles


def lookup_metric(
    transfer: pd.DataFrame,
    family: str,
    source: str,
    target: str,
    metric: str,
) -> float:
    mask = (
        (transfer["family"] == family)
        & (transfer["source_direction"] == source)
        & (transfer["target_data"] == target)
    )
    rows = transfer.loc[mask, metric]
    if len(rows) == 0:
        raise KeyError(f"No transfer row for family={family}, source={source}, target={target}")
    if len(rows) > 1:
        raise ValueError(
            f"Multiple transfer rows for family={family}, "
            f"source={source}, target={target}; expected exactly one"
        )
    return float(rows.iloc[0])


def lookup_angle(angles: pd.DataFrame, family: str, a: str, b: str) -> float:
    mask = (angles["family"] == family) & (
        ((angles["direction_a"] == a) & (angles["direction_b"] == b))
        | ((angles["direction_a"] == b) & (angles["direction_b"] == a))
    )
    rows = angles.loc[mask, "angle_deg"]
    if len(rows) == 0:
        raise KeyError(f"No angle row for family={family}, ({a}, {b})")
    return float(rows.iloc[0])


def fmt_delta_inline(x: float, decimals: int = 3) -> str:
    rounded = round(x, decimals)
    if rounded == 0:
        return f"$\\pm${abs(rounded):.{decimals}f}"
    sign = "$+$" if rounded > 0 else "$-$"
    return f"{sign}{abs(rounded):.{decimals}f}"


def fmt_value_with_delta(value: float, delta: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f} ({fmt_delta_inline(delta, decimals)})"


def fmt_angle(deg: float) -> str:
    return f"${round(deg)}^\\circ$"


def build_row(
    transfer: pd.DataFrame,
    angles: pd.DataFrame,
    family_key: str,
    size_label: str,
) -> str:
    own_auroc = {
        v: lookup_metric(transfer, family_key, v, v, "eff_auroc") for v in [SOURCE, *TARGETS]
    }
    own_tpr = {
        v: lookup_metric(transfer, family_key, v, v, "tpr_corrected") for v in [SOURCE, *TARGETS]
    }

    a_bi = lookup_angle(angles, family_key, "base", "instruct")
    a_ba = lookup_angle(angles, family_key, "base", "abliterated")
    a_ia = lookup_angle(angles, family_key, "instruct", "abliterated")

    transfer_cells: list[str] = []
    for target in TARGETS:
        t_auroc = lookup_metric(transfer, family_key, SOURCE, target, "eff_auroc")
        t_tpr = lookup_metric(transfer, family_key, SOURCE, target, "tpr_corrected")
        d_auroc = t_auroc - own_auroc[target]
        d_tpr = t_tpr - own_tpr[target]
        transfer_cells.append(fmt_value_with_delta(t_auroc, d_auroc))
        transfer_cells.append(fmt_value_with_delta(t_tpr, d_tpr))

    return (
        " & ".join(
            [
                size_label,
                fmt_angle(a_bi),
                fmt_angle(a_ba),
                fmt_angle(a_ia),
                *transfer_cells,
            ]
        )
        + r" \\"
    )


def build_table(transfer: pd.DataFrame, angles: pd.DataFrame) -> str:
    body = "\n".join(build_row(transfer, angles, key, label) for key, label in SCALE_ORDER)

    return (
        r"""\begin{table}[htb]
\centering\small
\caption{Cross-variant direction transfer across Qwen3.5 model sizes.
  Same column structure as Table~\ref{tab:transfer}, restricted to the
  Qwen3.5 family across an 11$\times$ parameter range (0.8B--9B).
  The 0.8B row reproduces the Qwen3.5 entry from Table~\ref{tab:transfer}
  and is included here for direct size-wise comparison.
  AUROC, TPR@1\%FPR and direction-angle conventions are as in
  Table~\ref{tab:transfer}.}
\label{tab:transfer_scale}
\begin{tabular}{lcccccccc}
\toprule
 & \multicolumn{3}{c}{Angles}
   & \multicolumn{2}{c}{Instruct}
   & \multicolumn{2}{c}{Abliterated} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}
Size & B$\leftrightarrow$I & B$\leftrightarrow$A & I$\leftrightarrow$A
       & AUROC & TPR@1\%FPR
       & AUROC & TPR@1\%FPR \\
\midrule
"""
        + body
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
        help="Directory containing the input CSVs and receiving the output .tex",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="transfer_table_scale.tex",
        help="Filename for the generated LaTeX table",
    )
    args = parser.parse_args()

    transfer, angles = load_inputs(args.results_dir)
    table_tex = build_table(transfer, angles)

    out_path = args.results_dir / args.output_name
    out_path.write_text(table_tex, encoding="utf-8")

    print(f"Wrote {out_path}\n")
    print("Row summary for verification:")
    for family_key, size_label in SCALE_ORDER:
        a_bi = lookup_angle(angles, family_key, "base", "instruct")
        a_ba = lookup_angle(angles, family_key, "base", "abliterated")
        a_ia = lookup_angle(angles, family_key, "instruct", "abliterated")
        print(
            f"\n  {size_label} ({family_key})  angles  "
            f"B-I: {a_bi:.1f}deg, B-A: {a_ba:.1f}deg, I-A: {a_ia:.1f}deg"
        )
        for variant in [SOURCE, *TARGETS]:
            oa = lookup_metric(transfer, family_key, variant, variant, "eff_auroc")
            ot = lookup_metric(transfer, family_key, variant, variant, "tpr_corrected")
            print(f"    own  {variant:>12}: AUROC {oa:.3f}  TPR {ot:.3f}")
        for target in TARGETS:
            ta = lookup_metric(transfer, family_key, SOURCE, target, "eff_auroc")
            tt = lookup_metric(transfer, family_key, SOURCE, target, "tpr_corrected")
            print(f"    base -> {target:>10}: AUROC {ta:.3f}  TPR {tt:.3f}")


if __name__ == "__main__":
    main()
