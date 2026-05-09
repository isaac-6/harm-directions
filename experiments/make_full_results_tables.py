#!/usr/bin/env python3
"""
make_full_results_tables.py
----------------------------
Generate the paper's full-results tables with bootstrap 95% CIs, in the
long-format layout (rows = model-strategy pairs, columns = Layer, AUROC,
TPR) split into two parts across families.

Reads:
  results/artifacts/all_bootstrap_cis.csv
  results/artifacts/<slug>/score_distributions.csv  (for layer info)

Writes:
  results/table_full_a.tex  — Qwen2.5 + Qwen3.5
  results/table_full_b.tex  — Llama-3.2 + Gemma-3

Each table row: one (model, strategy) pair with layer, AUROC [CI], TPR [CI].
Models are grouped via \\multirow for the Family column.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

VARIANT_ORDER = ["base", "instruct", "abliterated"]

# Model ID → (family, variant)
MODEL_METADATA: dict[str, tuple[str, str]] = {
    "Qwen/Qwen2.5-0.5B": ("Qwen2.5", "base"),
    "Qwen/Qwen2.5-0.5B-Instruct": ("Qwen2.5", "instruct"),
    "huihui-ai/Qwen2.5-0.5B-Instruct-abliterated": ("Qwen2.5", "abliterated"),
    "Qwen/Qwen3.5-0.8B-Base": ("Qwen3.5", "base"),
    "Qwen/Qwen3.5-0.8B": ("Qwen3.5", "instruct"),
    "huihui-ai/Huihui-Qwen3.5-0.8B-abliterated": ("Qwen3.5", "abliterated"),
    "meta-llama/Llama-3.2-1B": ("Llama-3.2", "base"),
    "meta-llama/Llama-3.2-1B-Instruct": ("Llama-3.2", "instruct"),
    "huihui-ai/Llama-3.2-1B-Instruct-abliterated": ("Llama-3.2", "abliterated"),
    "google/gemma-3-1b-pt": ("Gemma-3", "base"),
    "google/gemma-3-1b-it": ("Gemma-3", "instruct"),
    "huihui-ai/gemma-3-1b-it-abliterated": ("Gemma-3", "abliterated"),
}

# Two-table split (two families per table, matching paper convention)
FAMILY_SPLITS = [
    ("a", ["Qwen2.5", "Qwen3.5"], "Qwen2.5 and Qwen3.5"),
    ("b", ["Llama-3.2", "Gemma-3"], "Llama-3.2 and Gemma-3"),
]

# Strategy display order within each model block
STRATEGY_ORDER = [
    "mean_diff",
    "soft_auc",
    "pc1_normative",
    "theta_normative",
    "theta_two_class",
    "random",
    "perplexity",
]
STRATEGY_LABELS = {
    "mean_diff": r"$\wmd$",
    "soft_auc": r"$\wopt$",
    "pc1_normative": r"$\wpca$",
    "theta_normative": r"$\theta$-norm",
    "theta_two_class": r"$\theta$ two-class",
    "random": "Random",
    "perplexity": "Perplexity",
}


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


# ---------------------------------------------------------------------------
# Layer lookup (pulled from the score_distributions.csv per model)
# ---------------------------------------------------------------------------


def load_layers(artifacts_dir: Path) -> dict[tuple[str, str], int]:
    """
    Return mapping (model_id, strategy) -> layer index.

    Layer 'perplexity' is -1 (not applicable), which we display as '---'.
    """
    out: dict[tuple[str, str], int] = {}
    for model_id in MODEL_METADATA:
        score_csv = artifacts_dir / model_slug(model_id) / "score_distributions.csv"
        if not score_csv.exists():
            continue
        df = pd.read_csv(score_csv)
        for strat in df["strategy"].unique():
            layers = df.loc[df["strategy"] == strat, "layer"].unique()
            if len(layers) == 0:
                continue
            out[(model_id, strat)] = int(layers[0])
    return out


def format_layer(layer: int | None) -> str:
    if layer is None or layer == -1:
        return "---"
    return str(layer)


# ---------------------------------------------------------------------------
# Cell formatters
# ---------------------------------------------------------------------------


def fmt_auroc(value: float, lo: float | None, hi: float | None, bold: bool) -> str:
    value_str = f"{value:.3f}"
    if bold:
        value_str = r"\textbf{" + value_str + "}"
    if lo is None or hi is None:
        return value_str
    return f"{value_str} [{lo:.3f}, {hi:.3f}]"


def fmt_tpr(value: float, lo: float | None, hi: float | None) -> str:
    value_str = f"{value:.3f}"
    if lo is None or hi is None:
        return value_str
    return f"{value_str} [{lo:.3f}, {hi:.3f}]"


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------


def build_table_part(
    df: pd.DataFrame,
    layer_map: dict[tuple[str, str], int],
    part: str,
    families: list[str],
    fam_desc: str,
) -> str:
    """Build one half of the split table."""
    lines: list[str] = []
    lines.append(r"\begin{table}[p]")
    lines.append(r"\centering\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(
        r"\caption{Full per-model results, Part~"
        + ("1" if part == "a" else "2")
        + f": {fam_desc}. "
        + r"All metrics evaluated against the full benign set "
        + r"(Alpaca\,+\,XSTest) at the validation-selected layer. "
        + r"Bold marks the highest AUROC median within each model. "
        + r"Stratified bootstrap 95\% CIs (1{,}000 resamples) in brackets.}"
    )
    lines.append(r"\label{tab:full_" + part + "}")
    lines.append(r"\begin{tabular}{llrll}")
    lines.append(r"\toprule")
    lines.append(
        r"Model & Strategy & Layer & "
        r"AUROC [95\% CI] & TPR@1\%FPR [95\% CI] \\"
    )
    lines.append(r"\midrule")

    for fam in families:
        for variant in VARIANT_ORDER:
            # Find the model ID for this family/variant
            model_id = next(
                (m for m, (f, v) in MODEL_METADATA.items() if f == fam and v == variant),
                None,
            )
            if model_id is None:
                continue

            sub = df[df["model"] == model_id]
            if sub.empty:
                continue

            # Which strategies are present, in order
            present_strats = [s for s in STRATEGY_ORDER if s in sub["strategy"].values]
            if not present_strats:
                continue

            # Find the best strategy by AUROC median (for bolding)
            max_auroc = sub["auroc_median"].max()

            first_row = True
            for strat in present_strats:
                row = sub[sub["strategy"] == strat]
                if row.empty:
                    continue
                r = row.iloc[0]

                auroc_med = float(r["auroc_median"])
                auroc_lo = float(r["auroc_lo"])
                auroc_hi = float(r["auroc_hi"])
                tpr_med = float(r["tpr_median"])
                tpr_lo = float(r["tpr_lo"])
                tpr_hi = float(r["tpr_hi"])
                layer = layer_map.get((model_id, strat))

                is_bold = auroc_med == max_auroc
                auroc_str = fmt_auroc(auroc_med, auroc_lo, auroc_hi, bold=is_bold)
                tpr_str = fmt_tpr(tpr_med, tpr_lo, tpr_hi)
                layer_str = format_layer(layer)

                if first_row:
                    model_cell = (
                        r"\multirow{"
                        + str(len(present_strats))
                        + r"}{*}{\shortstack[l]{"
                        + fam
                        + r"\\"
                        + f"({variant})"
                        + r"}}"
                    )
                    first_row = False
                else:
                    model_cell = ""

                lines.append(
                    f"  {model_cell} & {STRATEGY_LABELS[strat]} & "
                    f"{layer_str} & {auroc_str} & {tpr_str} \\\\"
                )

            lines.append(r"\midrule")

    # Replace trailing \midrule with \bottomrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


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
        "--artifacts-dir",
        type=Path,
        default=Path("results/artifacts"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.bootstrap_csv)
    layer_map = load_layers(args.artifacts_dir)

    for part, families, fam_desc in FAMILY_SPLITS:
        tex = build_table_part(df, layer_map, part, families, fam_desc)
        out = args.out_dir / f"table_full_{part}.tex"
        out.write_text(tex, encoding="utf-8")
        print(f"Wrote {out}")

    # Quick sanity check
    print("\nVerification:")
    print(f"  Models in bootstrap CSV: {df['model'].nunique()}")
    print(f"  Strategies: {sorted(df['strategy'].unique())}")
    print(f"  Layers captured: {len(layer_map)}")
    print("\n  Grand-mean AUROC / TPR per strategy:")
    grand = (
        df.groupby("strategy")
        .agg(
            auroc_mean=("auroc_median", "mean"),
            auroc_std=("auroc_median", "std"),
            tpr_mean=("tpr_median", "mean"),
            tpr_std=("tpr_median", "std"),
        )
        .reset_index()
    )
    for _, r in grand.iterrows():
        print(
            f"    {r['strategy']:<18s}  "
            f"AUROC {r['auroc_mean']:.3f} ± {r['auroc_std']:.3f}  "
            f"TPR {r['tpr_mean']:.3f} ± {r['tpr_std']:.3f}"
        )


if __name__ == "__main__":
    main()
