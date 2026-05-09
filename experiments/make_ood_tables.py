#!/usr/bin/env python3
"""
make_ood_tables.py
-------------------
Generate Appendix D tables: per-model, per-strategy effective AUROC across
all (harmful_source x benign_source) combinations, split into two tables by
family (Gemma-3 + Llama-3.2 in Part 1, Qwen2.5 + Qwen3.5 in Part 2).

Rows: (model, strategy) with \\multirow for each model.
Columns: 6 (harm, benign) pairs.

Reads:  results/artifacts/<slug>/score_distributions.csv
Writes:
  results/table_ood_a.tex  (Gemma-3, Llama-3.2)
  results/table_ood_b.tex  (Qwen2.5, Qwen3.5)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

VARIANT_ORDER = ["base", "instruct", "abliterated"]

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

# Two-table split matching the example
TABLE_SPLITS = [
    ("a", ["Gemma-3", "Llama-3.2"], "Gemma-3 and Llama-3.2"),
    ("b", ["Qwen2.5", "Qwen3.5"], "Qwen2.5 and Qwen3.5"),
]

# Strategies shown per model block, in display order
STRATEGY_ORDER = [
    ("mean_diff", "Mean diff (LDA)"),
    ("soft_auc", "Soft-AUC"),
    ("pc1_normative", "PC1 (normative)"),
    ("theta_normative", r"$\theta$ normative"),
    ("theta_two_class", r"$\theta$ two-class"),
]

# Column headers: (harm_source, benign_source, display_label)
SOURCE_PAIRS = [
    ("advbench", "alpaca", "AdvB/Alp"),
    ("advbench", "xstest", "AdvB/XS"),
    ("harmbench", "alpaca", "HB/Alp"),
    ("harmbench", "xstest", "HB/XS"),
    ("jailbreakbench", "alpaca", "JBB/Alp"),
    ("jailbreakbench", "xstest", "JBB/XS"),
]


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def load_scores(artifacts_dir: Path, model_id: str) -> pd.DataFrame | None:
    p = artifacts_dir / model_slug(model_id) / "score_distributions.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def compute_auroc_eff(harm: np.ndarray, benign: np.ndarray) -> float:
    if len(harm) == 0 or len(benign) == 0:
        return float("nan")
    y = np.concatenate([np.zeros(len(benign)), np.ones(len(harm))])
    s = np.concatenate([benign, harm])
    raw = roc_auc_score(y, s)
    return float(max(raw, 1 - raw))


def cell_auroc(df: pd.DataFrame, strategy: str, harm_src: str, benign_src: str) -> float:
    sub = df[df["strategy"] == strategy]
    harm = sub[(sub["split"] == "harmful") & (sub["source"] == harm_src)]["score"].to_numpy()
    benign = sub[(sub["split"] == "benign") & (sub["source"] == benign_src)]["score"].to_numpy()
    return compute_auroc_eff(harm, benign)


def format_cell(v: float) -> str:
    if np.isnan(v):
        return "---"
    return f"{v:.3f}"


def models_in_family_order(families: list[str]) -> list[str]:
    """Ordered list of model IDs for the given families, by (family, variant)."""
    out = []
    for fam in families:
        for variant in VARIANT_ORDER:
            for m, (f, v) in MODEL_METADATA.items():
                if f == fam and v == variant:
                    out.append(m)
                    break
    return out


def build_table(
    artifacts_dir: Path,
    part: str,
    families: list[str],
    fam_desc: str,
) -> str:
    models = models_in_family_order(families)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(
        r"\caption{Disaggregated OOD AUROC, Part~"
        + ("1" if part == "a" else "2")
        + f": {fam_desc}. "
        + r"Each cell: effective AUROC for the given harmful source vs benign "
        + r"source, at the validation-selected layer. "
        + r"AdvB: AdvBench, HB: HarmBench, JBB: JailbreakBench, Alp: Alpaca, "
        + r"XS: XSTest.}"
    )
    lines.append(r"\label{tab:ood_" + part + "}")
    lines.append(r"\begin{tabular}{llrrrrrr}")
    lines.append(r"\toprule")

    # Header: Model, Strategy, then 6 source-pair columns
    header_cells = ["Model", "Strategy"] + [label for (_, _, label) in SOURCE_PAIRS]
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    n_strats = len(STRATEGY_ORDER)

    for i, model_id in enumerate(models):
        df = load_scores(artifacts_dir, model_id)
        if df is None:
            continue

        family, variant = MODEL_METADATA[model_id]
        first = True
        for strat_key, strat_label in STRATEGY_ORDER:
            cells = []
            for h, b, _ in SOURCE_PAIRS:
                v = cell_auroc(df, strat_key, h, b)
                cells.append(format_cell(v))

            if first:
                model_cell = (
                    r"\multirow{"
                    + str(n_strats)
                    + r"}{*}{\shortstack[l]{"
                    + family
                    + r"\\"
                    + f"({variant})"
                    + r"}}"
                )
                first = False
            else:
                model_cell = ""

            lines.append(f"  {model_cell} & {strat_label} & " + " & ".join(cells) + r" \\")

        # Midrule between models, except after the last one
        if i < len(models) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("results/artifacts"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    for part, families, fam_desc in TABLE_SPLITS:
        tex = build_table(args.artifacts_dir, part, families, fam_desc)
        out = args.out_dir / f"table_ood_{part}.tex"
        out.write_text(tex, encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
