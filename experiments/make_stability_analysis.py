"""
make_stability_analysis.py
---------------------
Measures how detection performance and reference direction stability change
as a function of normative set size N.

Primary metric  — AUROC(N)
    Fit the normative reference on the first N prompts; score a fixed held-out
    set of harmful and benign-aggressive prompts; compute AUROC.
    This directly answers: "how many normative prompts do I need before
    detection performance plateaus?" — the operationally meaningful question.

Secondary metric — PC1 angle drift(N)
    Angular distance between the PC1 computed on N prompts and the PC1 on
    the full set.  Useful as a geometric sanity check but not sufficient on
    its own: a direction that has drifted 20° may still yield identical AUROC
    if the manifold has a broad safe basin.

Both metrics are computed on a forward ordering (first N prompts, growing N).

Outputs (under results/eval/):
    stability_auroc.png        AUROC vs N — primary figure (paper-ready)
    stability_pc1_angle.png    PC1 angle drift vs N — appendix figure
    stability_summary.csv      all numerical values

Usage
-----
    python experiments/make_stability_analysis.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --normative-file   data/raw/splits/eval_benign_alpaca.txt \\
        --harmful-file     data/raw/splits/eval_harmful_advbench.txt \\
        --benign-agg-file  data/raw/splits/eval_benign_xstest.txt \\
        --normative-n  500 \\
        --harmful-n    200 \\
        --benign-agg-n 200 \\
        --layers 0 6 12 19 22 \\
        --target-layer 19 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from harm_directions import extract_all_layers
from harm_directions.directions import score_angular, theta_normative

RESULTS_DIR = Path("results/eval")  # overridden at runtime by --output-dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_prompts(path: str, n: int, seed: int) -> list[str]:
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    random.seed(seed)
    return random.sample(lines, min(n, len(lines)))


def compute_auroc(scores_neg: np.ndarray, scores_pos: np.ndarray) -> float:
    """AUROC where scores_pos should be higher for true positives."""
    y_true = np.concatenate([np.zeros(len(scores_neg)), np.ones(len(scores_pos))])
    y_score = np.concatenate([scores_neg, scores_pos])
    if np.isnan(y_score).any() or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _normalise_pc1(v: np.ndarray) -> np.ndarray:
    """Fix PCA sign ambiguity: make first non-near-zero coefficient positive."""
    v = v.copy()
    nz = np.where(np.abs(v) > 1e-9)[0]
    if len(nz) > 0 and v[nz[0]] < 0:
        v = -v
    return v


def pc1_angle_deg(X_subset: np.ndarray, pc1_full: np.ndarray) -> float:
    """
    Angle in degrees between PC1 of X_subset and a pre-computed reference PC1.
    Both vectors are sign-normalised before computing the angle.
    """
    pca = PCA(n_components=1)
    pca.fit(X_subset)
    v = _normalise_pc1(pca.components_[0])
    cos = float(np.clip(np.dot(v, pc1_full), -1.0, 1.0))
    return float(np.degrees(np.arccos(np.abs(cos))))


def auroc_at_layer(
    norm_acts_fit: np.ndarray,  # (n_fit, L, D)  — used to fit the direction
    norm_acts_eval: np.ndarray,  # (n_eval, L, D) — held-out normative, for scoring only
    harm_acts: np.ndarray,  # (N_h, L, D)
    benign_acts: np.ndarray,  # (N_b, L, D)
    layer: int,
) -> tuple[float, float]:
    """
    Fit theta_normative on norm_acts_fit at the given layer; score held-out sets.
    Separating fit and eval avoids the in-sample artefact at small N.

    Returns (AUROC_harmful_vs_norm, AUROC_harmful_vs_benign).
    """
    w = theta_normative(norm_acts_fit[:, layer, :])
    scores_norm = score_angular(norm_acts_eval[:, layer, :], w)
    scores_harm = score_angular(harm_acts[:, layer, :], w)
    scores_benign = score_angular(benign_acts[:, layer, :], w)
    return compute_auroc(scores_norm, scores_harm), compute_auroc(scores_benign, scores_harm)


def auroc_at_layer_harmful_ref(
    harm_acts_fit: np.ndarray,
    harm_acts_eval: np.ndarray,
    norm_acts_eval: np.ndarray,
    benign_acts: np.ndarray,
    layer: int,
) -> tuple[float, float]:
    """
    Harmful-reference stability: fit theta_normative on harm_acts_fit, score held-out sets.
    Sign-corrected so that harmful scores > normative on average.
    Returns (AUROC_harmful_vs_norm, AUROC_harmful_vs_benign).
    """
    w = theta_normative(harm_acts_fit[:, layer, :])
    s_harm = score_angular(harm_acts_eval[:, layer, :], w)
    s_norm = score_angular(norm_acts_eval[:, layer, :], w)
    s_benign = score_angular(benign_acts[:, layer, :], w)
    # Harmful prompts lie close to the harmful centroid → small angle → lower score.
    # Flip so that harmful > normative (higher score = more harmful).
    if s_harm.mean() < s_norm.mean():
        s_harm, s_norm, s_benign = -s_harm, -s_norm, -s_benign
    return compute_auroc(s_norm, s_harm), compute_auroc(s_benign, s_harm)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    global RESULTS_DIR
    RESULTS_DIR = Path(args.output_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    # ---- Load prompts ----
    norm_prompts = load_prompts(args.normative_file, args.normative_n, args.seed)
    harm_prompts = load_prompts(args.harmful_file, args.harmful_n, args.seed)
    benign_prompts = load_prompts(args.benign_agg_file, args.benign_agg_n, args.seed)

    print(
        f"Normative: {len(norm_prompts)} | "
        f"Harmful: {len(harm_prompts)} | "
        f"Benign-Agg: {len(benign_prompts)}"
    )
    print(f"Loading model: {args.model}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(  # type: ignore[arg-type]
            args.model, dtype="auto", trust_remote_code=True
        )
        .to(device)
        .eval()
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Extracting normative activations...")
    norm_acts = extract_all_layers(model, tokenizer, norm_prompts, pooling="last")
    print("Extracting harmful activations...")
    harm_acts = extract_all_layers(model, tokenizer, harm_prompts, pooling="last")
    print("Extracting benign-aggressive activations...")
    benign_acts = extract_all_layers(model, tokenizer, benign_prompts, pooling="last")

    del model
    torch.cuda.empty_cache()

    L = norm_acts.shape[1]

    N_norm = norm_acts.shape[0]

    # Reserve a fixed held-out normative evaluation set (20% of total).
    # The fit set grows from min_n up to N_fit_max.
    # Crucially, the eval set is NEVER used for fitting — this eliminates
    # the in-sample memorisation artifact at small N.
    n_eval = max(20, int(0.20 * N_norm))
    n_fit_max = N_norm - n_eval
    norm_acts_eval = norm_acts[n_fit_max:]  # last n_eval prompts — fixed
    norm_acts_fit_pool = norm_acts[:n_fit_max]  # pool from which subsets are drawn

    print(f"Normative pool: {n_fit_max} fit / {n_eval} eval (held-out, fixed)")

    layers = args.layers if args.layers else list(range(L))

    # ---- Sample sizes: log-spaced over the fit pool, dense at small N ----
    min_n = max(10, int(0.02 * n_fit_max))
    sizes_raw = np.unique(np.round(np.geomspace(min_n, n_fit_max, num=30)).astype(int))
    sizes: list[int] = [int(s) for s in sizes_raw if 2 <= s <= n_fit_max]
    if sizes[-1] != n_fit_max:
        sizes.append(n_fit_max)

    # ---- Pre-compute full-set PC1 from the entire fit pool ----
    full_pc1: dict[int, np.ndarray] = {}
    for layer in layers:
        pca = PCA(n_components=1)
        pca.fit(norm_acts_fit_pool[:, layer, :])
        full_pc1[layer] = _normalise_pc1(pca.components_[0])

    # ---- Forward ordering ----
    all_rows: list[dict] = []

    print("\nComputing stability...")
    for layer in layers:
        print(f"  Layer {layer} ...", end="", flush=True)
        for n in sizes:
            subset = norm_acts_fit_pool[:n]
            X_sub = subset[:, layer, :]

            auroc_h, auroc_b = auroc_at_layer(subset, norm_acts_eval, harm_acts, benign_acts, layer)
            angle = pc1_angle_deg(X_sub, full_pc1[layer]) if n >= 3 else float("nan")

            all_rows.append(
                {
                    "layer": layer,
                    "n": n,
                    "auroc_harmful": auroc_h,
                    "auroc_harmful_vs_benign": auroc_b,
                    "pc1_angle_deg": angle,
                }
            )
        print(" done")

    # ---- Harmful-ref stability ----
    N_harm = harm_acts.shape[0]
    n_harm_eval = max(20, int(0.20 * N_harm))
    n_harm_fit_max = N_harm - n_harm_eval
    harm_acts_eval_stab = harm_acts[n_harm_fit_max:]
    harm_acts_fit_pool = harm_acts[:n_harm_fit_max]
    print(f"\nHarmful pool: {n_harm_fit_max} fit / {n_harm_eval} eval (held-out)")

    harm_sizes_raw = np.unique(
        np.round(np.geomspace(max(10, int(0.02 * n_harm_fit_max)), n_harm_fit_max, num=20)).astype(
            int
        )
    )
    harm_sizes: list[int] = [int(s) for s in harm_sizes_raw if 2 <= s <= n_harm_fit_max]
    if harm_sizes[-1] != n_harm_fit_max:
        harm_sizes.append(n_harm_fit_max)

    harm_rows: list[dict] = []
    print("\nHarmful-ref stability...")
    for layer in layers:
        print(f"  Layer {layer} ...", end="", flush=True)
        for n in harm_sizes:
            auroc_h, auroc_b = auroc_at_layer_harmful_ref(
                harm_acts_fit_pool[:n], harm_acts_eval_stab, norm_acts_eval, benign_acts, layer
            )
            harm_rows.append(
                {
                    "layer": layer,
                    "n": n,
                    "auroc_harmful": auroc_h,
                    "auroc_harmful_vs_benign": auroc_b,
                }
            )
        print(" done")

    df_harm = pd.DataFrame(harm_rows)
    out_csv_harm = RESULTS_DIR / "stability_harmful_ref_summary.csv"
    df_harm.to_csv(out_csv_harm, index=False)
    print(f"\nSaved → {out_csv_harm}")
    _plot_auroc_stability(
        df_harm,
        layers,
        args.target_layer,
        title_suffix=" (harmful-ref)",
        out_stem="stability_auroc_harmful_ref",
    )

    # ---- Normative-ref ----
    df = pd.DataFrame(all_rows)
    out_csv = RESULTS_DIR / "stability_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    _plot_auroc_stability(df, layers, args.target_layer)
    _plot_pc1_angle(df, layers)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_auroc_stability(
    df: pd.DataFrame,
    layers: list[int],
    target_layer: int,
    title_suffix: str = "",
    out_stem: str = "stability_auroc",
) -> None:
    """
    Primary stability figure (two panels).
    title_suffix: appended to panel titles (e.g. " (harmful-ref strategy)").
    out_stem: filename stem under RESULTS_DIR (PDF and PNG are both written).
    """
    cmap = plt.get_cmap("viridis")
    layer_colors = {layer: cmap(i / max(len(layers) - 1, 1)) for i, layer in enumerate(layers)}

    _fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1 — all layers, harmful-vs-norm
    ax = axes[0]
    for layer in layers:
        sub = df[df["layer"] == layer].sort_values("n")
        ax.plot(
            sub["n"],
            sub["auroc_harmful"],
            marker="o",
            markersize=3,
            lw=1.8,
            color=layer_colors[layer],
            label=f"Layer {layer}",
        )

    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.5, label="AUROC=0.90")
    ax.set_xscale("log")
    ax.set_xlabel("Normative set size (N)")
    ax.set_ylabel("AUROC — harmful vs normative")
    ax.set_title(f"Harmful detection AUROC vs N{title_suffix}\n(all layers)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.3)

    # Panel 2 — target layer, both eval comparisons
    ax = axes[1]
    style_map = {
        "auroc_harmful": ("#d62728", "-", "Harm vs norm"),
        "auroc_harmful_vs_benign": ("#2ca02c", "-", "Harm vs benign-aggressive"),
    }
    for col, (color, ls, label) in style_map.items():
        sub = df[df["layer"] == target_layer].sort_values("n")
        ax.plot(
            sub["n"],
            sub[col],
            color=color,
            ls=ls,
            marker="o",
            markersize=3,
            lw=1.8,
            label=label,
            alpha=0.9,
        )

    ax.axhline(0.9, color="gray", linestyle=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Normative set size (N)")
    ax.set_ylabel("AUROC")
    ax.set_title(f"AUROC vs N — Layer {target_layer}{title_suffix}")
    ax.legend(fontsize=9)
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = RESULTS_DIR / f"{out_stem}.{ext}"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved → {out}")
    plt.close()


def _plot_pc1_angle(df: pd.DataFrame, layers: list[int]) -> None:
    """Appendix figure: PC1 direction drift vs N."""
    cmap = plt.get_cmap("viridis")
    layer_colors = {layer: cmap(i / max(len(layers) - 1, 1)) for i, layer in enumerate(layers)}

    _fig, ax = plt.subplots(figsize=(8, 6))
    for layer in layers:
        row = df[df["layer"] == layer].sort_values("n")
        ax.plot(
            row["n"],
            row["pc1_angle_deg"],
            marker="o",
            markersize=3,
            lw=1.8,
            color=layer_colors[layer],
            label=f"Layer {layer}",
        )

    ax.axhline(5.0, color="gray", linestyle="--", alpha=0.6, label="5° threshold")
    ax.set_xscale("log")
    ax.set_xlabel("Normative set size (N)")
    ax.set_ylabel("Angle to full-set PC1 (degrees)")
    ax.set_title("PC1 direction drift vs N")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = RESULTS_DIR / f"stability_pc1_angle.{ext}"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normative set size stability analysis.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--normative-file", default="data/raw/normative.txt")
    p.add_argument("--harmful-file", default="data/raw/harmful.txt")
    p.add_argument("--benign-agg-file", default="data/raw/benign_aggressive.txt")
    p.add_argument("--normative-n", type=int, default=500)
    p.add_argument("--harmful-n", type=int, default=520)
    p.add_argument("--benign-agg-n", type=int, default=250)
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layers to analyse. Default: all. Recommended: --layers 0 6 12 19 22",
    )
    p.add_argument(
        "--target-layer",
        type=int,
        default=19,
        help="Layer shown in the two-comparison panel.",
    )
    p.add_argument(
        "--output-dir", default="results/eval", help="Output directory. Default: results/eval."
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()
