#!/usr/bin/env python3
"""
generate_figure_artifacts.py
----------------------------
Re-runs extraction and fitting across the 12 paper models, saving per-prompt
scores, per-layer AUROC sweeps, direction vectors, and cross-strategy angles
to results/artifacts/<model_slug>/.

Unlike reproduce.py which saves aggregated results, this saves the underlying
data needed to regenerate figures (bootstrap CIs, score distributions, angle
plots, layer profiles).

Usage
-----
    # Run all 12 models, skipping those with complete artifacts
    python .beta/generate_figure_artifacts.py

    # Force re-run of a specific model
    python .beta/generate_figure_artifacts.py \
        --model Qwen/Qwen2.5-0.5B-Instruct --refresh

    # Different output directory
    python .beta/generate_figure_artifacts.py --output-dir results/artifacts
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harm_directions import (
    extract_activations,
    extract_all_layers,
    score_angular,
    score_projection,
)
from harm_directions.directions import (
    mean_diff,
    pc1_normative,
    random_direction,
    soft_auc,
    theta_normative,
    theta_two_class,
)
from harm_directions.evaluation import (
    auroc,
    direction_angle,
    effective_auroc,
    select_layer_val,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "huihui-ai/Qwen2.5-0.5B-Instruct-abliterated",
    "Qwen/Qwen3.5-0.8B-Base",
    "Qwen/Qwen3.5-0.8B",
    "huihui-ai/Huihui-Qwen3.5-0.8B-abliterated",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "huihui-ai/Llama-3.2-1B-Instruct-abliterated",
    "google/gemma-3-1b-pt",
    "google/gemma-3-1b-it",
    "huihui-ai/gemma-3-1b-it-abliterated",
]

# Strategy definitions. Each: direction function, scoring function, whether harm labels are needed.
STRATEGIES: dict[str, dict] = {
    "pc1_normative": {
        "fit": lambda fit_norm, fit_harm: pc1_normative(fit_norm),
        "score": score_projection,
        "projection": True,
    },
    "theta_normative": {
        "fit": lambda fit_norm, fit_harm: theta_normative(fit_norm),
        "score": score_angular,
        "projection": False,
    },
    "mean_diff": {
        "fit": lambda fit_norm, fit_harm: mean_diff(fit_norm, fit_harm),
        "score": score_projection,
        "projection": True,
    },
    "soft_auc": {
        "fit": lambda fit_norm, fit_harm: soft_auc(fit_norm, fit_harm),
        "score": score_projection,
        "projection": True,
    },
    "theta_two_class": {
        "fit": lambda fit_norm, fit_harm: theta_two_class(fit_norm, fit_harm),
        "score": score_angular,
        "projection": False,
    },
    "random": {
        "fit": lambda fit_norm, fit_harm: random_direction(fit_norm.shape[1], seed=42),
        "score": score_projection,
        "projection": True,
    },
}

# Angle pairs to compute. Format: (strategy_a, strategy_b, pair_label).
# Includes pairs cited in §4.3 plus a few reference pairs.
ANGLE_PAIRS: list[tuple[str, str]] = [
    ("mean_diff", "soft_auc"),
    ("mean_diff", "pc1_normative"),
    ("mean_diff", "theta_two_class"),
    ("mean_diff", "random"),
    ("soft_auc", "pc1_normative"),
    ("soft_auc", "theta_two_class"),
    ("pc1_normative", "pc1_harmful"),  # special: fitted from harmful PCs
    ("theta_normative", "theta_two_class"),
]


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_prompts(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_splits(splits_dir: Path) -> dict:
    return {
        "fit_harm": load_prompts(splits_dir / "fit_harmful_advbench.txt"),
        "fit_norm": load_prompts(splits_dir / "fit_normative_alpaca.txt"),
        "val_harm": load_prompts(splits_dir / "val_harmful_advbench.txt"),
        "val_norm": load_prompts(splits_dir / "val_normative_alpaca.txt"),
        "eval_harm": {
            p.stem.replace("eval_harmful_", ""): load_prompts(p)
            for p in sorted(splits_dir.glob("eval_harmful_*.txt"))
        },
        "eval_benign": {
            p.stem.replace("eval_benign_", ""): load_prompts(p)
            for p in sorted(splits_dir.glob("eval_benign_*.txt"))
        },
    }


# ---------------------------------------------------------------------------
# Perplexity baseline
# ---------------------------------------------------------------------------


def compute_perplexity(
    model,
    tokenizer,
    prompts: list[str],
    device: str,
) -> np.ndarray:
    """Mean per-token NLL for each prompt. Higher = more surprising."""
    scores = []
    with torch.no_grad():
        for p in prompts:
            inputs = tokenizer(p, return_tensors="pt").to(device)
            ids = inputs["input_ids"]
            if ids.shape[1] < 2:
                scores.append(float("nan"))
                continue
            out = model(**inputs, labels=ids)
            scores.append(float(out.loss.cpu()))
    return np.array(scores, dtype=np.float64)


# ---------------------------------------------------------------------------
# Per-model pipeline
# ---------------------------------------------------------------------------


def run_model(
    model_id: str,
    splits: dict,
    output_dir: Path,
    device: str,
    refresh: bool,
) -> None:
    slug = model_slug(model_id)
    model_out = output_dir / slug
    model_out.mkdir(parents=True, exist_ok=True)

    score_csv = model_out / "score_distributions.csv"
    layer_csv = model_out / "layer_sweep.csv"
    dirs_npz = model_out / "directions.npz"
    angles_csv = model_out / "angles.csv"

    if (not refresh) and all(p.exists() for p in [score_csv, layer_csv, dirs_npz, angles_csv]):
        print(f"[skip] {model_id} (all artifacts present)")
        return

    print(f"\n{'=' * 70}")
    print(f"  {model_id}")
    print(f"{'=' * 70}")

    t0 = time.time()

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", trust_remote_code=True)
        .to(device)
        .eval()
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Extract activations (fit + val, all layers) ----
    print("Extracting fit/val activations (all layers)...")
    fit_harm_all = extract_all_layers(model, tokenizer, splits["fit_harm"], pooling="max")
    fit_norm_all = extract_all_layers(model, tokenizer, splits["fit_norm"], pooling="max")
    val_harm_all = extract_all_layers(model, tokenizer, splits["val_harm"], pooling="max")
    val_norm_all = extract_all_layers(model, tokenizer, splits["val_norm"], pooling="max")

    n_layers = fit_harm_all.shape[1]
    d_model = fit_harm_all.shape[2]
    print(f"  Layers: {n_layers}, D: {d_model}")

    # ---- Layer selection (mean_diff on validation) ----
    best_layer = select_layer_val(fit_harm_all, fit_norm_all, val_harm_all, val_norm_all)
    print(f"Selected layer: {best_layer}")

    # ---- Per-layer AUROC sweep for all strategies ----
    print("Computing per-layer AUROC sweep...")
    layer_rows = []
    for layer in range(n_layers):
        fh = fit_harm_all[:, layer, :]
        fn = fit_norm_all[:, layer, :]
        vh = val_harm_all[:, layer, :]
        vn = val_norm_all[:, layer, :]
        for strat_name, strat in STRATEGIES.items():
            w = strat["fit"](fn, fh)
            s_h = strat["score"](vh, w)
            s_n = strat["score"](vn, w)
            layer_rows.append(
                {
                    "layer": layer,
                    "strategy": strat_name,
                    "auroc_raw": auroc(s_n, s_h),
                    "auroc_eff": effective_auroc(auroc(s_n, s_h)),
                }
            )
    pd.DataFrame(layer_rows).to_csv(layer_csv, index=False)
    print(f"  Wrote {layer_csv}")

    # ---- Extract eval-set activations (at best layer only) ----
    print("Extracting eval-set activations at best layer...")
    eval_harm_acts = {
        name: extract_activations(model, tokenizer, prompts, best_layer, pooling="max")
        for name, prompts in splits["eval_harm"].items()
    }
    eval_benign_acts = {
        name: extract_activations(model, tokenizer, prompts, best_layer, pooling="max")
        for name, prompts in splits["eval_benign"].items()
    }

    # ---- Compute perplexity for all eval prompts ----
    print("Computing perplexity baseline...")
    eval_harm_ppl = {
        name: compute_perplexity(model, tokenizer, prompts, device)
        for name, prompts in splits["eval_harm"].items()
    }
    eval_benign_ppl = {
        name: compute_perplexity(model, tokenizer, prompts, device)
        for name, prompts in splits["eval_benign"].items()
    }

    # Free GPU memory: we have what we need as numpy arrays
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Fit directions at best layer ----
    print("Fitting directions...")
    fh_best = fit_harm_all[:, best_layer, :]
    fn_best = fit_norm_all[:, best_layer, :]

    directions = {}
    for strat_name, strat in STRATEGIES.items():
        directions[strat_name] = strat["fit"](fn_best, fh_best)

    # Additional direction for angle pair: pc1 of harmful activations
    from sklearn.decomposition import PCA

    pca_harm = PCA(n_components=1).fit(fh_best)
    directions["pc1_harmful"] = pca_harm.components_[0]

    # Save directions
    np.savez(
        dirs_npz,
        layer=np.array(best_layer),
        **directions,
    )
    print(f"  Wrote {dirs_npz}")

    # ---- Compute cross-strategy angles ----
    print("Computing cross-strategy angles...")
    angle_rows = []
    for a, b in ANGLE_PAIRS:
        if a not in directions or b not in directions:
            continue
        ang = direction_angle(directions[a], directions[b])
        angle_rows.append(
            {
                "model": model_id,
                "layer": best_layer,
                "direction_a": a,
                "direction_b": b,
                "angle_deg": ang,
            }
        )
    pd.DataFrame(angle_rows).to_csv(angles_csv, index=False)
    print(f"  Wrote {angles_csv}")

    # ---- Score all eval prompts with all strategies ----
    print("Scoring all eval prompts...")
    score_rows = []
    for strat_name, strat in STRATEGIES.items():
        w = directions[strat_name]
        for source_name, acts in eval_harm_acts.items():
            s = strat["score"](acts, w)
            for i, score_val in enumerate(s):
                score_rows.append(
                    {
                        "model": model_id,
                        "strategy": strat_name,
                        "layer": best_layer,
                        "split": "harmful",
                        "source": source_name,
                        "prompt_idx": i,
                        "score": float(score_val),
                    }
                )
        for source_name, acts in eval_benign_acts.items():
            s = strat["score"](acts, w)
            for i, score_val in enumerate(s):
                score_rows.append(
                    {
                        "model": model_id,
                        "strategy": strat_name,
                        "layer": best_layer,
                        "split": "benign",
                        "source": source_name,
                        "prompt_idx": i,
                        "score": float(score_val),
                    }
                )

    # Perplexity baseline rows
    for source_name, ppl in eval_harm_ppl.items():
        for i, score_val in enumerate(ppl):
            score_rows.append(
                {
                    "model": model_id,
                    "strategy": "perplexity",
                    "layer": -1,  # not applicable
                    "split": "harmful",
                    "source": source_name,
                    "prompt_idx": i,
                    "score": float(score_val),
                }
            )
    for source_name, ppl in eval_benign_ppl.items():
        for i, score_val in enumerate(ppl):
            score_rows.append(
                {
                    "model": model_id,
                    "strategy": "perplexity",
                    "layer": -1,
                    "split": "benign",
                    "source": source_name,
                    "prompt_idx": i,
                    "score": float(score_val),
                }
            )

    pd.DataFrame(score_rows).to_csv(score_csv, index=False)
    print(f"  Wrote {score_csv}")

    dt = time.time() - t0
    print(f"  Done in {dt / 60:.1f} min")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=str, default=None, help="Run only this model (default: all)."
    )
    parser.add_argument("--splits-dir", type=Path, default=Path("data/raw/splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/artifacts"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--refresh", action="store_true", help="Re-run even if artifacts exist.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    models = [args.model] if args.model else MODELS

    print(f"Loading splits from {args.splits_dir}")
    splits = load_splits(args.splits_dir)
    print(f"  Fit: {len(splits['fit_harm'])} harm, {len(splits['fit_norm'])} norm")
    print(f"  Val: {len(splits['val_harm'])} harm, {len(splits['val_norm'])} norm")
    print(f"  Eval harm sources: {list(splits['eval_harm'])}")
    print(f"  Eval benign sources: {list(splits['eval_benign'])}")
    print(f"\nOutput: {args.output_dir}")
    print(f"Models: {len(models)}")

    for model_id in models:
        try:
            run_model(model_id, splits, args.output_dir, device, args.refresh)
        except Exception as e:
            print(f"\n[ERROR] {model_id}: {e}")
            raise


if __name__ == "__main__":
    main()
