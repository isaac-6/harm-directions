#!/usr/bin/env python3
"""
reproduce.py
------------
Reproduce the main results from:

  "Supervised Harmful Prompt Detection via Linear Discriminant Geometry
   in LLM Residual Streams" (2026)

Usage
-----
    # Full reproduction (all 12 models)
    python reproduce.py --all

    # Single model
    python reproduce.py --model Qwen/Qwen2.5-0.5B-Instruct

    # Specific strategies only
    python reproduce.py --model Qwen/Qwen2.5-0.5B-Instruct --strategies mean_diff soft_auc
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

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
    soft_auc,
    theta_normative,
    theta_two_class,
)
from harm_directions.evaluation import (
    auroc,
    effective_auroc,
    select_layer_val,
    tpr_at_fpr,
)

_DirectionFn = Callable[..., np.ndarray]
_ScoreFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


class _StrategySpec(TypedDict):
    fn: _DirectionFn
    score: _ScoreFn
    needs_harm: bool


# ---------------------------------------------------------------------------
# Models from the paper
# ---------------------------------------------------------------------------

MODELS = [
    # Qwen2.5
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "huihui-ai/Qwen2.5-0.5B-Instruct-abliterated",
    # Qwen3.5
    "Qwen/Qwen3.5-0.8B-Base",
    "Qwen/Qwen3.5-0.8B",
    "huihui-ai/Huihui-Qwen3.5-0.8B-abliterated",
    # Llama-3.2
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "huihui-ai/Llama-3.2-1B-Instruct-abliterated",
    # Gemma-3
    "google/gemma-3-1b-pt",
    "google/gemma-3-1b-it",
    "huihui-ai/gemma-3-1b-it-abliterated",
]

# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, _StrategySpec] = {
    "mean_diff": {
        "fn": mean_diff,
        "score": score_projection,
        "needs_harm": True,
    },
    "soft_auc": {
        "fn": soft_auc,
        "score": score_projection,
        "needs_harm": True,
    },
    "pc1_normative": {
        "fn": pc1_normative,
        "score": score_projection,
        "needs_harm": False,
    },
    "theta_normative": {
        "fn": theta_normative,
        "score": score_angular,
        "needs_harm": False,
    },
    "theta_two_class": {
        "fn": theta_two_class,
        "score": score_angular,
        "needs_harm": True,
    },
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_prompts(path: Path) -> list[str]:
    return [line.strip() for line in open(path) if line.strip()]


def load_all_splits(splits_dir: Path) -> dict:
    """Load all fit, validation, and eval prompt splits."""
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
# Single model evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model_id: str,
    splits: dict,
    strategies: list[str],
    pooling: str = "max",
    device: str | None = None,
) -> pd.DataFrame:
    """Run full evaluation for one model. Returns a DataFrame of results."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print(f"  Model: {model_id}")
    print(f"{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, trust_remote_code=True)
        .to(device)
        .eval()
    )  # type: ignore[arg-type]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Extract activations for all layers (fit + validation sets)
    print("Extracting fit-set activations (all layers)...")
    fit_harm_all = extract_all_layers(model, tokenizer, splits["fit_harm"], pooling=pooling)
    fit_norm_all = extract_all_layers(model, tokenizer, splits["fit_norm"], pooling=pooling)

    print("Extracting validation-set activations (all layers)...")
    val_harm_all = extract_all_layers(model, tokenizer, splits["val_harm"], pooling=pooling)
    val_norm_all = extract_all_layers(model, tokenizer, splits["val_norm"], pooling=pooling)

    # Select single operating layer via mean_diff on validation set
    print("Selecting layer by validation holdout (mean_diff)...")
    best_layer = select_layer_val(
        fit_harm_all,
        fit_norm_all,
        val_harm_all,
        val_norm_all,
    )
    print(f"  Best layer: {best_layer}")

    # Extract eval-set activations at best layer
    print("Extracting eval-set activations...")
    eval_harm_acts = {}
    for name, prompts in splits["eval_harm"].items():
        eval_harm_acts[name] = extract_activations(
            model, tokenizer, prompts, best_layer, pooling=pooling
        )
    eval_benign_acts = {}
    for name, prompts in splits["eval_benign"].items():
        eval_benign_acts[name] = extract_activations(
            model, tokenizer, prompts, best_layer, pooling=pooling
        )

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # Fit-set activations at best layer
    fit_harm = fit_harm_all[:, best_layer, :]
    fit_norm = fit_norm_all[:, best_layer, :]

    # Evaluate each strategy at the shared layer
    rows = []
    for strat_name in strategies:
        strat = STRATEGIES[strat_name]
        print(f"\n  Strategy: {strat_name}")

        t0 = time.perf_counter()
        w = strat["fn"](fit_norm, fit_harm) if strat["needs_harm"] else strat["fn"](fit_norm)
        fit_ms = (time.perf_counter() - t0) * 1000
        score_fn = strat["score"]

        for h_name, h_acts in eval_harm_acts.items():
            for b_name, b_acts in eval_benign_acts.items():
                s_harm = score_fn(h_acts, w)
                s_benign = score_fn(b_acts, w)
                raw = auroc(s_benign, s_harm)
                eff = effective_auroc(raw)

                if raw < 0.5:
                    s_harm, s_benign = -s_harm, -s_benign
                tpr = tpr_at_fpr(s_benign, s_harm)

                rows.append(
                    {
                        "model": model_id,
                        "strategy": strat_name,
                        "layer": best_layer,
                        "harmful_source": h_name,
                        "benign_source": b_name,
                        "auroc": raw,
                        "eff_auroc": eff,
                        "tpr_1pct_fpr": tpr,
                        "fit_ms": fit_ms,
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Reproduce paper results.")
    parser.add_argument("--model", type=str, default=None, help="Single model to evaluate.")
    parser.add_argument("--all", action="store_true", help="Evaluate all 12 models from the paper.")
    parser.add_argument(
        "--strategies", nargs="*", default=list(STRATEGIES.keys()), help="Strategies to evaluate."
    )
    parser.add_argument("--pooling", default="max", choices=["max", "mean", "last"])
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/raw/splits"),
        help="Directory containing the fit/val/eval text files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to write per-model CSVs and summary.",
    )
    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Provide --model or --all.")

    models = MODELS if args.all else [args.model]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if not args.splits_dir.exists():
        parser.error(
            f"Splits directory not found: {args.splits_dir}. "
            "Pass --splits-dir or run from the repo root after "
            "executing `python scripts/download_datasets.py`."
        )
    splits = load_all_splits(args.splits_dir)
    print(
        f"Loaded: {len(splits['fit_harm'])} harmful fit, "
        f"{len(splits['fit_norm'])} normative fit, "
        f"{sum(len(v) for v in splits['eval_harm'].values())} harmful eval, "
        f"{sum(len(v) for v in splits['eval_benign'].values())} benign eval."
    )

    # Evaluate
    all_dfs = []
    for model_id in models:
        df = evaluate_model(
            model_id,
            splits,
            args.strategies,
            pooling=args.pooling,
            device=args.device,
        )
        all_dfs.append(df)

        # Save per-model
        slug = model_id.replace("/", "__")
        df.to_csv(out_dir / f"{slug}_results.csv", index=False)

    # Aggregate
    if len(all_dfs) > 1:
        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df.to_csv(out_dir / "all_results.csv", index=False)

        # Summary table (mean across harm/benign source pairs)
        summary = (
            full_df.groupby(["model", "strategy"])
            .agg(
                mean_eff_auroc=("eff_auroc", "mean"),
                mean_tpr=("tpr_1pct_fpr", "mean"),
                layer=("layer", "first"),
            )
            .reset_index()
        )
        summary.to_csv(out_dir / "summary.csv", index=False)
        print(f"\n{'=' * 60}")
        print("  Summary")
        print(f"{'=' * 60}")
        print(summary.to_string(index=False))

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
