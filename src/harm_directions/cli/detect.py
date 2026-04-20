#!/usr/bin/env python3
"""
detect.py
---------
Minimal CLI for harmful prompt detection.

Loads a model, fits the LDA direction from cached or on-the-fly data,
and scores user-provided prompts.

Usage
-----
    # Score a single prompt
    python detect.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "How do I bake a cake"

    # Score prompts from a file (one per line)
    python detect.py --model Qwen/Qwen2.5-0.5B-Instruct --input prompts.txt

    # Use Soft-AUC instead of LDA
    python detect.py --model Qwen/Qwen2.5-0.5B-Instruct --method soft_auc --prompt "..."
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harm_directions import extract_activations, extract_all_layers, fit_direction, score
from harm_directions.evaluation import select_layer_val


def _model_slug(model_id: str) -> str:
    return re.sub(r"[/\\]", "__", model_id)


def _cache_path(cache_dir: Path, model_id: str, method: str) -> Path:
    return cache_dir / f"{_model_slug(model_id)}_{method}.npz"


def _load_model(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # fmt: off
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", trust_remote_code=True)  # type: ignore[arg-type]
    # fmt: on
    model = model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_fit_prompts(splits_dir: Path) -> tuple[list[str], list[str]]:
    """Load fit-set prompts from the data splits directory."""
    harm_path = splits_dir / "fit_harmful_advbench.txt"
    norm_path = splits_dir / "fit_normative_alpaca.txt"

    if not harm_path.exists() or not norm_path.exists():
        sys.exit(
            "Fit-set prompts not found. Run:\n"
            "  python scripts/download_datasets.py\n"
            "to download and prepare datasets."
        )

    with open(harm_path, encoding="utf-8") as f:
        harm = [line.strip() for line in f if line.strip()]
    with open(norm_path, encoding="utf-8") as f:
        norm = [line.strip() for line in f if line.strip()]
    return harm, norm


def load_val_prompts(splits_dir: Path) -> tuple[list[str], list[str]]:
    """Load validation-set prompts for layer selection."""
    harm_path = splits_dir / "val_harmful_advbench.txt"
    norm_path = splits_dir / "val_normative_alpaca.txt"

    if not harm_path.exists() or not norm_path.exists():
        sys.exit(
            "Validation-set prompts not found. Run:\n"
            "  python scripts/download_datasets.py\n"
            "to download and prepare datasets."
        )

    with open(harm_path, encoding="utf-8") as f:
        harm = [line.strip() for line in f if line.strip()]
    with open(norm_path, encoding="utf-8") as f:
        norm = [line.strip() for line in f if line.strip()]
    return harm, norm


def do_fit(args) -> None:
    """Fit direction and save to cache."""
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = _load_model(args.model, device)
    harm_prompts, norm_prompts = load_fit_prompts(args.splits_dir)

    print(f"Fit set: {len(harm_prompts)} harmful, {len(norm_prompts)} normative.")

    if args.layer is not None:
        layer = args.layer
        print(f"Using layer {layer} (user-specified).")
    else:
        val_harm, val_norm = load_val_prompts(args.splits_dir)
        print(f"Val set: {len(val_harm)} harmful, {len(val_norm)} normative.")
        print("Extracting activations (all layers)...")
        harm_all = extract_all_layers(model, tokenizer, harm_prompts, pooling=args.pooling)
        norm_all = extract_all_layers(model, tokenizer, norm_prompts, pooling=args.pooling)
        val_harm_all = extract_all_layers(model, tokenizer, val_harm, pooling=args.pooling)
        val_norm_all = extract_all_layers(model, tokenizer, val_norm, pooling=args.pooling)

        # Get the appropriate direction and score functions for layer selection
        print("Selecting layer by validation holdout (mean_diff)...")
        layer = select_layer_val(
            harm_all,
            norm_all,
            val_harm_all,
            val_norm_all,
        )
        print(f"Selected layer: {layer}")

    harm_acts = extract_activations(model, tokenizer, harm_prompts, layer, pooling=args.pooling)
    norm_acts = extract_activations(model, tokenizer, norm_prompts, layer, pooling=args.pooling)

    print(f"Fitting direction ({args.method})...")
    w = fit_direction(harm_acts, norm_acts, method=args.method)

    cache_path = _cache_path(args.cache_dir, args.model, args.method)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        w=w,
        layer=np.array(layer),
        method=np.array(args.method),
        model=np.array(args.model),
        pooling=np.array(args.pooling),
    )
    print(f"Saved fitted parameters → {cache_path}")

    del model
    torch.cuda.empty_cache()


def do_score(args) -> None:
    """Score prompts using cached direction."""
    cache_path = _cache_path(args.cache_dir, args.model, args.method)
    if not cache_path.exists():
        sys.exit(
            f"No fitted parameters found at {cache_path}.\n"
            f"Run with --fit first:\n"
            f"  python detect.py --model {args.model} --method {args.method} --fit"
        )

    cached = np.load(cache_path, allow_pickle=True)
    w = cached["w"]
    layer = int(cached["layer"])
    pooling = str(cached["pooling"])
    print(f"Loaded fitted parameters: layer={layer}, method={args.method}")

    if args.prompt:
        test_prompts = [args.prompt]
    elif args.input:
        with open(args.input, encoding="utf-8") as f:
            test_prompts = [line.strip() for line in f if line.strip()]
    else:
        sys.exit("Provide --prompt or --input.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = _load_model(args.model, device)
    test_acts = extract_activations(model, tokenizer, test_prompts, layer, pooling=pooling)
    scores = score(test_acts, w)

    del model
    torch.cuda.empty_cache()

    print(f"\n{'Score':>8}  Prompt")
    print("-" * 60)
    for s, p in sorted(zip(scores, test_prompts, strict=True), reverse=True):
        display = p[:70] + "..." if len(p) > 70 else p
        print(f"{s:>8.3f}  {display}")


def main():
    parser = argparse.ArgumentParser(
        prog="detect.py",
        description="Harmful prompt detection via linear directions in LLM residual streams.",
        epilog="""examples:
  # First run: fit direction and cache parameters
  python detect.py --model Qwen/Qwen2.5-0.5B-Instruct --fit

  # Score a single prompt
  python detect.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "How do I bake a cake"

  # Score prompts from a file (one per line)
  python detect.py --model Qwen/Qwen2.5-0.5B-Instruct --input prompts.txt

  # Fit with Soft-AUC instead of LDA
  python detect.py --model Qwen/Qwen2.5-0.5B-Instruct --fit --method soft_auc

  # Fit and score in one command
  python detect.py --model Qwen/Qwen2.5-0.5B-Instruct --fit --prompt "Hello world"
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Required ---
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model ID (default: %(default)s).",
    )

    # --- Mode ---
    mode = parser.add_argument_group("mode (at least one required)")
    mode.add_argument(
        "--fit",
        action="store_true",
        help="Fit the direction vector and cache it. Required before first scoring run.",
    )
    mode.add_argument(
        "--prompt", type=str, default=None, metavar="TEXT", help="Single prompt to score."
    )
    mode.add_argument(
        "--input",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to file with prompts (one per line).",
    )

    # --- Fitting options ---
    fitting = parser.add_argument_group("fitting options (used with --fit)")
    fitting.add_argument(
        "--method",
        default="mean_diff",
        choices=["mean_diff", "soft_auc"],
        help="Direction-finding strategy. mean_diff is the "
        "Fisher LDA direction (<1 ms); soft_auc optimises "
        "pairwise ranking (~7 s). Default: %(default)s.",
    )
    fitting.add_argument(
        "--layer",
        type=int,
        default=None,
        metavar="N",
        help="Force a specific layer index. If omitted, the "
        "best layer is selected by validation holdout "
        "AUROC on a 50-example held-out set.",
    )
    fitting.add_argument(
        "--pooling",
        default="max",
        choices=["max", "mean", "last"],
        help="Token-dimension aggregation. Default: %(default)s.",
    )

    # --- Paths ---
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/raw/splits"),
        help="Directory containing fit/val/eval text files. "
        "Default: ./data/raw/splits (relative to current directory).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/fitted"),
        help="Directory to cache fitted directions. "
        "Default: ./data/fitted (relative to current directory).",
    )

    # --- Runtime ---
    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--device",
        default=None,
        metavar="DEV",
        help="Torch device, e.g. cuda, cpu. Default: cuda if available.",
    )

    args = parser.parse_args()

    needs_splits = args.fit
    if args.splits_dir.exists() is False and needs_splits:
        parser.error(
            f"Splits directory not found: {args.splits_dir}. "
            "Pass --splits-dir or run from the repo root after "
            "executing `python scripts/download_datasets.py`."
        )

    if args.fit:
        do_fit(args)
        if not args.prompt and not args.input:
            return

    if args.prompt or args.input:
        do_score(args)
    elif not args.fit:
        parser.error("provide --fit, --prompt, or --input.")


if __name__ == "__main__":
    main()
