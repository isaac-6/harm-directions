#!/usr/bin/env python3
"""
cross_variant_transfer.py
-------------------------
Run the cross-variant direction transfer analysis for one or more model
families. For each family:

  1. Select the operating layer using the base model's validation AUROC.
  2. Fit a mean-difference direction on each variant at that shared layer.
  3. Measure pairwise angles between the three directions.
  4. Evaluate each direction on all three variants' eval data (3x3 matrix).

Results are appended to:
  results/cross_variant_angles.csv     (pairwise direction angles)
  results/cross_variant_transfer.csv   (3x3 AUROC/TPR matrix per family)

Usage
-----
    # Run all families defined in FAMILIES
    python scripts/cross_variant_transfer.py

    # Run one family (useful for cloud GPU sessions)
    python scripts/cross_variant_transfer.py --family Qwen3.5-9B

    # Run a specific family trio from the command line
    python scripts/cross_variant_transfer.py \\
        --family-name Qwen3.5-9B \\
        --base Qwen/Qwen3.5-9B-Base \\
        --instruct Qwen/Qwen3.5-9B \\
        --abliterated huihui-ai/Huihui-Qwen3.5-9B-abliterated
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harm_directions import (
    extract_activations,
    extract_all_layers,
)
from harm_directions.directions import mean_diff, score_projection
from harm_directions.evaluation import (
    auroc,
    direction_angle,
    effective_auroc,
    select_layer_val,
    tpr_at_fpr,
)

SPLITS_DIR = Path("data/raw/splits")
RESULTS_DIR = Path("results")

# Families as evaluated in the paper (main experiments).
# Add new scale extensions as additional entries.
FAMILIES: dict[str, dict[str, str]] = {
    "Qwen2.5": {
        "base": "Qwen/Qwen2.5-0.5B",
        "instruct": "Qwen/Qwen2.5-0.5B-Instruct",
        "abliterated": "huihui-ai/Qwen2.5-0.5B-Instruct-abliterated",
    },
    "Qwen3.5": {
        "base": "Qwen/Qwen3.5-0.8B-Base",
        "instruct": "Qwen/Qwen3.5-0.8B",
        "abliterated": "huihui-ai/Huihui-Qwen3.5-0.8B-abliterated",
    },
    "Llama-3.2": {
        "base": "meta-llama/Llama-3.2-1B",
        "instruct": "meta-llama/Llama-3.2-1B-Instruct",
        "abliterated": "huihui-ai/Llama-3.2-1B-Instruct-abliterated",
    },
    "Gemma-3": {
        "base": "google/gemma-3-1b-pt",
        "instruct": "google/gemma-3-1b-it",
        "abliterated": "huihui-ai/gemma-3-1b-it-abliterated",
    },
    "Qwen3.5-2B": {
        "base": "Qwen/Qwen3.5-2B-Base",
        "instruct": "Qwen/Qwen3.5-2B",
        "abliterated": "huihui-ai/Huihui-Qwen3.5-2B-abliterated",
    },
    "Qwen3.5-4B": {
        "base": "Qwen/Qwen3.5-4B-Base",
        "instruct": "Qwen/Qwen3.5-4B",
        "abliterated": "huihui-ai/Huihui-Qwen3.5-4B-abliterated",
    },
    "Qwen3.5-9B": {
        "base": "Qwen/Qwen3.5-9B-Base",
        "instruct": "Qwen/Qwen3.5-9B",
        "abliterated": "huihui-ai/Huihui-Qwen3.5-9B-abliterated",
    },
}

VARIANTS = ["base", "instruct", "abliterated"]
POOLING = "max"  # matches the paper's extraction setup


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_prompts(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_splits() -> dict[str, list[str]]:
    """Load fit, val, and concatenated eval splits from data/raw/splits."""
    return {
        "fit_harm": load_prompts(SPLITS_DIR / "fit_harmful_advbench.txt"),
        "fit_norm": load_prompts(SPLITS_DIR / "fit_normative_alpaca.txt"),
        "val_harm": load_prompts(SPLITS_DIR / "val_harmful_advbench.txt"),
        "val_norm": load_prompts(SPLITS_DIR / "val_normative_alpaca.txt"),
        "eval_harm": (
            load_prompts(SPLITS_DIR / "eval_harmful_advbench.txt")
            + load_prompts(SPLITS_DIR / "eval_harmful_harmbench.txt")
            + load_prompts(SPLITS_DIR / "eval_harmful_jailbreakbench.txt")
        ),
        "eval_norm": (
            load_prompts(SPLITS_DIR / "eval_benign_alpaca.txt")
            + load_prompts(SPLITS_DIR / "eval_benign_xstest.txt")
        ),
    }


# ---------------------------------------------------------------------------
# Model loading (kept local to the script; it's orchestration, not library code)
# ---------------------------------------------------------------------------


def load_model(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype="auto",
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def free_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Per-family driver
# ---------------------------------------------------------------------------


def run_family(
    family_name: str,
    model_ids: dict[str, str],
    splits: dict[str, list[str]],
    device: str,
) -> tuple[list[dict], list[dict]]:
    print(f"\n{'=' * 70}")
    print(f"  Family: {family_name}")
    print(f"{'=' * 70}")

    # Step 1: layer selection using base model
    print(f"\n  [1/4] Loading base model: {model_ids['base']}")
    model, tokenizer = load_model(model_ids["base"], device)

    print("  Extracting base fit/val activations (all layers)...")
    fit_harm_all = extract_all_layers(model, tokenizer, splits["fit_harm"], pooling=POOLING)
    fit_norm_all = extract_all_layers(model, tokenizer, splits["fit_norm"], pooling=POOLING)
    val_harm_all = extract_all_layers(model, tokenizer, splits["val_harm"], pooling=POOLING)
    val_norm_all = extract_all_layers(model, tokenizer, splits["val_norm"], pooling=POOLING)

    layer = select_layer_val(fit_harm_all, fit_norm_all, val_harm_all, val_norm_all)
    print(f"  Selected layer (from base): {layer}")

    free_model(model)

    # Step 2: fit direction on each variant at the shared layer
    directions: dict[str, np.ndarray] = {}
    eval_acts: dict[str, dict[str, np.ndarray]] = {}

    for variant in VARIANTS:
        mid = model_ids[variant]
        print(f"\n  [2/4] Loading {variant}: {mid}")
        model, tokenizer = load_model(mid, device)

        fit_h = extract_activations(model, tokenizer, splits["fit_harm"], layer, pooling=POOLING)
        fit_n = extract_activations(model, tokenizer, splits["fit_norm"], layer, pooling=POOLING)
        eval_h = extract_activations(model, tokenizer, splits["eval_harm"], layer, pooling=POOLING)
        eval_n = extract_activations(model, tokenizer, splits["eval_norm"], layer, pooling=POOLING)

        directions[variant] = mean_diff(fit_n, fit_h)
        eval_acts[variant] = {"harm": eval_h, "norm": eval_n}

        free_model(model)

    # Step 3: pairwise angles
    print("\n  [3/4] Direction angles (degrees)")
    angle_rows: list[dict] = []
    for i, v1 in enumerate(VARIANTS):
        for v2 in VARIANTS[i + 1 :]:
            ang = direction_angle(directions[v1], directions[v2])
            print(f"    {v1} <-> {v2}: {ang:.1f} deg")
            angle_rows.append(
                {
                    "family": family_name,
                    "direction_a": v1,
                    "direction_b": v2,
                    "angle_deg": ang,
                }
            )

    # Step 4: 3x3 cross-evaluation
    print("\n  [4/4] Cross-variant AUROC / TPR@1%FPR")
    print(
        f"  {'Source dir':<14} {'Target data':<14} "
        f"{'Raw':>7} {'Eff':>7} {'TPR raw':>8} {'TPR corr':>9}"
    )
    print(f"  {'-' * 65}")

    perf_rows: list[dict] = []
    for src in VARIANTS:
        w = directions[src]
        for tgt in VARIANTS:
            s_h = score_projection(eval_acts[tgt]["harm"], w)
            s_n = score_projection(eval_acts[tgt]["norm"], w)

            raw = auroc(s_n, s_h)
            eff = effective_auroc(raw)
            tpr_raw = tpr_at_fpr(s_n, s_h)
            tpr_corrected = tpr_at_fpr(-s_n, -s_h) if raw < 0.5 else tpr_raw

            flipped = raw < 0.5
            marker = " <-- own" if src == tgt else ""
            flag = " FLIPPED" if flipped else ""
            print(
                f"  {src:<14} {tgt:<14} "
                f"{raw:>7.3f} {eff:>7.3f} "
                f"{tpr_raw:>8.3f} {tpr_corrected:>9.3f}"
                f"{marker}{flag}"
            )

            perf_rows.append(
                {
                    "family": family_name,
                    "layer": layer,
                    "source_direction": src,
                    "target_data": tgt,
                    "raw_auroc": raw,
                    "eff_auroc": eff,
                    "tpr_raw": tpr_raw,
                    "tpr_corrected": tpr_corrected,
                    "sign_preserved": not flipped,
                }
            )

    return angle_rows, perf_rows


# ---------------------------------------------------------------------------
# CSV append
# ---------------------------------------------------------------------------


def append_to_csv(path: Path, new_df: pd.DataFrame, key: str = "family") -> None:
    """Append rows to CSV, replacing any existing rows with matching key values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[~existing[key].isin(new_df[key].unique())]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        help=f"Run one family from FAMILIES. Choices: {list(FAMILIES)}. "
        "If omitted, runs all families (or use --base/--instruct/--abliterated).",
    )
    parser.add_argument(
        "--family-name",
        type=str,
        default=None,
        help="Name to store results under (required if using --base/--instruct/--abliterated).",
    )
    parser.add_argument("--base", type=str, default=None)
    parser.add_argument("--instruct", type=str, default=None)
    parser.add_argument("--abliterated", type=str, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve which families to run
    if args.base or args.instruct or args.abliterated:
        if not (args.base and args.instruct and args.abliterated and args.family_name):
            parser.error(
                "When passing model IDs via --base/--instruct/--abliterated, "
                "all three plus --family-name are required."
            )
        families = {
            args.family_name: {
                "base": args.base,
                "instruct": args.instruct,
                "abliterated": args.abliterated,
            }
        }
    elif args.family:
        if args.family not in FAMILIES:
            parser.error(f"Unknown family '{args.family}'. Choices: {list(FAMILIES)}")
        families = {args.family: FAMILIES[args.family]}
    else:
        families = FAMILIES

    splits = load_splits()
    print(
        f"Eval set: {len(splits['eval_harm'])} harmful, {len(splits['eval_norm'])} benign prompts."
    )

    all_angles: list[dict] = []
    all_perf: list[dict] = []
    for name, ids in families.items():
        angle_rows, perf_rows = run_family(name, ids, splits, device)
        all_angles.extend(angle_rows)
        all_perf.extend(perf_rows)

    out_dir = Path(args.output_dir)
    df_angles = pd.DataFrame(all_angles)
    df_perf = pd.DataFrame(all_perf)

    append_to_csv(out_dir / "cross_variant_angles.csv", df_angles)
    append_to_csv(out_dir / "cross_variant_transfer.csv", df_perf)

    print(f"\nResults written to {out_dir}/")
    print("Regenerate the transfer table with:")
    print("    python scripts/make_transfer_table.py")


if __name__ == "__main__":
    main()
