#!/usr/bin/env python3
r"""
run_g4_self_projected_refit.py
-------------------------------
G4 -- Refit harm direction in self-orthogonal subspace.

Test whether harm is linearly *decodable* through multiple
non-collinear directions (rather than encoded along a single
privileged axis):

  1. Fit $\wmd$ at the operating layer (validation-selected for the
     given protocol).
  2. Project $\wmd$ out of fit, val, eval activations at that layer:
     $x' = x - (x \cdot \wmd)\,\wmd$.
  3. Refit $\wopt$ from scratch by Soft-AUC optimisation on the
     projected fit set. Note: refitting $\wmd$ on projected data
     yields a degenerate (zero-magnitude) direction by
     construction, since $\mu_H - \mu_N$ is collinear with the
     projected-out direction; we therefore evaluate only $\wopt'$.
  4. Evaluate $\wopt'$ on projected eval activations. Bootstrap CIs.

Two protocols are tested:
  - Main analysis: max-pool over raw prompts (12 models).
  - Deployment-realistic: last-token of chat-templated input
    (4 instruction-tuned models, one per family).

Outputs:
  results/g4_self_projected_refit.csv
  results/table_g4.tex
"""

from __future__ import annotations

import argparse
import gc

# Reuse helpers from G1 and G2
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
import run_g1_chat_template as g1

# Same instruct models as G1/G2
INSTRUCT_MODELS = g1.INSTRUCT_MODELS

# All 12 models for the main-protocol run
ALL_MODELS = [
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


def project_out(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Project out the unit-vector w from x along the last axis."""
    w_unit = w / (np.linalg.norm(w) + 1e-12)
    if x.ndim == 2:
        return x - np.outer(x @ w_unit, w_unit)
    return x - (x @ w_unit) * w_unit


@torch.no_grad()
def extract_maxpool_raw_per_layer(model, tokenizer, prompts, device):
    """Max-pool over all tokens of the *raw* prompt (no chat template)."""
    out = []
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        layer_acts = [
            h[0].float().max(dim=0).values.cpu().numpy() for h in outputs.hidden_states[1:]
        ]
        out.append(np.stack(layer_acts, axis=0))
    return np.stack(out, axis=0)


def evaluate(scores_h, scores_b, eval_harm_src, eval_benign_src):
    scores_arr = np.concatenate([scores_b, scores_h])
    labels = np.concatenate([np.zeros(len(scores_b)), np.ones(len(scores_h))])
    sources = np.concatenate([eval_benign_src, eval_harm_src])
    return g1.bootstrap_metrics(scores_arr, labels, sources)


def run_protocol(model_id, splits, out_dir, device, protocol):
    """
    protocol in {"maxpool_raw", "lasttok_templated"}.
    Fits w_LDA at the protocol's validation-selected layer, projects it out,
    refits w_opt on projected activations, evaluates.
    """
    print(f"\n{'=' * 70}\n  G4 [{protocol}]: {model_id}\n{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", trust_remote_code=True)
        .to(device)
        .eval()
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare inputs
    eval_harm_prompts, eval_harm_src = g1.flatten_eval(splits["eval_harm"])
    eval_benign_prompts, eval_benign_src = g1.flatten_eval(splits["eval_benign"])

    if protocol == "maxpool_raw":
        # Raw prompts (no template)
        fit_h_in = splits["fit_harm"]
        fit_b_in = splits["fit_benign"]
        val_h_in = splits["val_harm"]
        val_b_in = splits["val_benign"]
        eval_h_in = eval_harm_prompts
        eval_b_in = eval_benign_prompts
        extract = extract_maxpool_raw_per_layer
    elif protocol == "lasttok_templated":
        # Chat-templated, last-token extraction
        fit_h_in = [g1.apply_chat_template(tokenizer, p) for p in splits["fit_harm"]]
        fit_b_in = [g1.apply_chat_template(tokenizer, p) for p in splits["fit_benign"]]
        val_h_in = [g1.apply_chat_template(tokenizer, p) for p in splits["val_harm"]]
        val_b_in = [g1.apply_chat_template(tokenizer, p) for p in splits["val_benign"]]
        eval_h_in = [g1.apply_chat_template(tokenizer, p) for p in eval_harm_prompts]
        eval_b_in = [g1.apply_chat_template(tokenizer, p) for p in eval_benign_prompts]
        extract = g1.extract_lasttok_per_layer
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    print(f"  Extracting activations ({protocol})...")
    fit_h = extract(model, tokenizer, fit_h_in, device)
    fit_b = extract(model, tokenizer, fit_b_in, device)
    val_h = extract(model, tokenizer, val_h_in, device)
    val_b = extract(model, tokenizer, val_b_in, device)
    eval_h = extract(model, tokenizer, eval_h_in, device)
    eval_b = extract(model, tokenizer, eval_b_in, device)

    # Validation-selected layer
    L = g1.select_best_layer(fit_h, fit_b, val_h, val_b)
    print(f"  Selected layer L = {L}")

    fit_h_L = fit_h[:, L]
    fit_b_L = fit_b[:, L]
    eval_h_L = eval_h[:, L]
    eval_b_L = eval_b[:, L]

    # Baseline at L (unprojected)
    w_lda = g1.fit_mean_diff(fit_h_L, fit_b_L)
    w_opt = g1.fit_soft_auc(fit_h_L, fit_b_L, init=w_lda)

    s_h_lda = eval_h_L @ w_lda
    s_b_lda = eval_b_L @ w_lda
    m_lda_baseline = evaluate(s_h_lda, s_b_lda, eval_harm_src, eval_benign_src)

    s_h_opt = eval_h_L @ w_opt
    s_b_opt = eval_b_L @ w_opt
    m_opt_baseline = evaluate(s_h_opt, s_b_opt, eval_harm_src, eval_benign_src)

    # Project out w_LDA from all activations at L
    fit_h_proj = project_out(fit_h_L, w_lda)
    fit_b_proj = project_out(fit_b_L, w_lda)
    eval_h_proj = project_out(eval_h_L, w_lda)
    eval_b_proj = project_out(eval_b_L, w_lda)

    # Refit w_opt on projected activations.
    # Note: refitting w_LDA on projected activations yields a near-zero
    # direction by construction (mu_H - mu_N is collinear with w_LDA);
    # we therefore evaluate only w_opt' here. We do report the magnitude
    # of mu_H - mu_N on projected activations as a sanity check.
    mu_h_proj = fit_h_proj.mean(axis=0)
    mu_b_proj = fit_b_proj.mean(axis=0)
    md_proj_norm = float(np.linalg.norm(mu_h_proj - mu_b_proj))
    md_orig_norm = float(np.linalg.norm(fit_h_L.mean(axis=0) - fit_b_L.mean(axis=0)))
    print(f"  ||mu_H - mu_N|| (orig)      = {md_orig_norm:.4e}")
    print(f"  ||mu_H - mu_N|| (projected) = {md_proj_norm:.4e}")

    # Initialise Soft-AUC from a random unit vector to avoid the degenerate
    # mean-diff init in the orthogonal subspace.
    rng = np.random.default_rng(42)
    init = rng.normal(size=fit_h_L.shape[1])
    init = init / np.linalg.norm(init)
    # Make sure init lives in the orthogonal subspace
    init = init - (init @ w_lda) * w_lda
    init = init / np.linalg.norm(init)

    w_opt_refit = g1.fit_soft_auc(fit_h_proj, fit_b_proj, init=init)

    # Evaluate refit
    s_h_refit = eval_h_proj @ w_opt_refit
    s_b_refit = eval_b_proj @ w_opt_refit
    m_refit = evaluate(s_h_refit, s_b_refit, eval_harm_src, eval_benign_src)

    # Angles
    angle_orig_vs_refit = g1.angle_deg(w_opt, w_opt_refit)
    angle_lda_vs_refit = g1.angle_deg(w_lda, w_opt_refit)  # should be ~90

    rows = [
        {
            "model": model_id,
            "protocol": protocol,
            "layer": L,
            "condition": "baseline_w_lda",
            **m_lda_baseline,
        },
        {
            "model": model_id,
            "protocol": protocol,
            "layer": L,
            "condition": "baseline_w_opt",
            **m_opt_baseline,
        },
        {
            "model": model_id,
            "protocol": protocol,
            "layer": L,
            "condition": "w_opt_refit_after_projecting_w_lda",
            **m_refit,
        },
    ]

    angles = {
        "model": model_id,
        "protocol": protocol,
        "layer": L,
        "md_orig_norm": md_orig_norm,
        "md_proj_norm": md_proj_norm,
        "md_proj_norm_ratio": md_proj_norm / (md_orig_norm + 1e-12),
        "angle_baseline_opt_vs_refit": angle_orig_vs_refit,
        "angle_lda_vs_refit": angle_lda_vs_refit,
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {"rows": rows, "angles": angles}


def write_table(df, out_path):
    """Build a single LaTeX table for the soft_auc results."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{G4: Refit of $\wopt$ in the subspace orthogonal to "
        r"$\wmd$, at the validation-selected layer per model under each "
        r"extraction protocol. Baseline rows: $\wmd$ and $\wopt$ AUROC "
        r"on unprojected activations. Refit row: $\wopt'$ trained from "
        r"scratch on activations with $\wmd$ projected out. AUROC with "
        r"stratified bootstrap 95\% CIs.}",
        r"\label{tab:g4_refit}",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Model & Protocol & Condition & AUROC [95\% CI] \\",
        r"\midrule",
    ]
    cond_label = {
        "baseline_w_lda": r"baseline $\wmd$",
        "baseline_w_opt": r"baseline $\wopt$",
        "w_opt_refit_after_projecting_w_lda": r"$\wopt'$ (refit, $\wmd$ projected out)",
    }
    proto_label = {
        "maxpool_raw": "max-pool/raw",
        "lasttok_templated": "last-token/chat",
    }
    for model in df["model"].unique():
        for protocol in df[df["model"] == model]["protocol"].unique():
            sub = df[(df["model"] == model) & (df["protocol"] == protocol)]
            for _, r in sub.iterrows():
                mname = model.split("/")[-1]
                lines.append(
                    f"  {mname} & {proto_label.get(r['protocol'], r['protocol'])} & "
                    f"{cond_label.get(r['condition'], r['condition'])} & "
                    f"{r['auroc_median']:.3f} "
                    f"[{r['auroc_lo']:.3f}, {r['auroc_hi']:.3f}] \\\\"
                )
            lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table}"])
    out_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=["maxpool_raw", "lasttok_templated", "both"],
        default="both",
        help="Which extraction protocol to test. 'both' runs maxpool_raw on "
        "all 12 models and lasttok_templated on the four instruct models.",
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Single model id; defaults to the appropriate set."
    )
    parser.add_argument("--splits-dir", type=Path, default=Path("data/raw/splits"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    splits = g1.load_splits(args.splits_dir)

    all_rows = []
    angle_rows = []

    runs = []
    if args.model:
        protocols = (
            ["maxpool_raw", "lasttok_templated"] if args.protocol == "both" else [args.protocol]
        )
        for p in protocols:
            runs.append((args.model, p))
    else:
        if args.protocol in ("maxpool_raw", "both"):
            for m in INSTRUCT_MODELS:
                runs.append((m, "maxpool_raw"))
        if args.protocol in ("lasttok_templated", "both"):
            for m in INSTRUCT_MODELS:
                runs.append((m, "lasttok_templated"))

    for m, p in runs:
        r = run_protocol(m, splits, args.out_dir, device, p)
        all_rows.extend(r["rows"])
        angle_rows.append(r["angles"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df_a = pd.DataFrame(angle_rows)
    df.to_csv(args.out_dir / "g4_self_projected_refit.csv", index=False)
    df_a.to_csv(args.out_dir / "g4_angles.csv", index=False)
    write_table(df, args.out_dir / "table_g4.tex")
    print("\nG4 done.")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print()
    print(df_a.to_string(index=False))


if __name__ == "__main__":
    main()
