#!/usr/bin/env python3
"""
run_g3_arditi_projected_refit.py
---------------------------------
G3 — Refit harm direction in refusal-orthogonal subspace.

At Arditi's selected layer, project out w_arditi (or other directions)
from activations, refit harm direction, evaluate.

Conditions per model:
  1. original_at_arditi_layer:  baseline at L_arditi
  2. original_projected:        original direction on projected activations
  3. arditi_projected_refit:    refit on activations with w_arditi removed

Outputs:
  results/g3_refit.csv
  results/g3_angles.csv
  results/table_g3.tex
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
import run_g1_chat_template as g1
import run_g2_arditi_angle as g2


def project_out_multi(x, ws):
    """Project out multiple unit vectors sequentially via Gram-Schmidt order."""
    out = x.copy()
    for w in ws:
        w_unit = w / (np.linalg.norm(w) + 1e-12)
        out = (
            out - np.outer(out @ w_unit, w_unit) if out.ndim == 2 else out - (out @ w_unit) * w_unit
        )
    return out


def run_model(model_id, splits, out_dir, device):
    slug = model_id.replace("/", "__")
    g2_cache = out_dir / "artifacts_neurips" / slug / "g2" / "g2_artifacts.npz"
    if not g2_cache.exists():
        raise FileNotFoundError(f"Run G2 first; {g2_cache} not found.")

    g2_data = np.load(g2_cache)
    L_arditi = int(g2_data["L_arditi"])
    w_arditi = g2_data["w_arditi"]

    print(f"\n{'=' * 70}\n  G3: {model_id} at L_arditi = {L_arditi}\n{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", trust_remote_code=True)
        .to(device)
        .eval()
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Re-extract maxpool-templated activations at all layers (we only need L_arditi)
    fit_h_t = [g1.apply_chat_template(tokenizer, p) for p in splits["fit_harm"]]
    fit_b_t = [g1.apply_chat_template(tokenizer, p) for p in splits["fit_benign"]]

    eval_harm_prompts, eval_harm_src = g1.flatten_eval(splits["eval_harm"])
    eval_benign_prompts, eval_benign_src = g1.flatten_eval(splits["eval_benign"])
    eval_h_t = [g1.apply_chat_template(tokenizer, p) for p in eval_harm_prompts]
    eval_b_t = [g1.apply_chat_template(tokenizer, p) for p in eval_benign_prompts]

    print("  Extracting maxpool-templated activations...")
    fit_h = g2.extract_maxpool_per_layer(model, tokenizer, fit_h_t, device)[:, L_arditi]
    fit_b = g2.extract_maxpool_per_layer(model, tokenizer, fit_b_t, device)[:, L_arditi]
    eval_h = g2.extract_maxpool_per_layer(model, tokenizer, eval_h_t, device)[:, L_arditi]
    eval_b = g2.extract_maxpool_per_layer(model, tokenizer, eval_b_t, device)[:, L_arditi]

    # Helper to fit + eval
    def fit_and_eval(fit_h_a, fit_b_a, eval_h_a, eval_b_a):
        w_lda = g1.fit_mean_diff(fit_h_a, fit_b_a)
        w_opt = g1.fit_soft_auc(fit_h_a, fit_b_a, init=w_lda)
        results = {}
        for strat, w in [("mean_diff", w_lda), ("soft_auc", w_opt)]:
            s_h = eval_h_a @ w
            s_b = eval_b_a @ w
            scores_arr = np.concatenate([s_b, s_h])
            labels = np.concatenate([np.zeros(len(s_b)), np.ones(len(s_h))])
            sources = np.concatenate([eval_benign_src, eval_harm_src])
            results[strat] = {"w": w, "metrics": g1.bootstrap_metrics(scores_arr, labels, sources)}
        return results

    rows = []
    angles = {}

    # Condition 1: original at L_arditi (no projection)
    cond1 = fit_and_eval(fit_h, fit_b, eval_h, eval_b)
    for strat, r in cond1.items():
        rows.append(
            {
                "model": model_id,
                "condition": "original_at_arditi_layer",
                "strategy": strat,
                "layer": L_arditi,
                **r["metrics"],
            }
        )

    # Condition 2: original direction on projected activations
    eval_h_proj_arditi = project_out_multi(eval_h, [w_arditi])
    eval_b_proj_arditi = project_out_multi(eval_b, [w_arditi])
    for strat, r in cond1.items():
        s_h = eval_h_proj_arditi @ r["w"]
        s_b = eval_b_proj_arditi @ r["w"]
        scores_arr = np.concatenate([s_b, s_h])
        labels = np.concatenate([np.zeros(len(s_b)), np.ones(len(s_h))])
        sources = np.concatenate([eval_benign_src, eval_harm_src])
        m = g1.bootstrap_metrics(scores_arr, labels, sources)
        rows.append(
            {
                "model": model_id,
                "condition": "original_projected",
                "strategy": strat,
                "layer": L_arditi,
                **m,
            }
        )

    # Condition 3: refit on Arditi-projected
    fit_h_proj_arditi = project_out_multi(fit_h, [w_arditi])
    fit_b_proj_arditi = project_out_multi(fit_b, [w_arditi])
    cond3 = fit_and_eval(
        fit_h_proj_arditi, fit_b_proj_arditi, eval_h_proj_arditi, eval_b_proj_arditi
    )
    for strat, r in cond3.items():
        rows.append(
            {
                "model": model_id,
                "condition": "arditi_projected_refit",
                "strategy": strat,
                "layer": L_arditi,
                **r["metrics"],
            }
        )
        angles[f"{strat}_orig_vs_refit_arditi"] = g1.angle_deg(cond1[strat]["w"], r["w"])
        angles[f"{strat}_refit_arditi_vs_arditi"] = g1.angle_deg(
            r["w"], w_arditi
        )  # sanity check, should be ~90 deg

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {"rows": rows, "angles": angles, "model": model_id, "L_arditi": L_arditi}


def write_table(df, out_path):
    lines = [
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{G3: Refit of $\wopt$ in subspaces with various directions "
        r"projected out, at Arditi's selected layer per model. AUROC with "
        r"stratified bootstrap 95\% CIs.}",
        r"\label{tab:g3_refit}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Model & Condition & AUROC [95\% CI] \\",
        r"\midrule",
    ]
    sub = df[df["strategy"] == "soft_auc"]
    cond_label = {
        "original_at_arditi_layer": "original (baseline)",
        "original_projected": "original on projected (G2 result)",
        "arditi_projected_refit": r"refit, $w_{\text{arditi}}$ removed",
    }
    for model in sub["model"].unique():
        for _, r in sub[sub["model"] == model].iterrows():
            mname = model.split("/")[-1]
            lines.append(
                f"  {mname} & {cond_label.get(r['condition'], r['condition'])} & "
                f"{r['auroc_median']:.3f} [{r['auroc_lo']:.3f}, {r['auroc_hi']:.3f}] \\\\"
            )
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table}"])
    out_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/raw/splits"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    splits = g1.load_splits(args.splits_dir)
    models = [args.model] if args.model else g1.INSTRUCT_MODELS

    all_rows = []
    angle_rows = []
    for m in models:
        r = run_model(m, splits, args.out_dir, device)
        all_rows.extend(r["rows"])
        for k, v in r["angles"].items():
            angle_rows.append({"model": m, "pair": k, "angle_deg": v})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df_a = pd.DataFrame(angle_rows)
    df.to_csv(args.out_dir / "g3_refit.csv", index=False)
    df_a.to_csv(args.out_dir / "g3_angles.csv", index=False)
    write_table(df, args.out_dir / "table_g3.tex")
    print("\nG3 done.")
    print(df.to_string())


if __name__ == "__main__":
    main()
