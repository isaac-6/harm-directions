"""
run_g2_arditi_angle.py
----------------------
G2 -- Geometric relationship between our harm direction and Arditi's
refusal direction on instruction-tuned models.

Two extraction protocols:
  - Ours:    max-pool over chat-templated input.
  - Arditi:  last-token of chat-templated input.

Each protocol independently selects its best layer by validation
AUROC. The script then computes:

(a) Cross-evaluation matrix: each direction (w_opt under our
    protocol, w_arditi under Arditi's protocol) applied to the
    other protocol's evaluation activations at its own
    validation-selected layer. Four cells per model.

(b) Symmetric angle measurement: angles between w_opt_ours and
    w_arditi computed at both L_ours and L_arditi. Where the two
    protocols select the same layer (Qwen3.5, Gemma-3) the two
    measurements collapse to one. Where layers differ (Qwen2.5,
    Llama-3.2) we report both, eliminating the "angle depends on
    where it is computed" reviewer concern.

Outputs:
  results/g2_arditi.csv          (cross-evaluation, 4 cells per model)
  results/g2_angles.csv          (per-model angles at both layers)
  results/table_g2_cross.tex     (LaTeX cross-evaluation table)

Cached artifacts (npz per model) are written to
results/artifacts_neurips/<slug>/g2/ for downstream reuse.
"""

from __future__ import annotations

import argparse
import gc

# Reuse helpers from G1
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
import run_g1_chat_template as g1

INSTRUCT_MODELS = g1.INSTRUCT_MODELS


@torch.no_grad()
def extract_maxpool_per_layer(model, tokenizer, texts, device):
    """Max-pool over all tokens, per layer."""
    out = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        layer_acts = [
            h[0].float().max(dim=0).values.cpu().numpy() for h in outputs.hidden_states[1:]
        ]
        out.append(np.stack(layer_acts, axis=0))
    return np.stack(out, axis=0)


def angle_deg(u, v):
    return g1.angle_deg(u, v)


def evaluate(scores_h, scores_b, eval_harm_src, eval_benign_src):
    scores_arr = np.concatenate([scores_b, scores_h])
    labels = np.concatenate([np.zeros(len(scores_b)), np.ones(len(scores_h))])
    sources = np.concatenate([eval_benign_src, eval_harm_src])
    return g1.bootstrap_metrics(scores_arr, labels, sources)


def run_model(model_id, splits, out_dir, device):
    slug = model_id.replace("/", "__")
    cache_dir = out_dir / "artifacts_neurips" / slug / "g2"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}\n  G2: {model_id}\n{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", trust_remote_code=True)
        .to(device)
        .eval()
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Templated prompts for fit, val, eval
    fit_h_t = [g1.apply_chat_template(tokenizer, p) for p in splits["fit_harm"]]
    fit_b_t = [g1.apply_chat_template(tokenizer, p) for p in splits["fit_benign"]]
    val_h_t = [g1.apply_chat_template(tokenizer, p) for p in splits["val_harm"]]
    val_b_t = [g1.apply_chat_template(tokenizer, p) for p in splits["val_benign"]]

    eval_harm_prompts, eval_harm_src = g1.flatten_eval(splits["eval_harm"])
    eval_benign_prompts, eval_benign_src = g1.flatten_eval(splits["eval_benign"])
    eval_h_t = [g1.apply_chat_template(tokenizer, p) for p in eval_harm_prompts]
    eval_b_t = [g1.apply_chat_template(tokenizer, p) for p in eval_benign_prompts]

    # Per-protocol per-layer extractions
    print("  Extracting (ours = maxpool-templated)...")
    ours_fit_h = extract_maxpool_per_layer(model, tokenizer, fit_h_t, device)
    ours_fit_b = extract_maxpool_per_layer(model, tokenizer, fit_b_t, device)
    ours_val_h = extract_maxpool_per_layer(model, tokenizer, val_h_t, device)
    ours_val_b = extract_maxpool_per_layer(model, tokenizer, val_b_t, device)
    ours_eval_h = extract_maxpool_per_layer(model, tokenizer, eval_h_t, device)
    ours_eval_b = extract_maxpool_per_layer(model, tokenizer, eval_b_t, device)

    print("  Extracting (Arditi = lasttok-templated)...")
    arditi_fit_h = g1.extract_lasttok_per_layer(model, tokenizer, fit_h_t, device)
    arditi_fit_b = g1.extract_lasttok_per_layer(model, tokenizer, fit_b_t, device)
    arditi_val_h = g1.extract_lasttok_per_layer(model, tokenizer, val_h_t, device)
    arditi_val_b = g1.extract_lasttok_per_layer(model, tokenizer, val_b_t, device)
    arditi_eval_h = g1.extract_lasttok_per_layer(model, tokenizer, eval_h_t, device)
    arditi_eval_b = g1.extract_lasttok_per_layer(model, tokenizer, eval_b_t, device)

    # Layer selection per protocol
    L_ours = g1.select_best_layer(ours_fit_h, ours_fit_b, ours_val_h, ours_val_b)
    L_arditi = g1.select_best_layer(arditi_fit_h, arditi_fit_b, arditi_val_h, arditi_val_b)
    print(f"  L_ours = {L_ours}, L_arditi = {L_arditi}")

    # Fit each direction at its protocol's own best layer
    w_lda_ours = g1.fit_mean_diff(ours_fit_h[:, L_ours], ours_fit_b[:, L_ours])
    w_opt_ours = g1.fit_soft_auc(
        ours_fit_h[:, L_ours],
        ours_fit_b[:, L_ours],
        init=w_lda_ours,
    )
    w_arditi = g1.fit_mean_diff(arditi_fit_h[:, L_arditi], arditi_fit_b[:, L_arditi])

    # Cross-evaluation matrix (same as before): 4 cells.
    cross_rows = []
    # Cell 1: ours direction x ours protocol at L_ours
    for strat, w in [("mean_diff", w_lda_ours), ("soft_auc", w_opt_ours)]:
        s_h = ours_eval_h[:, L_ours] @ w
        s_b = ours_eval_b[:, L_ours] @ w
        m = evaluate(s_h, s_b, eval_harm_src, eval_benign_src)
        cross_rows.append(
            {
                "model": model_id,
                "direction": f"ours_{strat}",
                "eval_protocol": "ours",
                "layer": L_ours,
                **m,
            }
        )

    # Cell 2: arditi x arditi at L_arditi
    s_h = arditi_eval_h[:, L_arditi] @ w_arditi
    s_b = arditi_eval_b[:, L_arditi] @ w_arditi
    m = evaluate(s_h, s_b, eval_harm_src, eval_benign_src)
    cross_rows.append(
        {
            "model": model_id,
            "direction": "arditi",
            "eval_protocol": "arditi",
            "layer": L_arditi,
            **m,
        }
    )

    # Cell 3: ours direction x Arditi protocol (at L_ours)
    for strat, w in [("mean_diff", w_lda_ours), ("soft_auc", w_opt_ours)]:
        s_h = arditi_eval_h[:, L_ours] @ w
        s_b = arditi_eval_b[:, L_ours] @ w
        m = evaluate(s_h, s_b, eval_harm_src, eval_benign_src)
        cross_rows.append(
            {
                "model": model_id,
                "direction": f"ours_{strat}",
                "eval_protocol": "arditi",
                "layer": L_ours,
                **m,
            }
        )

    # Cell 4: arditi x ours protocol (at L_arditi)
    s_h = ours_eval_h[:, L_arditi] @ w_arditi
    s_b = ours_eval_b[:, L_arditi] @ w_arditi
    m = evaluate(s_h, s_b, eval_harm_src, eval_benign_src)
    cross_rows.append(
        {
            "model": model_id,
            "direction": "arditi",
            "eval_protocol": "ours",
            "layer": L_arditi,
            **m,
        }
    )

    # ---- Symmetric angle measurement ----
    # At L_ours: w_arditi extracted at L_ours (transport into our preferred layer)
    w_arditi_at_L_ours = g1.fit_mean_diff(arditi_fit_h[:, L_ours], arditi_fit_b[:, L_ours])

    # At L_arditi: w_opt_ours extracted at L_arditi (our direction at Arditi's layer)
    w_lda_ours_at_L_arditi = g1.fit_mean_diff(ours_fit_h[:, L_arditi], ours_fit_b[:, L_arditi])
    w_opt_ours_at_L_arditi = g1.fit_soft_auc(
        ours_fit_h[:, L_arditi],
        ours_fit_b[:, L_arditi],
        init=w_lda_ours_at_L_arditi,
    )

    angles_row = {
        "model": model_id,
        "L_ours": L_ours,
        "L_arditi": L_arditi,
        "shared_layer": int(L_ours == L_arditi),
        "angle_lda_ours_vs_arditi_at_L_ours": angle_deg(w_lda_ours, w_arditi_at_L_ours),
        "angle_opt_ours_vs_arditi_at_L_ours": angle_deg(w_opt_ours, w_arditi_at_L_ours),
        "angle_lda_ours_vs_arditi_at_L_arditi": angle_deg(w_lda_ours_at_L_arditi, w_arditi),
        "angle_opt_ours_vs_arditi_at_L_arditi": angle_deg(w_opt_ours_at_L_arditi, w_arditi),
    }

    # Cache fitted directions for downstream reuse if needed
    np.savez(
        cache_dir / "g2_artifacts.npz",
        L_ours=np.array(L_ours),
        L_arditi=np.array(L_arditi),
        w_lda_ours=w_lda_ours,
        w_opt_ours=w_opt_ours,
        w_arditi=w_arditi,
        w_arditi_at_L_ours=w_arditi_at_L_ours,
        w_lda_ours_at_L_arditi=w_lda_ours_at_L_arditi,
        w_opt_ours_at_L_arditi=w_opt_ours_at_L_arditi,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {"cross": cross_rows, "angles": angles_row}


def write_cross_table(df, out_path):
    lines = [
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{G2: Cross-evaluation of our harm direction "
        r"($\wopt$) and Arditi's refusal direction. Each cell gives "
        r"AUROC with stratified bootstrap 95\% CIs.}",
        r"\label{tab:g2_cross}",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Model & Direction $\to$ Eval & Layer & AUROC [95\% CI] \\",
        r"\midrule",
    ]
    sub = df[df["direction"].isin(["ours_soft_auc", "arditi"])]
    for model in sub["model"].unique():
        for _, r in sub[sub["model"] == model].iterrows():
            mname = model.split("/")[-1]
            d_label = "ours" if "ours" in r["direction"] else "arditi"
            lines.append(
                f"  {mname} & {d_label} $\\to$ {r['eval_protocol']} & "
                f"{int(r['layer'])} & "
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
        "--model", type=str, default=None, help="Single model id; defaults to all four."
    )
    parser.add_argument("--splits-dir", type=Path, default=Path("data/raw/splits"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    splits = g1.load_splits(args.splits_dir)
    models = [args.model] if args.model else INSTRUCT_MODELS

    cross_rows, angle_rows = [], []
    for m in models:
        r = run_model(m, splits, args.out_dir, device)
        cross_rows.extend(r["cross"])
        angle_rows.append(r["angles"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df_c = pd.DataFrame(cross_rows)
    df_a = pd.DataFrame(angle_rows)
    df_c.to_csv(args.out_dir / "g2_arditi.csv", index=False)
    df_a.to_csv(args.out_dir / "g2_angles.csv", index=False)
    write_cross_table(df_c, args.out_dir / "table_g2_cross.tex")

    print("\nG2 done.")
    print("\n=== Angles (degrees) ===")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df_a.to_string(index=False))


if __name__ == "__main__":
    main()
