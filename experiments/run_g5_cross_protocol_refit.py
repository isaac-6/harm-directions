#!/usr/bin/env python3
r"""
run_g5_cross_protocol_refit.py
-------------------------------
G5 -- Cross-protocol projection-and-refit at $L_\mathrm{arditi}$.

Test whether different extraction protocols recover genuinely
different concentrated directions of the harm signal, or
aggregate the same underlying direction differently.

Procedure per model:
  1. Recover $\warditi$ and $L_\mathrm{arditi}$ from G2 cache.
  2. At $L_\mathrm{arditi}$, extract activations under three
     protocols (all on chat-templated input to fix templating;
     plus one raw-input protocol as a robustness probe):
       (a) last-token of chat-templated input -- same as Arditi
       (b) max-pool over chat-templated input
       (c) max-pool over raw input (no chat template)
  3. Project $\warditi$ out of activations under each protocol:
     $x' = x - (x \cdot \warditi)\,\warditi$.
  4. Refit $\wopt'$ from scratch on the projected fit set under
     each protocol.
  5. Evaluate $\wopt'$ on projected eval activations.

Predictions, given G2 + G4:
  - (a) refit AUROC collapses to ~0.5 (same protocol as
        $\warditi$; G4 self-projection result applies).
  - (b) refit AUROC retains substantial signal (different
        aggregation; G4 chat-templated showed ~0.78 under
        self-projection, cross-projection should be similar
        or more retentive).
  - (c) refit AUROC retains substantial signal.

Outputs:
  results/g5_cross_protocol_refit.csv
  results/g5_angles.csv
  results/table_g5.tex
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

INSTRUCT_MODELS = g1.INSTRUCT_MODELS


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


# Three protocols, named for the table:
#   (a) "lasttok_templated"   -- same as w_arditi's protocol
#   (b) "maxpool_templated"   -- max-pool, chat template applied
#   (c) "maxpool_raw"         -- max-pool, raw input (main paper protocol)
PROTOCOLS = ["lasttok_templated", "maxpool_templated", "maxpool_raw"]


def extract_for_protocol(model, tokenizer, prompts, device, protocol):
    """Extract activations under the given protocol; return [N, L, D]."""
    if protocol == "lasttok_templated":
        prompts_t = [g1.apply_chat_template(tokenizer, p) for p in prompts]
        return g1.extract_lasttok_per_layer(model, tokenizer, prompts_t, device)
    if protocol == "maxpool_templated":
        prompts_t = [g1.apply_chat_template(tokenizer, p) for p in prompts]
        return g2.extract_maxpool_per_layer(model, tokenizer, prompts_t, device)
    if protocol == "maxpool_raw":
        return extract_maxpool_raw_per_layer(model, tokenizer, prompts, device)
    raise ValueError(f"Unknown protocol: {protocol}")


def run_model(model_id, splits, out_dir, device):
    slug = model_id.replace("/", "__")
    g2_cache = out_dir / "artifacts_neurips" / slug / "g2" / "g2_artifacts.npz"
    if not g2_cache.exists():
        raise FileNotFoundError(f"Run G2 first; {g2_cache} not found.")

    g2_data = np.load(g2_cache)
    L_arditi = int(g2_data["L_arditi"])
    w_arditi = g2_data["w_arditi"]

    print(f"\n{'=' * 70}\n  G5: {model_id} at L_arditi = {L_arditi}\n{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", trust_remote_code=True)
        .to(device)
        .eval()
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Eval set source labels
    eval_harm_prompts, eval_harm_src = g1.flatten_eval(splits["eval_harm"])
    eval_benign_prompts, eval_benign_src = g1.flatten_eval(splits["eval_benign"])

    rows = []
    angle_rows = []

    for protocol in PROTOCOLS:
        print(f"\n  -- Protocol: {protocol} --")

        print("    Extracting fit/eval activations...")
        fit_h = extract_for_protocol(model, tokenizer, splits["fit_harm"], device, protocol)[
            :, L_arditi
        ]
        fit_b = extract_for_protocol(model, tokenizer, splits["fit_benign"], device, protocol)[
            :, L_arditi
        ]
        eval_h = extract_for_protocol(model, tokenizer, eval_harm_prompts, device, protocol)[
            :, L_arditi
        ]
        eval_b = extract_for_protocol(model, tokenizer, eval_benign_prompts, device, protocol)[
            :, L_arditi
        ]

        # Baseline at L_arditi under this protocol (no projection)
        w_lda_p = g1.fit_mean_diff(fit_h, fit_b)
        w_opt_p = g1.fit_soft_auc(fit_h, fit_b, init=w_lda_p)
        s_h = eval_h @ w_opt_p
        s_b = eval_b @ w_opt_p
        m_baseline = evaluate(s_h, s_b, eval_harm_src, eval_benign_src)
        rows.append(
            {
                "model": model_id,
                "protocol": protocol,
                "condition": "baseline_w_opt",
                "layer": L_arditi,
                **m_baseline,
            }
        )

        # Project w_arditi out of activations under this protocol
        fit_h_proj = project_out(fit_h, w_arditi)
        fit_b_proj = project_out(fit_b, w_arditi)
        eval_h_proj = project_out(eval_h, w_arditi)
        eval_b_proj = project_out(eval_b, w_arditi)

        # Diagnostic: how much signal does projecting w_arditi remove?
        # Magnitude of mean-difference before/after projection
        md_orig = float(np.linalg.norm(fit_h.mean(axis=0) - fit_b.mean(axis=0)))
        md_proj = float(np.linalg.norm(fit_h_proj.mean(axis=0) - fit_b_proj.mean(axis=0)))

        # Refit w_opt on projected activations
        # Initialise from the projected mean-difference, falling back to
        # a random orthogonal vector if md_proj is essentially zero.
        if md_proj > 1e-6 * md_orig:
            init = fit_h_proj.mean(axis=0) - fit_b_proj.mean(axis=0)
            init = init / (np.linalg.norm(init) + 1e-12)
        else:
            rng = np.random.default_rng(42)
            init = rng.normal(size=fit_h.shape[1])
            init = init - (init @ w_arditi) * w_arditi
            init = init / np.linalg.norm(init)

        w_opt_refit = g1.fit_soft_auc(fit_h_proj, fit_b_proj, init=init)

        s_h = eval_h_proj @ w_opt_refit
        s_b = eval_b_proj @ w_opt_refit
        m_refit = evaluate(s_h, s_b, eval_harm_src, eval_benign_src)
        rows.append(
            {
                "model": model_id,
                "protocol": protocol,
                "condition": "refit_after_projecting_w_arditi",
                "layer": L_arditi,
                **m_refit,
            }
        )

        angle_rows.append(
            {
                "model": model_id,
                "protocol": protocol,
                "L_arditi": L_arditi,
                "md_orig_norm": md_orig,
                "md_proj_norm": md_proj,
                "md_proj_norm_ratio": md_proj / (md_orig + 1e-12),
                "angle_warditi_vs_baseline_wopt": g1.angle_deg(w_arditi, w_opt_p),
                "angle_warditi_vs_refit_wopt": g1.angle_deg(w_arditi, w_opt_refit),
                "angle_baseline_wopt_vs_refit_wopt": g1.angle_deg(w_opt_p, w_opt_refit),
            }
        )

        print(f"    baseline AUROC = {m_baseline['auroc_median']:.3f}")
        print(f"    refit AUROC    = {m_refit['auroc_median']:.3f}")
        print(f"    md_orig norm   = {md_orig:.4e}")
        print(f"    md_proj norm   = {md_proj:.4e}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {"rows": rows, "angles": angle_rows}


def write_table(df, out_path):
    """Two-column compact LaTeX table: baseline vs refit, per protocol."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{G5: cross-protocol projection-and-refit at "
        r"$L_\mathrm{arditi}$. For each model and extraction protocol, "
        r"we report baseline $\wopt$ AUROC and refit $\wopt'$ AUROC "
        r"after projecting $\warditi$ (extracted at $L_\mathrm{arditi}$ "
        r"under last-token chat-templated extraction) out of activations. "
        r"AUROC with stratified bootstrap 95\% CIs.}",
        r"\label{tab:g5_cross_protocol}",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Model & Protocol & Baseline $\wopt$ & Refit $\wopt'$ "
        r"after $-\warditi$ \\",
        r"\midrule",
    ]
    proto_label = {
        "lasttok_templated": "last-token / chat-templated",
        "maxpool_templated": "max-pool / chat-templated",
        "maxpool_raw": "max-pool / raw",
    }
    for model in df["model"].unique():
        for protocol in PROTOCOLS:
            sub = df[(df["model"] == model) & (df["protocol"] == protocol)]
            if len(sub) == 0:
                continue
            mname = model.split("/")[-1]
            base = sub[sub["condition"] == "baseline_w_opt"].iloc[0]
            refit = sub[sub["condition"] == "refit_after_projecting_w_arditi"].iloc[0]
            base_str = (
                f"{base['auroc_median']:.3f} [{base['auroc_lo']:.3f}, {base['auroc_hi']:.3f}]"
            )
            refit_str = (
                f"{refit['auroc_median']:.3f} [{refit['auroc_lo']:.3f}, {refit['auroc_hi']:.3f}]"
            )
            lines.append(f"  {mname} & {proto_label[protocol]} & {base_str} & {refit_str} \\\\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table}"])
    out_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single model id; defaults to all four instruct models.",
    )
    parser.add_argument("--splits-dir", type=Path, default=Path("data/raw/splits"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    splits = g1.load_splits(args.splits_dir)
    models = [args.model] if args.model else INSTRUCT_MODELS

    all_rows = []
    angle_rows = []
    for m in models:
        r = run_model(m, splits, args.out_dir, device)
        all_rows.extend(r["rows"])
        angle_rows.extend(r["angles"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df_a = pd.DataFrame(angle_rows)
    df.to_csv(args.out_dir / "g5_cross_protocol_refit.csv", index=False)
    df_a.to_csv(args.out_dir / "g5_angles.csv", index=False)
    write_table(df, args.out_dir / "table_g5.tex")
    print("\nG5 done.")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n=== Results ===")
    print(df.to_string(index=False))
    print("\n=== Angles & projection diagnostics ===")
    print(df_a.to_string(index=False))


if __name__ == "__main__":
    main()
