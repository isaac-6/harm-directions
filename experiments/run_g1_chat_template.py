#!/usr/bin/env python3
"""
run_g1_chat_template.py
-----------------------
G1 — Chat template ablation.

Tests whether the harm direction is recoverable from chat-templated
activations using last-token pooling.

Three conditions per model:
  1. Raw prompt, last-token, at the validation-selected layer.
  2. Chat-templated prompt, last-token, at the raw-condition's layer.
  3. Chat-templated prompt, last-token, at the chat-best layer.

Outputs:
  results/g1_chat_template.csv
  results/g1_angles.csv
  results/table_g1.tex
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INSTRUCT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3.5-0.8B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-3-1b-it",
]

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def apply_chat_template(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def extract_lasttok_per_layer(
    model,
    tokenizer,
    texts: list[str],
    device: str,
) -> np.ndarray:
    """Extract last-token hidden state at every layer for each text.

    Returns array of shape (n_texts, n_layers, d_model).
    """
    out = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        # outputs.hidden_states is tuple of (n_layers+1) tensors of shape (1, seq, d)
        # Take last-token from each layer (skip embedding layer index 0)
        layer_acts = [h[0, -1, :].float().cpu().numpy() for h in outputs.hidden_states[1:]]
        out.append(np.stack(layer_acts, axis=0))
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# Direction fitting and scoring
# ---------------------------------------------------------------------------


def fit_mean_diff(harm: np.ndarray, benign: np.ndarray) -> np.ndarray:
    w = harm.mean(axis=0) - benign.mean(axis=0)
    return w / (np.linalg.norm(w) + 1e-12)


def fit_soft_auc(
    harm: np.ndarray,
    benign: np.ndarray,
    n_steps: int = 500,
    lr: float = 0.1,
    init: np.ndarray | None = None,
) -> np.ndarray:
    """Soft-AUC optimisation; warm-start from mean-diff if init not given."""
    w = fit_mean_diff(harm, benign) if init is None else init / (np.linalg.norm(init) + 1e-12)

    w_t = torch.tensor(w, dtype=torch.float32, requires_grad=True)
    h_t = torch.tensor(harm, dtype=torch.float32)
    b_t = torch.tensor(benign, dtype=torch.float32)
    optim = torch.optim.Adam([w_t], lr=lr)

    for _ in range(n_steps):
        optim.zero_grad()
        h_scores = h_t @ w_t
        b_scores = b_t @ w_t
        diffs = h_scores.unsqueeze(1) - b_scores.unsqueeze(0)
        loss = -torch.sigmoid(diffs).mean()
        loss.backward()
        optim.step()
        with torch.no_grad():
            w_t /= w_t.norm() + 1e-12

    return w_t.detach().cpu().numpy()


def score(acts: np.ndarray, w: np.ndarray) -> np.ndarray:
    return acts @ w


def metrics_with_signflip(scores: np.ndarray, labels: np.ndarray) -> dict:
    auroc = float(roc_auc_score(labels, scores))
    if auroc < 0.5:
        scores = -scores
        auroc = 1 - auroc
    fpr, tpr, _ = roc_curve(labels, scores)
    tpr_1 = float(np.interp(0.01, fpr, tpr))
    return {"auroc": auroc, "tpr_1pct_fpr": tpr_1}


def bootstrap_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Stratified bootstrap by source for AUROC and TPR@1%FPR."""
    rng = np.random.default_rng(seed)
    src_unique = np.unique(sources)
    src_idx = {s: np.where(sources == s)[0] for s in src_unique}

    auroc_vals, tpr_vals = [], []
    for _ in range(n_boot):
        idx = np.concatenate(
            [rng.choice(src_idx[s], len(src_idx[s]), replace=True) for s in src_unique]
        )
        m = metrics_with_signflip(scores[idx], labels[idx])
        auroc_vals.append(m["auroc"])
        tpr_vals.append(m["tpr_1pct_fpr"])

    return {
        "auroc_median": float(np.median(auroc_vals)),
        "auroc_lo": float(np.percentile(auroc_vals, 2.5)),
        "auroc_hi": float(np.percentile(auroc_vals, 97.5)),
        "tpr_median": float(np.median(tpr_vals)),
        "tpr_lo": float(np.percentile(tpr_vals, 2.5)),
        "tpr_hi": float(np.percentile(tpr_vals, 97.5)),
    }


# ---------------------------------------------------------------------------
# Layer selection
# ---------------------------------------------------------------------------


def select_best_layer(
    fit_harm: np.ndarray,
    fit_benign: np.ndarray,
    val_harm: np.ndarray,
    val_benign: np.ndarray,
) -> int:
    """Pick layer with highest validation AUROC for mean-diff direction."""
    n_layers = fit_harm.shape[1]
    best_layer, best_auroc = 0, 0.0
    for L in range(n_layers):
        w = fit_mean_diff(fit_harm[:, L, :], fit_benign[:, L, :])
        s_h = val_harm[:, L, :] @ w
        s_b = val_benign[:, L, :] @ w
        labels = np.concatenate([np.zeros(len(s_b)), np.ones(len(s_h))])
        scores_arr = np.concatenate([s_b, s_h])
        m = metrics_with_signflip(scores_arr, labels)
        if m["auroc"] > best_auroc:
            best_auroc = m["auroc"]
            best_layer = L
    return best_layer


# ---------------------------------------------------------------------------
# Per-model pipeline
# ---------------------------------------------------------------------------


def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    cos = float(np.clip(u @ v / ((np.linalg.norm(u) + 1e-12) * (np.linalg.norm(v) + 1e-12)), -1, 1))
    return float(np.degrees(np.arccos(abs(cos))))  # unsigned


def load_prompts(path: Path) -> list[str]:
    return [line.strip() for line in open(path) if line.strip()]


def load_splits(splits_dir: Path) -> dict:
    return {
        "fit_harm": load_prompts(splits_dir / "fit_harmful_advbench.txt"),
        "fit_benign": load_prompts(splits_dir / "fit_normative_alpaca.txt"),
        "val_harm": load_prompts(splits_dir / "val_harmful_advbench.txt"),
        "val_benign": load_prompts(splits_dir / "val_normative_alpaca.txt"),
        "eval_harm": {
            p.stem.replace("eval_harmful_", ""): load_prompts(p)
            for p in sorted(splits_dir.glob("eval_harmful_*.txt"))
        },
        "eval_benign": {
            p.stem.replace("eval_benign_", ""): load_prompts(p)
            for p in sorted(splits_dir.glob("eval_benign_*.txt"))
        },
    }


def flatten_eval(eval_dict: dict) -> tuple[list[str], np.ndarray]:
    prompts, sources = [], []
    for name, ps in eval_dict.items():
        prompts.extend(ps)
        sources.extend([name] * len(ps))
    return prompts, np.array(sources)


def run_model(model_id: str, splits: dict, out_dir: Path, device: str) -> dict:
    slug = model_id.replace("/", "__")
    cache_dir = out_dir / "artifacts_neurips" / slug / "g1"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}\n  G1: {model_id}\n{'=' * 70}")

    print("  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", trust_remote_code=True)
        .to(device)
        .eval()
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_harm_prompts, eval_harm_src = flatten_eval(splits["eval_harm"])
    eval_benign_prompts, eval_benign_src = flatten_eval(splits["eval_benign"])

    rows = []
    angles = {}

    # ------------------- Raw, last-token -------------------
    print("  Extracting raw last-token activations...")
    raw_fit_h = extract_lasttok_per_layer(model, tokenizer, splits["fit_harm"], device)
    raw_fit_b = extract_lasttok_per_layer(model, tokenizer, splits["fit_benign"], device)
    raw_val_h = extract_lasttok_per_layer(model, tokenizer, splits["val_harm"], device)
    raw_val_b = extract_lasttok_per_layer(model, tokenizer, splits["val_benign"], device)
    raw_eval_h = extract_lasttok_per_layer(model, tokenizer, eval_harm_prompts, device)
    raw_eval_b = extract_lasttok_per_layer(model, tokenizer, eval_benign_prompts, device)

    raw_layer = select_best_layer(raw_fit_h, raw_fit_b, raw_val_h, raw_val_b)
    print(f"  Raw best layer: {raw_layer}")

    # Fit at raw best layer
    w_lda_raw = fit_mean_diff(raw_fit_h[:, raw_layer], raw_fit_b[:, raw_layer])
    w_opt_raw = fit_soft_auc(raw_fit_h[:, raw_layer], raw_fit_b[:, raw_layer], init=w_lda_raw)

    for strat_name, w in [("mean_diff", w_lda_raw), ("soft_auc", w_opt_raw)]:
        s_h = score(raw_eval_h[:, raw_layer], w)
        s_b = score(raw_eval_b[:, raw_layer], w)
        scores_arr = np.concatenate([s_b, s_h])
        labels = np.concatenate([np.zeros(len(s_b)), np.ones(len(s_h))])
        sources = np.concatenate([eval_benign_src, eval_harm_src])
        m = bootstrap_metrics(scores_arr, labels, sources)
        rows.append(
            {
                "model": model_id,
                "condition": "raw",
                "layer": raw_layer,
                "strategy": strat_name,
                **m,
            }
        )

    # ------------------- Chat-templated, last-token -------------------
    print("  Extracting chat-templated last-token activations...")
    fit_h_t = [apply_chat_template(tokenizer, p) for p in splits["fit_harm"]]
    fit_b_t = [apply_chat_template(tokenizer, p) for p in splits["fit_benign"]]
    val_h_t = [apply_chat_template(tokenizer, p) for p in splits["val_harm"]]
    val_b_t = [apply_chat_template(tokenizer, p) for p in splits["val_benign"]]
    eval_h_t = [apply_chat_template(tokenizer, p) for p in eval_harm_prompts]
    eval_b_t = [apply_chat_template(tokenizer, p) for p in eval_benign_prompts]

    chat_fit_h = extract_lasttok_per_layer(model, tokenizer, fit_h_t, device)
    chat_fit_b = extract_lasttok_per_layer(model, tokenizer, fit_b_t, device)
    chat_val_h = extract_lasttok_per_layer(model, tokenizer, val_h_t, device)
    chat_val_b = extract_lasttok_per_layer(model, tokenizer, val_b_t, device)
    chat_eval_h = extract_lasttok_per_layer(model, tokenizer, eval_h_t, device)
    chat_eval_b = extract_lasttok_per_layer(model, tokenizer, eval_b_t, device)

    chat_best_layer = select_best_layer(chat_fit_h, chat_fit_b, chat_val_h, chat_val_b)
    print(f"  Chat best layer: {chat_best_layer}")

    for cond_name, layer in [("chat_same_layer", raw_layer), ("chat_best_layer", chat_best_layer)]:
        w_lda = fit_mean_diff(chat_fit_h[:, layer], chat_fit_b[:, layer])
        w_opt = fit_soft_auc(chat_fit_h[:, layer], chat_fit_b[:, layer], init=w_lda)

        for strat_name, w in [("mean_diff", w_lda), ("soft_auc", w_opt)]:
            s_h = score(chat_eval_h[:, layer], w)
            s_b = score(chat_eval_b[:, layer], w)
            scores_arr = np.concatenate([s_b, s_h])
            labels = np.concatenate([np.zeros(len(s_b)), np.ones(len(s_h))])
            sources = np.concatenate([eval_benign_src, eval_harm_src])
            m = bootstrap_metrics(scores_arr, labels, sources)
            rows.append(
                {
                    "model": model_id,
                    "condition": cond_name,
                    "layer": layer,
                    "strategy": strat_name,
                    **m,
                }
            )

    # Angles at raw_layer (same residual space)
    w_lda_chat_same = fit_mean_diff(chat_fit_h[:, raw_layer], chat_fit_b[:, raw_layer])
    w_opt_chat_same = fit_soft_auc(
        chat_fit_h[:, raw_layer], chat_fit_b[:, raw_layer], init=w_lda_chat_same
    )
    angles["mean_diff"] = angle_deg(w_lda_raw, w_lda_chat_same)
    angles["soft_auc"] = angle_deg(w_opt_raw, w_opt_chat_same)

    # Save activations cache for reuse
    np.savez(
        cache_dir / "activations.npz",
        raw_layer=np.array(raw_layer),
        chat_best_layer=np.array(chat_best_layer),
        w_lda_raw=w_lda_raw,
        w_opt_raw=w_opt_raw,
        w_lda_chat_same=w_lda_chat_same,
        w_opt_chat_same=w_opt_chat_same,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "rows": rows,
        "angles": angles,
        "raw_layer": raw_layer,
        "chat_best_layer": chat_best_layer,
    }


def write_latex_table(df: pd.DataFrame, out_path: Path) -> None:
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(
        r"\caption{G1: Harm direction recovery under chat templating, "
        r"using last-token pooling. AUROC and TPR@1\%FPR with stratified "
        r"bootstrap 95\% CIs.}"
    )
    lines.append(r"\label{tab:g1_chat}")
    lines.append(r"\begin{tabular}{llrll}")
    lines.append(r"\toprule")
    lines.append(r"Model & Condition & Layer & AUROC [95\% CI] & TPR@1\%FPR [95\% CI] \\")
    lines.append(r"\midrule")

    cond_label = {
        "raw": "raw",
        "chat_same_layer": "chat (raw layer)",
        "chat_best_layer": "chat (best layer)",
    }
    for model in df["model"].unique():
        sub = df[(df["model"] == model) & (df["strategy"] == "soft_auc")]
        for _, r in sub.iterrows():
            mname = model.split("/")[-1]
            lines.append(
                f"  {mname} & {cond_label[r['condition']]} & {int(r['layer'])} & "
                f"{r['auroc_median']:.3f} [{r['auroc_lo']:.3f}, {r['auroc_hi']:.3f}] & "
                f"{r['tpr_median']:.3f} [{r['tpr_lo']:.3f}, {r['tpr_hi']:.3f}] \\\\"
            )
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default=None, help="Run only this model")
    parser.add_argument("--splits-dir", type=Path, default=Path("data/raw/splits"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_splits(args.splits_dir)
    models = [args.model] if args.model else INSTRUCT_MODELS

    all_rows = []
    angle_rows = []
    for m in models:
        result = run_model(m, splits, args.out_dir, device)
        all_rows.extend(result["rows"])
        for strat, ang in result["angles"].items():
            angle_rows.append(
                {
                    "model": m,
                    "strategy": strat,
                    "layer": result["raw_layer"],
                    "angle_raw_vs_chat_same_deg": ang,
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(args.out_dir / "g1_chat_template.csv", index=False)
    pd.DataFrame(angle_rows).to_csv(args.out_dir / "g1_angles.csv", index=False)
    write_latex_table(df, args.out_dir / "table_g1.tex")
    print("\nWrote g1_chat_template.csv, g1_angles.csv, table_g1.tex")
    print(df.to_string())


if __name__ == "__main__":
    main()
