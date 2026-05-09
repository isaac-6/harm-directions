#!/usr/bin/env python3
"""
make_angle_figure.py
--------------------
Strip plot of cross-strategy direction angles, one dot per model.
Colour + shape double-encode family for accessibility. Alternating row
shading aids eye tracking. Legend sits outside the plot area.

Reads:  results/artifacts/<slug>/angles.csv
Writes: figures/fig_angle_strip.pdf, figures/fig_angle_strip.png  (Figure 6 in paper)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paul Tol bright palette + distinct shapes. Colour-blind safe.
FAMILY_STYLE = {
    "Qwen2.5": {"colour": "#4477AA", "marker": "o"},
    "Qwen3.5": {"colour": "#EE6677", "marker": "s"},
    "Llama-3.2": {"colour": "#228833", "marker": "D"},
    "Gemma-3": {"colour": "#CCBB44", "marker": "^"},
}

PAIR_ORDER: list[tuple[str, str, str]] = [
    ("mean_diff", "soft_auc", r"$\mathbf{w}_{\mathrm{LDA}}$ vs $\mathbf{w}_{\mathrm{opt}}$"),
    ("mean_diff", "theta_two_class", r"$\mathbf{w}_{\mathrm{LDA}}$ vs $\theta$ two-class"),
    ("theta_normative", "theta_two_class", r"$\theta$-norm vs $\theta$ two-class"),
    (
        "mean_diff",
        "pc1_normative",
        r"$\mathbf{w}_{\mathrm{LDA}}$ vs $\mathbf{w}_{\mathrm{PC1}}$ (benign)",
    ),
    (
        "pc1_normative",
        "pc1_harmful",
        r"$\mathbf{w}_{\mathrm{PC1}}$ (benign) vs $\mathbf{w}_{\mathrm{PC1}}$ (harmful)",
    ),
]


def model_family(model_id: str) -> str:
    if "Qwen2.5" in model_id:
        return "Qwen2.5"
    if "Qwen3.5" in model_id or "Qwen3-" in model_id:
        return "Qwen3.5"
    if "Llama-3.2" in model_id:
        return "Llama-3.2"
    if "gemma-3" in model_id.lower():
        return "Gemma-3"
    raise ValueError(f"Unknown family: {model_id}")


def load_all_angles(artifacts_dir: Path) -> pd.DataFrame:
    frames = []
    for model_dir in sorted(artifacts_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        p = model_dir / "angles.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f"No angles.csv under {artifacts_dir}")
    return pd.concat(frames, ignore_index=True)


def lookup_angle(df: pd.DataFrame, model: str, a: str, b: str) -> float | None:
    mask = (df["model"] == model) & (
        ((df["direction_a"] == a) & (df["direction_b"] == b))
        | ((df["direction_a"] == b) & (df["direction_b"] == a))
    )
    rows = df.loc[mask, "angle_deg"]
    return float(rows.iloc[0]) if len(rows) > 0 else None


def plot(df: pd.DataFrame, out_path: Path, jitter: float = 0.15) -> None:
    rng = np.random.default_rng(42)
    rows = []
    for idx, (a, b, label) in enumerate(PAIR_ORDER):
        for model in df["model"].unique():
            ang = lookup_angle(df, model, a, b)
            if ang is None:
                continue
            rows.append(
                {
                    "pair_idx": idx,
                    "pair_label": label,
                    "model": model,
                    "family": model_family(model),
                    "angle_deg": ang,
                }
            )
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        raise RuntimeError("No matching angle data for PAIR_ORDER.")

    n_pairs = len(PAIR_ORDER)
    # fig_h = max(3.0, 0.75 * n_pairs + 1.2)
    fig_h = max(2.5, 0.55 * n_pairs + 1.0)
    fig, ax = plt.subplots(figsize=(8.5, fig_h), dpi=150)

    # Alternating row shading (every other row gets a light background)
    for i in range(n_pairs):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#F5F5F5", zorder=0)

    # Reference lines at 0, 45, 90 degrees
    for x in [0, 45, 90]:
        ax.axvline(x, color="gray", linestyle=":", linewidth=0.7, alpha=0.6, zorder=1)

    # Scatter: one call per family for clean legend
    for family, style in FAMILY_STYLE.items():
        fam_data = plot_df[plot_df["family"] == family]
        if fam_data.empty:
            continue
        y_jitter = rng.uniform(-jitter, jitter, len(fam_data))
        ax.scatter(
            fam_data["angle_deg"],
            fam_data["pair_idx"] + y_jitter,
            c=style["colour"],
            marker=style["marker"],
            s=70,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.8,
            label=family,
            zorder=3,
        )

    # Axes
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels([label for _, _, label in PAIR_ORDER])
    ax.set_ylim(n_pairs - 0.5, -0.5)
    ax.set_xlabel("Angle (degrees)")
    ax.set_xlim(-5, 95)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.tick_params(axis="both", length=0)

    # Grid on x only, subtle
    ax.grid(True, axis="x", linestyle="-", linewidth=0.4, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Spines: keep bottom only
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)

    # Legend: outside plot, right side, boxed
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=1.0,
        edgecolor="0.7",
        fontsize=9,
        title="Family",
        title_fontsize=9,
        handletextpad=0.4,
        markerscale=1.0,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}.pdf and .png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("results/artifacts"))
    parser.add_argument("--out-path", type=Path, default=Path("figures/fig_angle_strip"))
    args = parser.parse_args()

    df = load_all_angles(args.artifacts_dir)
    plot(df, args.out_path)


if __name__ == "__main__":
    main()
