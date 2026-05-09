[![arXiv](https://img.shields.io/badge/arXiv-2604.18901-b31b1b.svg)](https://arxiv.org/abs/2604.18901)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19671546.svg)](https://doi.org/10.5281/zenodo.19671546)
[![Version](https://img.shields.io/github/v/release/isaac-6/harm-directions)](https://github.com/isaac-6/harm-directions/releases)

[![CI](https://github.com/isaac-6/harm-directions/actions/workflows/ci.yml/badge.svg)](https://github.com/isaac-6/harm-directions/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10+-blue)

# harm-directions

Lightweight (one dot product) harmful-prompt detection from LLM residual-stream activations.

> **Paper:** [Harmful Intent as a Geometrically Recoverable Feature of LLM Residual Streams](https://arxiv.org/abs/2604.18901) — arXiv:2604.18901

![AUROC vs TPR across 12 models](docs/figures/fig_auroc_vs_tpr.png)
*Mean-difference (left) and Soft-AUC-optimised (right) directions, evaluated
on held-out data across 12 models (4 families x 3 alignment variants).
Mean AUROC 0.975 and 0.982; TPR@1%FPR 0.706 and 0.797. Whiskers: stratified
bootstrap 95% CIs (1,000 resamples).*

## What this is

Detecting harmful prompts matters both for deployment safety and for understanding how models represent these concepts internally.
We show how a supervised linear probe over LLM residual-stream activations detects
harmful user prompts. The probe is stable across instruction tuning and abliteration, consistent with an account in which models acquire a representation of harmful intent as part of general language understanding, independently of how alignment shapes their response behaviour.

For the full analysis across 12 models (plus 9 extra for scaling), see the accompanying manuscript.

## Installation

Requires Python 3.10+. Tested on Ubuntu with CUDA 12, CPU-only supported
for the detection functions (extraction requires a GPU for reasonable runtime).

```bash
# If you have a GPU, install torch with matching CUDA version first:
pip install torch --index-url https://download.pytorch.org/whl/cu128  # or cu130

# Core library only (numpy, sklearn)
pip install -e .

# Add activation extraction (torch, transformers)
pip install -e ".[extract]"

# Everything needed to reproduce the paper
pip install -e ".[reproduce]"
```

## Usage

The following assumes you've installed the package with 
`pip install -e ".[extract]"` (or `.[reproduce]"` for the full pipeline).
Run `make help` to see all available targets.

### As a library

```python
from harm_directions import extract_activations, fit_direction, score

harm_acts = extract_activations(model, tokenizer, harmful_prompts, layer=22)
safe_acts = extract_activations(model, tokenizer, safe_prompts, layer=22)
w = fit_direction(harm_acts, safe_acts, method="mean_diff")

new_acts = extract_activations(model, tokenizer, ["How do I bake a cake"], layer=22)
scores = score(new_acts, w)
```

### As a CLI

```bash
# Fit a direction (caches to ./data/fitted/ by default)
harm-directions-detect --model Qwen/Qwen2.5-0.5B-Instruct --fit

# Score a single prompt
harm-directions-detect --model Qwen/Qwen2.5-0.5B-Instruct --prompt "How do I bake a cake"

# Score many prompts from a file
harm-directions-detect --model Qwen/Qwen2.5-0.5B-Instruct --input prompts.txt
```

## Reproducing the paper

```bash
# 1. Download datasets (AdvBench, HarmBench, JailbreakBench, XSTest, Alpaca)
python scripts/download_datasets.py

# 2. Full evaluation across all 12 models (~36 min on an RTX 3070)
harm-directions-reproduce --all

# Or one model at a time
harm-directions-reproduce --model Qwen/Qwen2.5-0.5B-Instruct
```

Results are written to `./results/` as per-model CSVs and an aggregate `summary.csv`.

## Geometry experiments (G1–G5)

The `experiments/` directory contains five scripts producing the
geometry analyses from the paper (chat-template ablation, comparison
to Arditi's refusal direction, and projection-and-refit tests).

```bash
# Run all five sequentially
make experiments

# Or one at a time (Make handles dependencies)
make g3   # automatically runs g1, g2 first
```

| Script | What it tests | Depends on |
|--------|--------------|-----------|
| G1 | Harm direction recovery under chat templating | — |
| G2 | Angle to Arditi's refusal direction; cross-protocol evaluation | — |
| G3 | Refit harm direction with Arditi direction projected out | G2 cache |
| G4 | Refit in self-orthogonal subspace (single-direction sufficiency test) | — |
| G5 | Cross-protocol projection-and-refit at Arditi's layer | G2 cache |

Outputs: per-script CSVs and LaTeX tables in `results/`. G2 additionally
caches fitted directions to `results/artifacts_neurips/` for G3 and G5.

Approximate runtime on an RTX 3070 mobile: ~25 min for the full sweep across
the four instruction-tuned models.

## Figures and supplementary tables

The `experiments/` directory also contains the scripts that produce every
figure and supplementary table in the paper.  They read from
`results/artifacts/` (generated by `generate_figure_artifacts.py`) and write
PDFs/PNGs to `figures/` and LaTeX to `results/`.

```bash
# Full annex pipeline: artifacts → baseline comparison → all figures/tables
make annex

# Or step by step
make figure_artifacts       # extract per-model scores, layer sweeps, directions
make baseline_comparison    # evaluate external safety-classifier baselines
make make_heatmaps          # regenerate a single figure
```

| Script | Output |
|--------|--------|
| `generate_figure_artifacts.py` | `results/artifacts/` — per-model scores, layer sweeps, directions |
| `baseline_comparison.py` | Llama Guard 3 / ShieldGemma / WildGuard / Latent Guard comparison |
| `make_auroc_vs_tpr_figure.py` | `fig_auroc_vs_tpr.{pdf,png}` |
| `make_heatmaps.py` | `fig_auroc_heatmap.{pdf,png}`, `fig_tpr_heatmap.{pdf,png}` |
| `make_layer_profile_figure.py` | `fig_layer_profiles.{pdf,png}` |
| `make_score_distribution_figure.py` | `fig_score_dist.{pdf,png}` |
| `make_angle_figure.py` | `fig_angle_strip.{pdf,png}` |
| `make_full_results_tables.py` | Full per-model AUROC/TPR LaTeX tables |
| `make_full_transfer_table.py` | Cross-variant transfer table |
| `make_ood_tables.py` | Per-source OOD breakdown tables |
| `make_scale_table.py` | Qwen3.5 scaling extension table |

Approximate runtime for `make annex` on an RTX 3070 mobile: ~45 min
(dominated by `generate_figure_artifacts.py` re-running extraction).

## Models evaluated

12 models across 4 families x 3 alignment variants, all 0.5–1.3B parameters:

| Family | Size | Layers | Hidden dim |
|--------|------|--------|------------|
| Qwen2.5 | 0.5B | 24 | 896 |
| Qwen3.5 | 0.8B | 24 | 1024 |
| Llama-3.2 | 1B | 16 | 2048 |
| Gemma-3 | 1B | 26 | 1152 |

For each family: base (pretrained), instruction-tuned, and abliterated
(refusal-direction-ablated) variants from HuggingFace. A Qwen3.5 scaling extension at 2B, 4B, and 9B is also reported in the paper, for a total of 21 models.

## Data splits

The evaluation uses three sample-disjoint sets, all drawn with `seed=42`:

- **Fit set** (direction fitting): 100 AdvBench harmful + 100 Alpaca normative.
- **Validation set** (layer selection only): 50 + 50.
- **Evaluation set**: held-out AdvBench (370) + HarmBench (200) + JailbreakBench (100)
  vs Alpaca (500) + XSTest hard-benign (250).

The operating layer is selected once per model using mean-difference validation
AUROC, then shared across all strategies. The evaluation set never contributes
to any fitting or selection decision.

## Repository structure

```
harm-directions/
├── src/harm_directions/
│   ├── directions.py               # Direction strategies (LDA, Soft-AUC, PC1, θ)
│   ├── evaluation.py               # AUROC, TPR@FPR, layer selection
│   ├── extraction.py               # Residual-stream extraction with forward hooks
│   └── cli/
│       ├── detect.py               # CLI: fit direction, score prompts
│       └── reproduce.py            # Full paper reproduction pipeline
├── experiments/
│   ├── run_g1_chat_template.py     # G1: chat-template ablation
│   ├── run_g2_arditi_angle.py      # G2: angle to Arditi's refusal direction
│   ├── run_g3_arditi_projected_refit.py  # G3: refit in refusal-orthogonal subspace
│   ├── run_g4_self_projected_refit.py    # G4: self-orthogonal sufficiency test
│   ├── run_g5_cross_protocol_refit.py    # G5: cross-protocol projection-and-refit
│   ├── generate_figure_artifacts.py      # Extract per-model data for figures
│   ├── baseline_comparison.py            # External classifier comparison
│   └── make_*.py                         # Figure and table generation scripts
├── scripts/
│   ├── download_datasets.py        # Dataset download + split composition
│   └── cross_variant_transfer.py   # Cross-variant direction transfer analysis
├── tests/                          # Unit tests (numpy only, no GPU required)
├── Makefile                        # `make experiments`, `make annex`, `make check`, etc.
└── pyproject.toml
```

## Citation

If you use this code or build on this work, please cite both the paper and the software archive:

```bibtex
@misc{llorentesaguer2026harmfulintentgeometricallyrecoverable,
      title={Harmful Intent as a Geometrically Recoverable Feature of LLM Residual Streams}, 
      author={Isaac Llorente-Saguer},
      year={2026},
      eprint={2604.18901},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.18901}
}

@software{llorente_saguer_harm_directions_2026,
  author  = {Llorente-Saguer, Isaac},
  title   = {harm-directions: Lightweight harmful-prompt detection from LLM residual-stream activations},
  year    = {2026},
  version = {v1.0.0},
  doi     = {10.5281/zenodo.19671546},
  url     = {https://github.com/isaac-6/harm-directions}
}
```

## License

MIT. See [LICENSE](LICENSE).