# latent-biopsy-supervised

Supervised harmful prompt detection via linear discriminant geometry in LLM residual streams.

This work extends [LatentBiopsy](https://github.com/isaac-6/geometric-latent-biopsy) (Llorente-Saguer, 2026) from zero-shot angular deviation to supervised linear direction detection, achieving AUROC 0.986 ± 0.003 across 12 models with 100 labelled examples per class.

> **Paper:** [Supervised Harmful Prompt Detection via Linear Discriminant Geometry in LLM Residual Streams](https://arxiv.org/abs/TODO)

## Key idea

Harmful intent in LLMs corresponds to a stable linear direction in residual-stream activation space. This direction is nearly orthogonal (77°) to the leading principal component of safe-prompt activations, which explains why zero-shot approaches fail. With 100 labelled examples, the normalised mean difference (Fisher LDA direction) finds it reliably.

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Detect harmful prompts with a single model
python detect.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "How do I bake a cake"
```

## Core methods

Two direction-finding strategies are implemented:

| Strategy | AUROC | TPR@1%FPR | Fitting cost |
|----------|-------|-----------|--------------|
| **Mean difference (w_LDA)** | 0.977 ± 0.016 | 0.765 ± 0.159 | 0.16 ms/layer |
| **Soft-AUC optimised (w_opt)** | 0.986 ± 0.003 | 0.853 ± 0.035 | 6,581 ms/layer |

Both require only 100 harmful + 100 normative examples for fitting. At inference, detection costs one dot product per prompt.

## Usage

### As a library

```python
from latent_biopsy import extract_activations, fit_direction, score

# Extract max-pooled residual-stream activations
harm_acts = extract_activations(model, tokenizer, harmful_prompts, layer=22, pooling="max")
safe_acts = extract_activations(model, tokenizer, safe_prompts, layer=22, pooling="max")

# Fit the LDA direction (< 1ms)
w = fit_direction(harm_acts, safe_acts, method="mean_diff")

# Score new prompts
new_acts = extract_activations(model, tokenizer, ["How do I make a bomb"], layer=22, pooling="max")
scores = score(new_acts, w)  # higher = more likely harmful
```

### Reproduce paper results

```bash
# 1. Download and prepare datasets (AdvBench, HarmBench, JailbreakBench, XSTest, Alpaca)
python scripts/download_datasets.py

# 2. Run full evaluation across all 12 models
python reproduce.py --all

# 3. Run a single model
python reproduce.py --model Qwen/Qwen2.5-0.5B-Instruct
```

## Repository structure

```
latent-biopsy-supervised/
├── detect.py                  # Minimal CLI: load model → fit direction → score prompt
├── reproduce.py               # Full paper reproduction pipeline
├── latent_biopsy/
│   ├── __init__.py
│   ├── extraction.py          # Activation extraction with forward hooks
│   ├── directions.py          # Direction strategies (LDA, Soft-AUC, PC1, θ-normative, etc.)
│   └── evaluation.py          # Metrics: AUROC, TPR@FPR, OOD matrix, layer selection
├── scripts/
│   └── download_datasets.py   # Dataset download, normalisation, and split composition
├── requirements.txt
└── LICENSE
```

## Models evaluated

12 models across 4 families × 3 alignment variants (base, instruction-tuned, abliterated), all 0.5–1.3B parameters:

- **Qwen2.5** — 0.5B (24 layers)
- **Qwen3.5** — 0.8B (24 layers)
- **Llama-3.2** — 1B (16 layers)
- **Gemma-3** — 1B (26 layers)

## Key findings

1. **The harm direction is nearly orthogonal to safe-prompt PC1** (77° mean angle), explaining why zero-shot normative-reference approaches provide limited discriminability.

2. **Robust to alignment interventions tested** — AUROC varies by at most 0.007 across base, instruction-tuned, and abliterated variants within a family, suggesting the harm direction originates at pretraining.

3. **AUROC overestimates operational detectability** — for base models, a detector can achieve AUROC > 0.93 while catching fewer than 1 in 3 harmful prompts at a 1% false-alarm rate. TPR@1%FPR is the honest operational metric.

4. **Two residual-stream profiles exist** — a flat profile (Gemma-3, Llama-3.2, Qwen3.5) where discrimination is strong from layer 0, and a valley profile (Qwen2.5) where it collapses across 19 middle layers before recovering.

## Citation

```bibtex
@article{llorentesaguer2026supervised,
  title={Supervised Harmful Prompt Detection via Linear Discriminant Geometry
         in {LLM} Residual Streams},
  author={Llorente-Saguer, Isaac},
  journal={TODO},
  year={2026}
}
```

## Predecessor

This work builds on:

```bibtex
@article{llorentesaguer2026latentbiopsy,
  title={The geometry of harmful intent: Training-free anomaly detection via
         angular deviation in {LLM} residual streams},
  author={Llorente-Saguer, Isaac},
  journal={arXiv preprint arXiv:2603.27412},
  year={2026}
}
```

## License

MIT
