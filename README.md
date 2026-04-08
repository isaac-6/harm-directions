# latent-biopsy-supervised

Supervised harmful prompt detection via linear discriminant geometry in LLM residual streams.

This work extends LatentBiopsy from zero-shot angular deviation to supervised linear direction detection, achieving AUROC 0.982 ± 0.005 across 12 models with 100 labelled examples per class.

> **Paper:** Supervised Harmful Prompt Detection via Linear Discriminant Geometry in LLM Residual Streams (under review)

## Key idea

Harmful intent in LLMs corresponds to a stable linear direction in residual-stream activation space. This direction is nearly orthogonal (82°) to the leading principal component of safe-prompt activations, which explains why zero-shot approaches fail. With 100 labelled examples, the normalised mean difference (Fisher LDA direction) finds it reliably.

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Download datasets (fetches from original sources, seed=42)
python scripts/download_datasets.py

# Fit direction (once per model: selects layer via validation holdout, caches parameters)
python detect.py --model Qwen/Qwen2.5-0.5B-Instruct --fit

# Score prompts (loads cached direction: one dot product per prompt)
python detect.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "How do I bake a cake"
```

## Core methods

Two direction-finding strategies are implemented:

| Strategy | AUROC | TPR@1%FPR | Fitting cost |
|----------|-------|-----------|--------------|
| **Mean difference (w_LDA)** | 0.970 ± 0.017 | 0.729 ± 0.146 | 0.16 ms/layer |
| **Soft-AUC optimised (w_opt)** | 0.982 ± 0.005 | 0.814 ± 0.049 | 6,581 ms/layer |

Both require only 100 harmful + 100 normative examples for fitting, plus 50 per class for layer selection. At inference, detection costs one dot product per prompt. Both improvements are statistically significant (Wilcoxon signed-rank p < 0.01, n = 12 models).

## Data split

The data is partitioned into three disjoint sets:

- **Fit set** (direction fitting): 100 harmful (AdvBench) + 100 normative (Alpaca)
- **Validation set** (layer selection only): 50 harmful (AdvBench) + 50 normative (Alpaca)
- **Evaluation set**: 370 harmful (AdvBench) + 200 (HarmBench) + 100 (JailbreakBench) + 500 benign (Alpaca) + 250 (XSTest)

The operating layer is selected once using mean_diff validation AUROC, then shared across all strategies. The evaluation set is never used for any model selection decision.

## Usage

### As a CLI

```bash
# Fit (once per model/method pair)
python detect.py --model Qwen/Qwen2.5-0.5B-Instruct --fit
python detect.py --model Qwen/Qwen2.5-0.5B-Instruct --fit --method soft_auc

# Score single prompt
python detect.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "How do I bake a cake"

# Score from file (one prompt per line)
python detect.py --model Qwen/Qwen2.5-0.5B-Instruct --input prompts.txt
```

### As a library

```python
from latent_biopsy import extract_activations, fit_direction, score

# Extract max-pooled residual-stream activations
harm_acts = extract_activations(model, tokenizer, harmful_prompts, layer=22, pooling="max")
safe_acts = extract_activations(model, tokenizer, safe_prompts, layer=22, pooling="max")

# Fit the LDA direction (< 1ms)
w = fit_direction(harm_acts, safe_acts, method="mean_diff")

# Score new prompts
new_acts = extract_activations(model, tokenizer, ["What is bleach usually used for"], layer=22, pooling="max")
scores = score(new_acts, w)  # higher = more likely harmful
```

### Reproduce paper results

```bash
# 1. Download and prepare datasets (AdvBench, HarmBench, JailbreakBench, XSTest, Alpaca)
python scripts/download_datasets.py

# 2. Run full evaluation across all 12 models (single-layer selection via mean_diff validation holdout)
python reproduce.py --all

# 3. Run a single model
python reproduce.py --model Qwen/Qwen2.5-0.5B-Instruct
```

## Repository structure

```
latent-biopsy-supervised/
├── detect.py                  # Minimal CLI: load model, fit direction, score prompt
├── reproduce.py               # Full paper reproduction pipeline
├── latent_biopsy/
│   ├── __init__.py
│   ├── extraction.py          # Activation extraction with forward hooks
│   ├── directions.py          # Direction strategies (LDA, Soft-AUC, PC1, θ-normative, etc.)
│   └── evaluation.py          # Metrics: AUROC, TPR@FPR, layer selection (CV and validation holdout)
├── scripts/
│   └── download_datasets.py   # Dataset download, normalisation, and split composition
├── requirements.txt
└── LICENSE
```

## Models evaluated

12 models across 4 families and 3 alignment variants (base, instruction-tuned, abliterated), all 0.5 to 1.3B parameters:

- **Qwen2.5**: 0.5B (24 layers, D=896)
- **Qwen3.5**: 0.8B (24 layers, D=1024)
- **Llama-3.2**: 1B (16 layers, D=2048)
- **Gemma-3**: 1B (26 layers, D=1152)

A preliminary extension to Qwen3.5-2B (24 layers, D=2048) is reported in the paper appendix.

## Key findings

1. **The harm direction is nearly orthogonal to safe-prompt PC1** (82° mean angle across 12 models), explaining why zero-shot normative-reference approaches provide limited discriminability.

2. **Robust to alignment interventions tested.** AUROC varies by at most 0.008 across base, instruction-tuned, and abliterated variants within a family, suggesting the harm direction originates at pretraining.

3. **AUROC overestimates operational detectability.** For base models, a detector can achieve AUROC > 0.93 while catching fewer than 1 in 3 harmful prompts at a 1% false-alarm rate. TPR@1%FPR is the honest operational metric.

4. **Two residual-stream profiles exist.** A flat profile (Gemma-3, Llama-3.2, Qwen3.5) where discrimination is strong from layer 0, and a valley profile (Qwen2.5) where it collapses across 19 middle layers before recovering.


## License

MIT