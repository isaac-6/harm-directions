# Changelog


## [2.0.0] - 2026-05-09
Release corresponding to v2 of arXiv:2604.18901

- Added experiment scripts for all main results and ablations
- Added figure and table generation scripts
- Updated Makefile with additional commands for reproducibility
- Updated README with current usage

## [1.0.0] - 2026-04-20

Release corresponding to arXiv:2604.18901

- Models loaded in native precision.
- Updated paper title, description and key figure.
- Added Zenodo DOI.

## [0.1.0] - 2026-04-19

Initial release.

- Supervised linear harm-direction probes for LLM residual streams
- Six direction strategies: PC1, mean difference, soft-AUC, θ-normative, θ two-class, random baseline
- Cross-variant transfer analysis pipeline
- Reproduction scripts for the 12-model main experiment and Qwen3.5 scaling extension (2B, 4B, 9B)
- Docs: README with headline figure, citation metadata, MIT license
- CI: ruff + mypy + pytest across Python 3.10/3.11/3.12
