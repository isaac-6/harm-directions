# Convenience targets for local development.
# Mirrors what CI runs; use `make check` before pushing.

PY      := python
RESULTS := results

.PHONY: help install check lint format format-check typecheck test clean \
        experiments g1 g2 g3 g4 g5 clean-experiments \
        annex figure_artifacts baseline_comparison \
        make_angle_figure make_auroc_vs_tpr_figure make_full_results_tables \
        make_full_transfer_table make_heatmaps make_layer_profile_figure \
        make_ood_tables make_scale_table make_score_distribution_figure \
        clean-annex

help:
	@echo "Available targets:"
	@echo "  install              Install package with dev dependencies"
	@echo "  check                Run all checks (lint, format, typecheck, test)"
	@echo "  lint                 Run ruff check"
	@echo "  format               Run ruff format (modifies files)"
	@echo "  typecheck            Run mypy"
	@echo "  test                 Run pytest"
	@echo "  experiments          Run all G1-G5 experiments sequentially"
	@echo "  g1..g5               Run a single experiment (with prerequisites)"
	@echo "  annex                Run full annex pipeline (artifacts → baseline → figures/tables)"
	@echo "  figure_artifacts     Extract per-model scores/layers/directions for figures"
	@echo "  baseline_comparison  Evaluate external safety-classifier baselines"
	@echo "  make_<x>             Run a single figure/table script (see experiments/make_*.py)"
	@echo "  clean                Remove caches and build artefacts"
	@echo "  clean-experiments    Remove G1-G5 outputs in $(RESULTS)/"
	@echo "  clean-annex          Remove annex outputs (figures/, artifacts/)"

install:
	pip install -e ".[dev]"

check: lint format-check typecheck test
	@echo "All checks passed."

lint:
	ruff check src tests scripts experiments

format:
	ruff format src tests scripts experiments

format-check:
	ruff format --check src tests scripts experiments

typecheck:
	mypy src

test:
	pytest

# ---------------------------------------------------------------------------
# Experiments (G1-G5). Run from repo root: `make experiments`.
# G2 caches artifacts under $(RESULTS)/artifacts_neurips/ that G3 and G5 read.
# ---------------------------------------------------------------------------

experiments: g1 g2 g3 g4 g5
	@echo "All G1-G5 experiments complete. See $(RESULTS)/"

g1:
	$(PY) experiments/run_g1_chat_template.py

g2: g1
	$(PY) experiments/run_g2_arditi_angle.py

g3: g2
	$(PY) experiments/run_g3_arditi_projected_refit.py

g4: g1
	$(PY) experiments/run_g4_self_projected_refit.py

g5: g2
	$(PY) experiments/run_g5_cross_protocol_refit.py

clean-experiments:
	rm -rf $(RESULTS)/g*.csv $(RESULTS)/table_g*.tex $(RESULTS)/artifacts_neurips

# ---------------------------------------------------------------------------
# Annex pipeline: figure artifacts → baseline comparison → figures & tables.
# Run with `make annex`; individual steps are also available as targets.
# make_* scripts depend on the artifacts produced by figure_artifacts.
# ---------------------------------------------------------------------------

annex:
	$(PY) experiments/generate_figure_artifacts.py
	$(PY) experiments/baseline_comparison.py
	$(PY) experiments/make_angle_figure.py
	$(PY) experiments/make_auroc_vs_tpr_figure.py
	$(PY) experiments/make_full_results_tables.py
	$(PY) experiments/make_full_transfer_table.py
	$(PY) experiments/make_heatmaps.py
	$(PY) experiments/make_layer_profile_figure.py
	$(PY) experiments/make_ood_tables.py
	$(PY) experiments/make_scale_table.py
	$(PY) experiments/make_score_distribution_figure.py
	@echo "Annex pipeline complete. See $(RESULTS)/ and figures/"

figure_artifacts:
	$(PY) experiments/generate_figure_artifacts.py

baseline_comparison:
	$(PY) experiments/baseline_comparison.py

make_angle_figure:
	$(PY) experiments/make_angle_figure.py

make_auroc_vs_tpr_figure:
	$(PY) experiments/make_auroc_vs_tpr_figure.py

make_full_results_tables:
	$(PY) experiments/make_full_results_tables.py

make_full_transfer_table:
	$(PY) experiments/make_full_transfer_table.py

make_heatmaps:
	$(PY) experiments/make_heatmaps.py

make_layer_profile_figure:
	$(PY) experiments/make_layer_profile_figure.py

make_ood_tables:
	$(PY) experiments/make_ood_tables.py

make_scale_table:
	$(PY) experiments/make_scale_table.py

make_score_distribution_figure:
	$(PY) experiments/make_score_distribution_figure.py

clean-annex:
	rm -rf $(RESULTS)/artifacts figures/*.pdf figures/*.png

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage coverage.xml