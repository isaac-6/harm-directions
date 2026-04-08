"""
tests/test_core.py
------------------
Unit tests for latent-biopsy-supervised core algorithms.
Runs on CPU with synthetic data — no GPU or model downloads needed.

    pytest tests/ -v
"""

import numpy as np
import pytest

from latent_biopsy import fit_direction, score
from latent_biopsy.directions import (
    mean_diff, soft_auc, pc1_normative, theta_normative,
    theta_two_class, random_direction, score_projection, score_angular,
)
from latent_biopsy.evaluation import (
    auroc, effective_auroc, tpr_at_fpr, direction_angle,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Two well-separated Gaussian clouds in 64 dimensions."""
    rng = np.random.default_rng(42)
    D = 64
    safe = rng.standard_normal((50, D))
    harm = rng.standard_normal((50, D)) + 1.0  # shifted mean
    return safe, harm, D


@pytest.fixture
def overlapping_data():
    """Two barely-separated clouds — stress test for metrics."""
    rng = np.random.default_rng(99)
    D = 32
    safe = rng.standard_normal((50, D))
    harm = rng.standard_normal((50, D)) + 0.05
    return safe, harm, D


# ---------------------------------------------------------------------------
# Direction strategies
# ---------------------------------------------------------------------------

class TestDirections:

    @pytest.mark.parametrize("method", [
        "mean_diff", "soft_auc", "pc1_normative",
        "theta_normative", "theta_two_class",
    ])
    def test_fit_returns_correct_shape(self, synthetic_data, method):
        safe, harm, D = synthetic_data
        w = fit_direction(harm, safe, method=method)
        assert w.shape == (D,)

    @pytest.mark.parametrize("method", [
        "mean_diff", "soft_auc", "theta_normative", "theta_two_class",
    ])
    def test_fit_returns_unit_norm(self, synthetic_data, method):
        safe, harm, D = synthetic_data
        w = fit_direction(harm, safe, method=method)
        assert abs(np.linalg.norm(w) - 1.0) < 1e-5, f"norm = {np.linalg.norm(w)}"

    def test_random_direction_shape_and_norm(self):
        w = random_direction(128)
        assert w.shape == (128,)
        assert abs(np.linalg.norm(w) - 1.0) < 1e-5

    def test_random_direction_deterministic(self):
        w1 = random_direction(64, seed=42)
        w2 = random_direction(64, seed=42)
        np.testing.assert_array_equal(w1, w2)

    def test_random_direction_different_seeds(self):
        w1 = random_direction(64, seed=42)
        w2 = random_direction(64, seed=99)
        assert not np.allclose(w1, w2)

    def test_mean_diff_points_toward_harm(self, synthetic_data):
        safe, harm, D = synthetic_data
        w = mean_diff(safe, harm)
        assert harm.mean(0) @ w > safe.mean(0) @ w

    def test_soft_auc_close_to_lda(self, synthetic_data):
        safe, harm, _ = synthetic_data
        w_lda = mean_diff(safe, harm)
        w_opt = soft_auc(safe, harm, n_iter=100)
        angle = direction_angle(w_lda, w_opt)
        assert angle < 30, f"LDA-SoftAUC angle = {angle:.1f}° (expected < 30°)"

    def test_invalid_method_raises(self, synthetic_data):
        safe, harm, _ = synthetic_data
        with pytest.raises(ValueError, match="Unknown method"):
            fit_direction(harm, safe, method="nonexistent")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestScoring:

    def test_projection_shape(self, synthetic_data):
        safe, harm, D = synthetic_data
        w = mean_diff(safe, harm)
        s = score_projection(harm, w)
        assert s.shape == (50,)

    def test_angular_shape(self, synthetic_data):
        safe, harm, D = synthetic_data
        w = theta_normative(safe)
        s = score_angular(harm, w)
        assert s.shape == (50,)

    def test_angular_range(self, synthetic_data):
        safe, harm, _ = synthetic_data
        w = theta_normative(safe)
        s = score_angular(np.vstack([safe, harm]), w)
        assert np.all(s >= 0) and np.all(s <= np.pi)

    def test_score_api_projection(self, synthetic_data):
        safe, harm, _ = synthetic_data
        w = mean_diff(safe, harm)
        s = score(harm, w, method="projection")
        assert s.shape == (50,)

    def test_score_api_angular(self, synthetic_data):
        safe, harm, _ = synthetic_data
        w = theta_normative(safe)
        s = score(harm, w, method="angular")
        assert s.shape == (50,)

    def test_score_invalid_method(self, synthetic_data):
        safe, harm, _ = synthetic_data
        w = mean_diff(safe, harm)
        with pytest.raises(ValueError, match="Unknown scoring method"):
            score(harm, w, method="nonexistent")

    def test_harm_scores_higher_than_safe(self, synthetic_data):
        safe, harm, _ = synthetic_data
        w = mean_diff(safe, harm)
        s_harm = score_projection(harm, w)
        s_safe = score_projection(safe, w)
        assert s_harm.mean() > s_safe.mean()


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

class TestMetrics:

    def test_auroc_perfect_separation(self):
        neg = np.array([0.0, 0.1, 0.2])
        pos = np.array([0.8, 0.9, 1.0])
        assert auroc(neg, pos) == 1.0

    def test_auroc_random(self):
        rng = np.random.default_rng(42)
        neg = rng.standard_normal(1000)
        pos = rng.standard_normal(1000)
        auc = auroc(neg, pos)
        assert 0.45 < auc < 0.55, f"AUROC = {auc} (expected ~0.5)"

    def test_effective_auroc_flips(self):
        assert effective_auroc(0.3) == 0.7
        assert effective_auroc(0.8) == 0.8
        assert effective_auroc(0.5) == 0.5

    def test_effective_auroc_nan(self):
        assert np.isnan(effective_auroc(float("nan")))

    def test_tpr_at_fpr_perfect(self):
        neg = np.zeros(100)
        pos = np.ones(100)
        tpr = tpr_at_fpr(neg, pos, target_fpr=0.01)
        assert tpr == 1.0

    def test_tpr_at_fpr_random(self):
        rng = np.random.default_rng(42)
        neg = rng.standard_normal(1000)
        pos = rng.standard_normal(1000)
        tpr = tpr_at_fpr(neg, pos, target_fpr=0.01)
        assert 0.0 <= tpr <= 0.1  # near chance at 1% FPR

    def test_tpr_at_fpr_range(self, synthetic_data):
        safe, harm, _ = synthetic_data
        w = mean_diff(safe, harm)
        s_safe = score_projection(safe, w)
        s_harm = score_projection(harm, w)
        tpr = tpr_at_fpr(s_safe, s_harm)
        assert 0.0 <= tpr <= 1.0


# ---------------------------------------------------------------------------
# Direction geometry
# ---------------------------------------------------------------------------

class TestGeometry:

    def test_angle_identical(self):
        w = np.array([1.0, 0.0, 0.0])
        assert direction_angle(w, w) == pytest.approx(0.0, abs=1e-5)

    def test_angle_orthogonal(self):
        w1 = np.array([1.0, 0.0, 0.0])
        w2 = np.array([0.0, 1.0, 0.0])
        assert direction_angle(w1, w2) == pytest.approx(90.0, abs=1e-5)

    def test_angle_antiparallel_is_zero(self):
        """Unsigned angle: antiparallel = 0°, not 180°."""
        w1 = np.array([1.0, 0.0])
        w2 = np.array([-1.0, 0.0])
        assert direction_angle(w1, w2) == pytest.approx(0.0, abs=1e-5)

    def test_angle_symmetric(self):
        rng = np.random.default_rng(42)
        w1 = rng.standard_normal(64)
        w2 = rng.standard_normal(64)
        assert direction_angle(w1, w2) == pytest.approx(
            direction_angle(w2, w1), abs=1e-10
        )

    def test_lda_orthogonal_to_random(self, synthetic_data):
        safe, harm, D = synthetic_data
        w_lda = mean_diff(safe, harm)
        w_rand = random_direction(D, seed=0)
        angle = direction_angle(w_lda, w_rand)
        assert angle > 60, f"Expected near-orthogonal, got {angle:.1f}°"

        
# ---------------------------------------------------------------------------
# Layer Selection
# ---------------------------------------------------------------------------

class TestLayerSelection:

    def test_select_layer_val(self):
        """Validation holdout selects the correct layer."""
        from latent_biopsy.evaluation import select_layer_val
        rng = np.random.default_rng(42)
        n_layers = 5
        D = 32

        # Make layer 3 the most separable
        fit_h = rng.standard_normal((30, n_layers, D)).astype(np.float32)
        fit_n = rng.standard_normal((30, n_layers, D)).astype(np.float32)
        val_h = rng.standard_normal((15, n_layers, D)).astype(np.float32)
        val_n = rng.standard_normal((15, n_layers, D)).astype(np.float32)

        # Add large separation at layer 3
        fit_h[:, 3, :] += 3.0
        val_h[:, 3, :] += 3.0

        best = select_layer_val(fit_h, fit_n, val_h, val_n)
        assert best == 3

    def test_select_layer_val_custom_fn(self):
        """Validation holdout works with custom direction and score functions."""
        from latent_biopsy.evaluation import select_layer_val
        from latent_biopsy.directions import soft_auc, score_projection
        rng = np.random.default_rng(42)
        n_layers = 3
        D = 16

        fit_h = rng.standard_normal((20, n_layers, D)).astype(np.float32)
        fit_n = rng.standard_normal((20, n_layers, D)).astype(np.float32)
        val_h = rng.standard_normal((10, n_layers, D)).astype(np.float32)
        val_n = rng.standard_normal((10, n_layers, D)).astype(np.float32)

        fit_h[:, 1, :] += 3.0
        val_h[:, 1, :] += 3.0

        best = select_layer_val(
            fit_h, fit_n, val_h, val_n,
            direction_fn=soft_auc, score_fn=score_projection,
        )
        assert best == 1


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_full_pipeline(self, synthetic_data):
        """fit_direction → score → auroc end-to-end."""
        safe, harm, _ = synthetic_data
        w = fit_direction(harm, safe, method="mean_diff")
        s_safe = score(safe, w)
        s_harm = score(harm, w)
        auc = effective_auroc(auroc(s_safe, s_harm))
        assert auc > 0.9

    def test_soft_auc_improves_on_random(self, synthetic_data):
        safe, harm, D = synthetic_data
        w_rand = random_direction(D)
        w_opt = fit_direction(harm, safe, method="soft_auc")
        auc_rand = effective_auroc(auroc(
            score(safe, w_rand), score(harm, w_rand)
        ))
        auc_opt = effective_auroc(auroc(
            score(safe, w_opt), score(harm, w_opt)
        ))
        assert auc_opt > auc_rand

    def test_all_methods_separate_well(self, synthetic_data):
        safe, harm, _ = synthetic_data
        for method in ["mean_diff", "soft_auc"]:
            w = fit_direction(harm, safe, method=method)
            s_safe = score(safe, w)
            s_harm = score(harm, w)
            auc = effective_auroc(auroc(s_safe, s_harm))
            assert auc > 0.85, f"{method}: AUROC = {auc}"

    def test_overlapping_data_does_not_crash(self, overlapping_data):
        safe, harm, _ = overlapping_data
        for method in ["mean_diff", "soft_auc", "pc1_normative",
                       "theta_normative", "theta_two_class"]:
            w = fit_direction(harm, safe, method=method)
            s = score(safe, w)
            assert not np.any(np.isnan(s)), f"{method} produced NaN"