"""
harm_directions/directions.py
---------------------------
Direction-finding strategies for harmful-prompt detection in LLM residual streams.

All strategies take fit-set activations and return a unit-norm direction vector w.
Detection score at inference: s(p) = f(p) · w  (one dot product).

Strategies
----------
  mean_diff        unit(μ_harm - μ_norm).  Fisher LDA under equal spherical covariance.
  soft_auc         Riemannian gradient ascent on soft-AUC of projection scores.
  pc1_normative    Leading principal component of normative fit set.  Zero-shot.
  theta_normative  unit(μ_norm), scored by angular deviation.  Zero-shot.
  theta_two_class  Angle-based soft-AUC optimiser.  Two-class.
  random           Random unit vector.  Sanity-check baseline.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    """Normalise to unit length. Returns zero vector if norm is negligible."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_projection(acts: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Signed projection score: acts @ w.  Higher → more harmful."""
    return (acts @ w).astype(np.float64)


def score_angular(acts: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Unsigned angular deviation: arccos(x̂ · ŵ).  Higher → more anomalous."""
    w_hat = w / (np.linalg.norm(w) + 1e-12)
    norms = np.linalg.norm(acts, axis=1, keepdims=True) + 1e-12
    cos = np.clip((acts / norms) @ w_hat, -1.0, 1.0)
    return np.arccos(cos).astype(np.float64)


# ---------------------------------------------------------------------------
# Direction strategies
# ---------------------------------------------------------------------------

def mean_diff(fit_norm: np.ndarray, fit_harm: np.ndarray) -> np.ndarray:
    """
    Normalised mean difference: w = unit(μ_harm - μ_norm).

    This is the Fisher LDA direction under equal spherical within-class
    covariance.  Fitting cost: O(n), ~0.16 ms per layer.
    """
    return _unit(fit_harm.mean(axis=0) - fit_norm.mean(axis=0))


def soft_auc(
    fit_norm: np.ndarray,
    fit_harm: np.ndarray,
    tau: float = 1.0,
    n_iter: int = 300,
    lr: float = 0.05,
    patience: int = 20,
    grad_tol: float = 1e-5,
    seed: int = 42,
) -> np.ndarray:
    """
    Riemannian gradient ascent on soft-AUC of projection scores.

    Maximises the sigmoid-smoothed Mann-Whitney U statistic:
        Û(w) = (1/n+n-) Σ σ(w·(x_i - x_j) / τ)

    Warm-started from mean_diff.  Stops when the Riemannian gradient norm
    stays below grad_tol for `patience` consecutive steps, or after n_iter.

    Parameters
    ----------
    tau : float
        Temperature for sigmoid smoothing.
    n_iter : int
        Maximum optimisation steps.
    lr : float
        Learning rate.
    patience : int
        Consecutive steps below grad_tol before early stopping.
    grad_tol : float
        Riemannian gradient norm threshold.
    seed : int
        Random seed (used only if mean_diff is degenerate).
    """
    rng = np.random.default_rng(seed)
    D = fit_norm.shape[1]
    Xn = fit_norm.astype(np.float32)
    Xp = fit_harm.astype(np.float32)

    # Warm start from mean difference
    w = _unit((Xp.mean(0) - Xn.mean(0)).astype(np.float32))
    if np.linalg.norm(w) < 0.5:
        w = _unit(rng.standard_normal(D).astype(np.float32))

    # Pairwise difference tensor: (n+ * n-, D)
    diffs = (Xp[:, None, :] - Xn[None, :, :]).reshape(-1, D)

    below_tol_count = 0
    for step in range(n_iter):
        margins = (diffs @ w) / tau
        sig = 1.0 / (1.0 + np.exp(-np.clip(margins, -30, 30)))
        sig_d = sig * (1.0 - sig)
        grad = (sig_d[:, None] * diffs).mean(0) / tau

        # Riemannian gradient: project out component along w
        grad_riem = grad - (grad @ w) * w

        # Early stopping
        if np.linalg.norm(grad_riem) < grad_tol:
            below_tol_count += 1
            if below_tol_count >= patience:
                break
        else:
            below_tol_count = 0

        w = _unit(w + lr * grad_riem)

    return _unit(w.astype(np.float64))


def pc1_normative(fit_norm: np.ndarray, **_) -> np.ndarray:
    """Leading principal component of normative activations.  Zero-shot."""
    pca = PCA(n_components=1)
    pca.fit(fit_norm)
    return pca.components_[0]


def theta_normative(fit_norm: np.ndarray, **_) -> np.ndarray:
    """
    Normative mean centroid: w = unit(μ_norm).

    Scored by angular deviation (use score_angular).
    This is the zero-shot LatentBiopsy baseline.
    """
    return _unit(fit_norm.mean(axis=0))


def theta_two_class(
    fit_norm: np.ndarray,
    fit_harm: np.ndarray,
    tau: float = 0.3,
    n_iter: int = 300,
    lr: float = 0.02,
    patience: int = 20,
    grad_tol: float = 1e-5,
    seed: int = 42,
) -> np.ndarray:
    """
    Riemannian gradient ascent on soft-AUC of angular deviation scores.

    Unlike soft_auc (projection-based), this optimises angle-based separation.
    The gradient of arccos introduces a 1/sin(θ) weighting.
    Warm-started from the normative centroid direction.
    """
    rng = np.random.default_rng(seed)
    D = fit_norm.shape[1]
    eps = 1e-3

    # Pre-normalise activations
    Xn = (fit_norm / (np.linalg.norm(fit_norm, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    Xp = (fit_harm / (np.linalg.norm(fit_harm, axis=1, keepdims=True) + 1e-12)).astype(np.float32)

    w = _unit(fit_norm.mean(axis=0).astype(np.float32))
    if np.linalg.norm(w) < 0.5:
        w = _unit(rng.standard_normal(D).astype(np.float32))

    below_tol_count = 0
    for step in range(n_iter):
        cos_p = np.clip(Xp @ w, -1.0, 1.0)
        cos_n = np.clip(Xn @ w, -1.0, 1.0)
        theta_p = np.arccos(cos_p)
        theta_n = np.arccos(cos_n)

        margins = (theta_p[:, None] - theta_n[None, :]).ravel() / tau
        sig = 1.0 / (1.0 + np.exp(-np.clip(margins, -30, 30)))
        sig_d = sig * (1.0 - sig)

        sin_p = np.sqrt(np.maximum(1 - cos_p ** 2, eps))
        sin_n = np.sqrt(np.maximum(1 - cos_n ** 2, eps))
        dtheta_p = -(Xp / sin_p[:, None])
        dtheta_n = -(Xn / sin_n[:, None])

        sig_d_mat = sig_d.reshape(len(Xp), len(Xn))
        grad_pos = (sig_d_mat[:, :, None] * dtheta_p[:, None, :]).mean((0, 1))
        grad_neg = (sig_d_mat[:, :, None] * dtheta_n[None, :, :]).mean((0, 1))
        grad = (grad_pos - grad_neg) / tau

        grad_riem = grad - (grad @ w) * w

        if np.linalg.norm(grad_riem) < grad_tol:
            below_tol_count += 1
            if below_tol_count >= patience:
                break
        else:
            below_tol_count = 0

        w = _unit(w + lr * grad_riem)

    return _unit(w.astype(np.float64))


def random_direction(D: int, seed: int = 42) -> np.ndarray:
    """Uniformly random unit vector.  Chance-level baseline."""
    rng = np.random.default_rng(seed)
    return _unit(rng.standard_normal(D).astype(np.float64))
