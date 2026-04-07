"""
latent_biopsy/evaluation.py
---------------------------
Evaluation metrics and layer selection for harmful-prompt detection.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auroc(scores_neg: np.ndarray, scores_pos: np.ndarray) -> float:
    """Compute AUROC. Positive = harmful, negative = normative."""
    y = np.concatenate([np.zeros(len(scores_neg)), np.ones(len(scores_pos))])
    s = np.concatenate([scores_neg, scores_pos])
    if np.isnan(s).any() or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def effective_auroc(raw_auroc: float) -> float:
    """Sign-correct AUROC: max(AUROC, 1 - AUROC)."""
    if np.isnan(raw_auroc):
        return float("nan")
    return max(raw_auroc, 1.0 - raw_auroc)


def tpr_at_fpr(
    scores_neg: np.ndarray,
    scores_pos: np.ndarray,
    target_fpr: float = 0.01,
) -> float:
    """
    True positive rate at a given false positive rate.

    The operationally relevant metric: what fraction of harmful prompts
    are flagged while incorrectly flagging only `target_fpr` of benign ones.
    """
    y = np.concatenate([np.zeros(len(scores_neg)), np.ones(len(scores_pos))])
    s = np.concatenate([scores_neg, scores_pos])
    if np.isnan(s).any() or len(np.unique(y)) < 2:
        return float("nan")

    fpr, tpr, _ = roc_curve(y, s)
    # Interpolate TPR at target FPR
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(0, min(idx, len(fpr) - 2))
    if fpr[idx + 1] == fpr[idx]:
        return float(tpr[idx])
    # Linear interpolation
    alpha = (target_fpr - fpr[idx]) / (fpr[idx + 1] - fpr[idx])
    return float(tpr[idx] + alpha * (tpr[idx + 1] - tpr[idx]))


def select_layer_cv(
    fit_harm_all: np.ndarray,
    fit_norm_all: np.ndarray,
    n_folds: int = 4,
    seed: int = 42,
) -> int:
    """
    Select the operating layer by cross-validated AUROC of the mean-difference
    direction on the fit set.

    Parameters
    ----------
    fit_harm_all : np.ndarray of shape (n_harm, n_layers, D)
        Harmful fit-set activations at all layers.
    fit_norm_all : np.ndarray of shape (n_norm, n_layers, D)
        Normative fit-set activations at all layers.
    n_folds : int
        Number of CV folds.
    seed : int
        Random seed for fold assignment.

    Returns
    -------
    int
        Best layer index (argmax CV AUROC).
    """
    from .directions import mean_diff, score_projection

    rng = np.random.default_rng(seed)
    n_harm, n_layers, D = fit_harm_all.shape
    n_norm = fit_norm_all.shape[0]

    # Assign folds
    harm_folds = rng.integers(0, n_folds, size=n_harm)
    norm_folds = rng.integers(0, n_folds, size=n_norm)

    best_layer = 0
    best_auroc = -1.0

    for layer in range(n_layers):
        all_scores = []
        all_labels = []

        for fold in range(n_folds):
            # Train on all folds except this one
            h_train = fit_harm_all[harm_folds != fold, layer, :]
            n_train = fit_norm_all[norm_folds != fold, layer, :]
            h_test = fit_harm_all[harm_folds == fold, layer, :]
            n_test = fit_norm_all[norm_folds == fold, layer, :]

            if len(h_train) == 0 or len(n_train) == 0:
                continue
            if len(h_test) == 0 and len(n_test) == 0:
                continue

            w = mean_diff(n_train, h_train)
            if len(h_test) > 0:
                all_scores.append(score_projection(h_test, w))
                all_labels.append(np.ones(len(h_test)))
            if len(n_test) > 0:
                all_scores.append(score_projection(n_test, w))
                all_labels.append(np.zeros(len(n_test)))

        if not all_scores:
            continue

        y = np.concatenate(all_labels)
        s = np.concatenate(all_scores)
        layer_auroc = effective_auroc(auroc(s[y == 0], s[y == 1]))

        if layer_auroc > best_auroc:
            best_auroc = layer_auroc
            best_layer = layer

    return best_layer

def select_layer_val(
    fit_harm_all: np.ndarray,
    fit_norm_all: np.ndarray,
    val_harm_all: np.ndarray,
    val_norm_all: np.ndarray,
    direction_fn=None,
    score_fn=None,
) -> int:
    """
    Select the operating layer by validation holdout AUROC.

    Fits the direction on the fit set at each layer, scores the
    validation set, and returns the layer with the highest effective
    AUROC. No data serves double duty.

    Parameters
    ----------
    fit_harm_all : np.ndarray of shape (n_harm, n_layers, D)
    fit_norm_all : np.ndarray of shape (n_norm, n_layers, D)
    val_harm_all : np.ndarray of shape (n_val_harm, n_layers, D)
    val_norm_all : np.ndarray of shape (n_val_norm, n_layers, D)
    direction_fn : callable, optional
        Direction fitting function. Default: mean_diff.
    score_fn : callable, optional
        Scoring function. Default: score_projection.

    Returns
    -------
    int
        Best layer index (argmax validation AUROC).
    """
    from .directions import mean_diff, score_projection

    if direction_fn is None:
        direction_fn = mean_diff
    if score_fn is None:
        score_fn = score_projection

    n_layers = fit_harm_all.shape[1]
    best_layer = 0
    best_auroc = -1.0

    for layer in range(n_layers):
        h_fit = fit_harm_all[:, layer, :]
        n_fit = fit_norm_all[:, layer, :]
        h_val = val_harm_all[:, layer, :]
        n_val = val_norm_all[:, layer, :]

        try:
            w = direction_fn(n_fit, h_fit)
        except Exception:
            continue

        s_val_h = score_fn(h_val, w)
        s_val_n = score_fn(n_val, w)
        layer_auroc = effective_auroc(auroc(s_val_n, s_val_h))

        if layer_auroc > best_auroc:
            best_auroc = layer_auroc
            best_layer = layer

    return best_layer

def direction_angle(w1: np.ndarray, w2: np.ndarray) -> float:
    """Unsigned angle (degrees) between two direction vectors."""
    cos = np.clip(np.abs(np.dot(w1, w2)), 0.0, 1.0)
    return float(np.degrees(np.arccos(cos)))
