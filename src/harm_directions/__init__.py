"""
latent-biopsy-supervised
========================

Supervised harmful prompt detection via linear discriminant geometry
in LLM residual streams.

Quick start::

    from harm_directions import fit_direction, score

    w = fit_direction(harm_acts, safe_acts, method="mean_diff")
    scores = score(new_acts, w)

For activation extraction (requires torch + transformers)::

    from harm_directions import extract_activations, extract_all_layers
"""

# Core algorithms (numpy only — no torch dependency)
from .directions import (
    mean_diff,
    pc1_normative,
    random_direction,
    score_angular,
    score_projection,
    soft_auc,
    theta_normative,
    theta_two_class,
)
from .evaluation import (
    auroc,
    direction_angle,
    effective_auroc,
    tpr_at_fpr,
)


# Lazy imports for extraction (requires torch + transformers)
def __getattr__(name):
    if name in ("extract_activations", "extract_all_layers"):
        from .extraction import extract_activations, extract_all_layers

        globals()["extract_activations"] = extract_activations
        globals()["extract_all_layers"] = extract_all_layers
        return globals()[name]
    raise AttributeError(f"module 'harm_directions' has no attribute {name!r}")


def fit_direction(harm_acts, safe_acts, method="mean_diff", **kwargs):
    """
    Fit a detection direction from labelled activations.

    Parameters
    ----------
    harm_acts : np.ndarray of shape (n_harm, D)
        Harmful fit-set activations at one layer.
    safe_acts : np.ndarray of shape (n_safe, D)
        Normative fit-set activations at one layer.
    method : str
        One of: "mean_diff", "soft_auc", "pc1_normative",
        "theta_normative", "theta_two_class".

    Returns
    -------
    np.ndarray of shape (D,)
        Unit-norm direction vector.
    """
    methods = {
        "mean_diff": mean_diff,
        "soft_auc": soft_auc,
        "pc1_normative": pc1_normative,
        "theta_normative": theta_normative,
        "theta_two_class": theta_two_class,
    }
    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(methods)}")

    fn = methods[method]
    if method in ("pc1_normative", "theta_normative"):
        return fn(safe_acts, **kwargs)
    return fn(safe_acts, harm_acts, **kwargs)


def score(acts, w, method="projection"):
    """
    Score activations against a direction vector.

    Parameters
    ----------
    acts : np.ndarray of shape (n, D)
        Activations to score.
    w : np.ndarray of shape (D,)
        Direction vector.
    method : str
        "projection" (default) or "angular".

    Returns
    -------
    np.ndarray of shape (n,)
        Detection scores.  Higher = more likely harmful.
    """
    if method == "projection":
        return score_projection(acts, w)
    elif method == "angular":
        return score_angular(acts, w)
    else:
        raise ValueError(f"Unknown scoring method '{method}'.")


__all__ = [
    "auroc",
    "direction_angle",
    "effective_auroc",
    "fit_direction",
    "mean_diff",
    "pc1_normative",
    "random_direction",
    "score",
    "score_angular",
    "score_projection",
    "soft_auc",
    "theta_normative",
    "theta_two_class",
    "tpr_at_fpr",
]
