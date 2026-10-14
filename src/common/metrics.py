"""
Evaluation metrics used across all assignments.

All functions take NumPy arrays (or objects convertible to NumPy) and return
Python floats.  This keeps metric computation independent of the deep-learning
framework and allows reuse in both training loops and API tests.
"""


import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

ArrayLike = np.ndarray | list


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Classification accuracy (fraction of correct predictions).

    Args:
        y_true: Ground-truth labels, shape (N,).
        y_pred: Predicted labels (not probabilities), shape (N,).

    Returns:
        Accuracy as a float in [0, 1].
    """
    return float(accuracy_score(np.asarray(y_true), np.asarray(y_pred)))


def macro_f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Unweighted macro-averaged F1 score.

    Computes per-class F1 and takes the unweighted mean, exposing poor
    performance on minority classes that accuracy would hide.

    Args:
        y_true: Ground-truth labels, shape (N,).
        y_pred: Predicted labels, shape (N,).

    Returns:
        Macro F1 as a float in [0, 1].
    """
    return float(
        f1_score(
            np.asarray(y_true),
            np.asarray(y_pred),
            average="macro",
            zero_division=0,
        )
    )


def roc_auc(
    y_true: ArrayLike,
    y_score: ArrayLike,
    multi_class: str = "ovr",
) -> float:
    """Area under the ROC curve.

    For binary classification `y_score` should be the positive-class
    probability or raw score.  For multi-class, `y_score` should be a
    (N, C) array of per-class probabilities and `multi_class` controls
    the averaging strategy.

    Args:
        y_true: Ground-truth labels, shape (N,).
        y_score: Predicted scores or probabilities.
        multi_class: One of ``"ovr"`` or ``"ovo"`` (multi-class only).

    Returns:
        AUC as a float in [0, 1].
    """
    y_true_np = np.asarray(y_true)
    y_score_np = np.asarray(y_score)

    # Binary case
    if y_score_np.ndim == 1 or y_score_np.shape[1] == 1:
        return float(roc_auc_score(y_true_np, y_score_np.ravel()))

    # Multi-class
    return float(
        roc_auc_score(
            y_true_np,
            y_score_np,
            multi_class=multi_class,
            average="macro",
        )
    )


def average_precision(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Average precision (area under the precision–recall curve).

    For link prediction, where positive-class imbalance makes AUC less
    informative about top-of-list quality.

    Args:
        y_true: Binary ground-truth labels, shape (N,).
        y_score: Predicted scores for the positive class, shape (N,).

    Returns:
        AP as a float in [0, 1].
    """
    return float(
        average_precision_score(np.asarray(y_true), np.asarray(y_score))
    )
