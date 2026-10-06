"""Classification metrics of the base model."""

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

METRIC_NAMES = ["auc", "sensitivity", "specificity", "balanced_accuracy"]


def compute_metrics(y_true, y_pred_proba, threshold=0.5) -> dict:
    """AUC, sensitivity, specificity and balanced accuracy.

    Metrics that are undefined (e.g. a single class present) are NaN.
    """
    y_true = np.asarray(y_true)
    if len(y_true) == 0:
        return {name: float("nan") for name in METRIC_NAMES}

    y_pred = (y_pred_proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) == 2 else float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        # NaN if either term is NaN
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
    }
