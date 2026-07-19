"""Classification metrics used throughout the pipeline: Accuracy, Balanced
Accuracy, F1, AUC (see CLAUDE.md Step 2 / Step 9)."""

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """`y_proba` is (N, num_classes) softmax output; `y_pred` is its argmax."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    try:
        if y_proba.shape[1] == 2:
            metrics["auc"] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            metrics["auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        # e.g. a class has zero samples in this split/batch — AUC is undefined
        metrics["auc"] = float("nan")
    return metrics
