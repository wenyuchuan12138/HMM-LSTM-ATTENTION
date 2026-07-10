from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .utils import mape


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true.ravel(), y_pred.ravel())),
        "RMSE": float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel()))),
        "MAPE": mape(y_true.ravel(), y_pred.ravel()),
        "R2": float(r2_score(y_true.ravel(), y_pred.ravel())),
    }


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        out["ROC_AUC"] = float(roc_auc_score(y_true, prob))
        out["PR_AUC"] = float(average_precision_score(y_true, prob))
    else:
        out["ROC_AUC"] = float("nan")
        out["PR_AUC"] = float("nan")
    return out


def horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> pd.DataFrame:
    rows = []
    for h in range(y_true.shape[1]):
        rows.append(
            {
                "model": model_name,
                "horizon": h + 1,
                "MAE": float(mean_absolute_error(y_true[:, h], y_pred[:, h])),
                "RMSE": float(np.sqrt(mean_squared_error(y_true[:, h], y_pred[:, h]))),
            }
        )
    return pd.DataFrame(rows)


def confusion_counts(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return confusion_matrix(y_true, (prob >= threshold).astype(int), labels=[0, 1])
