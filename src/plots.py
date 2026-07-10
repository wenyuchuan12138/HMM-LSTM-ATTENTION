from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .evaluate import confusion_counts


def save_figures(
    df: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    cm: np.ndarray,
    out_dir: Path,
    transition_matrix: list[list[float]] | None = None,
    attention: np.ndarray | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    sample = df.iloc[: min(len(df), 24 * 60)]
    plt.figure(figsize=(13, 5))
    sns.scatterplot(data=sample, x="timestamp", y="carbon_intensity_gCO2eq_per_kWh", hue="hmm_state", palette="viridis", s=12)
    plt.title("HMM states on carbon intensity")
    plt.ylabel("gCO2eq/kWh")
    plt.tight_layout()
    plt.savefig(out_dir / "hmm_states.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    order = metrics.sort_values("MAE")
    sns.barplot(data=order, x="MAE", y="model")
    plt.title("Model comparison by MAE")
    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison_mae.png", dpi=180)
    plt.savefig(out_dir / "branch_metric_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No transition", "Transition"], yticklabels=["No transition", "Transition"])
    plt.title("Transition warning confusion matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    if {"y_true_h1", "pred_h1"}.issubset(predictions.columns):
        subset = predictions.head(24 * 14)
        plt.figure(figsize=(13, 5))
        plt.plot(subset["timestamp"], subset["y_true_h1"], label="True h+1")
        plt.plot(subset["timestamp"], subset["pred_h1"], label="Pred h+1")
        plt.title("Test prediction curve")
        plt.ylabel("gCO2eq/kWh")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "prediction_curve.png", dpi=180)
        plt.close()

        err_cols = [c for c in predictions.columns if c.startswith("pred_h")]
        true_cols = [c for c in predictions.columns if c.startswith("y_true_h")]
        errors = predictions[err_cols].to_numpy() - predictions[true_cols].to_numpy()
        plt.figure(figsize=(8, 4.5))
        sns.histplot(errors.ravel(), bins=50, kde=True)
        plt.title("Forecast error distribution")
        plt.xlabel("Prediction error (gCO2eq/kWh)")
        plt.tight_layout()
        plt.savefig(out_dir / "error_distribution.png", dpi=180)
        plt.close()

    if transition_matrix is not None:
        plt.figure(figsize=(6, 5))
        sns.heatmap(np.asarray(transition_matrix), annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title("HMM transition probability matrix")
        plt.xlabel("Next state")
        plt.ylabel("Current state")
        plt.tight_layout()
        plt.savefig(out_dir / "hmm_transition_matrix.png", dpi=180)
        plt.close()

    if attention is not None and attention.size:
        plt.figure(figsize=(8, 4.5))
        avg_attention = attention.mean(axis=0)
        plt.bar(np.arange(-len(avg_attention) + 1, 1), avg_attention)
        plt.title("Average short-window attention weights")
        plt.xlabel("Lag hour relative to forecast origin")
        plt.ylabel("Attention weight")
        plt.tight_layout()
        plt.savefig(out_dir / "attention_weights.png", dpi=180)
        plt.close()
