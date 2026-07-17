from __future__ import annotations

import copy
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
)
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig
from .dataset import PreparedData
from .models import LGBMLogitResidualLSTM


def _clip_probability(prob: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(prob, dtype=np.float64), 1e-6, 1 - 1e-6)


def _logit(prob: np.ndarray) -> np.ndarray:
    prob = _clip_probability(prob)
    return np.log(prob / (1 - prob))


def probability_quality_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    """计算概率质量指标；ECE 使用文档要求的 10 个等频箱。"""
    y_true = np.asarray(y_true, dtype=int)
    prob = _clip_probability(prob)
    groups = np.array_split(np.argsort(prob), 10)
    ece = sum(
        len(group) / len(y_true) * abs(float(prob[group].mean()) - float(y_true[group].mean()))
        for group in groups
        if len(group)
    )

    # 以预测概率的 logit 为自变量，理想截距为 0、斜率为 1。
    calibration = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    calibration.fit(_logit(prob).reshape(-1, 1), y_true)
    return {
        "LogLoss": float(log_loss(y_true, prob)),
        "Brier": float(brier_score_loss(y_true, prob)),
        "ECE": float(ece),
        "CalibrationSlope": float(calibration.coef_[0, 0]),
        "CalibrationIntercept": float(calibration.intercept_[0]),
    }


def save_probability_figures(y_true, predictions, metrics, figures_dir: Path) -> None:
    """保存新增概率模型的可靠性曲线和核心概率指标对比图。"""
    import matplotlib.pyplot as plt

    names = [
        "LightGBM",
        "LightGBM + Platt",
        "LightGBM + Isotonic",
        "LGBM-LR-LSTM",
        "LGBM-LR-LSTM + Platt",
    ]
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Ideal")
    for name in names:
        observed, predicted = calibration_curve(
            y_true,
            predictions[name]["cls_prob"],
            n_bins=10,
            strategy="quantile",
        )
        plt.plot(predicted, observed, marker="o", label=name)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive rate")
    plt.title("Probability calibration comparison")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "probability_calibration_curve.png", dpi=180)
    plt.close()

    plot_data = metrics[metrics["model"].isin(names)].set_index("model")[["LogLoss", "Brier", "ECE"]]
    plot_data.plot(kind="bar", figsize=(9, 5))
    plt.ylabel("Lower is better")
    plt.title("Probability quality metrics")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "probability_quality_comparison.png", dpi=180)
    plt.close()


def _tabular_features(prepared: PreparedData):
    return tuple(
        np.concatenate([split.residual, split.seasonal], axis=1)
        for split in (prepared.train, prepared.val, prepared.test)
    )


def _state_layout(prepared: PreparedData):
    # HMM 离散状态编号不进入网络；这里只使用因果概率和连续状态统计量。
    base_idx = [prepared.feature_cols.index(col) for col in prepared.markov_feature_cols]
    aux_names = ["hmm_entropy", "hmm_transition_to_high", "hmm_state_duration"]
    aux_idx = [prepared.feature_cols.index(col) for col in aux_names if col in prepared.feature_cols]

    return base_idx, aux_idx


class _ResidualDataset(Dataset):
    def __init__(self, split, base_idx, aux_idx, base_prob, labels, indices=None):
        self.split = split
        self.base_idx = base_idx
        self.aux_idx = aux_idx
        self.base_logit = _logit(base_prob).astype(np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.indices = np.arange(len(self.labels)) if indices is None else np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = self.indices[item]
        short_state = [self.split.short_hmm[idx]]
        long_state = [self.split.long_hmm[idx]]
        if self.aux_idx:
            short_state.append(self.split.short[idx][:, self.aux_idx])
            long_state.append(self.split.long[idx][:, self.aux_idx])
        short_state = np.concatenate(short_state, axis=1).astype(np.float32)
        long_state = np.concatenate(long_state, axis=1).astype(np.float32)
        short = np.concatenate([self.split.short[idx][:, self.base_idx], short_state], axis=1).astype(np.float32)
        long = np.concatenate([self.split.long[idx][:, self.base_idx], long_state], axis=1).astype(np.float32)
        return (
            torch.from_numpy(short),
            torch.from_numpy(long),
            torch.from_numpy(short_state[-1]),
            torch.tensor(self.base_logit[item] if len(self.base_logit) == len(self.indices) else self.base_logit[idx]),
            torch.tensor(self.labels[idx]),
        )


def _predict_residual(model, loader, device):
    probs, deltas = [], []
    model.eval()
    with torch.no_grad():
        for short, long, state, base_logit, _ in loader:
            short, long = short.to(device), long.to(device)
            state, base_logit = state.to(device), base_logit.to(device)
            delta = model(short, long, state, base_logit)
            probs.append(torch.sigmoid(base_logit + delta).cpu().numpy())
            deltas.append(delta.cpu().numpy())
    return np.concatenate(probs), np.concatenate(deltas)


def _rolling_oof_probabilities(x_train, y_train, seed: int):
    """扩展时间窗口 OOF，保证二级 LSTM 看不到基模型的训练内概率。"""
    oof = np.full(len(y_train), np.nan, dtype=np.float64)
    fold_rows = []
    for fold, (fit_idx, holdout_idx) in enumerate(TimeSeriesSplit(n_splits=5).split(x_train), start=1):
        model = LGBMClassifier(
            n_estimators=180,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            verbose=-1,
        )
        model.fit(x_train[fit_idx], y_train[fit_idx])
        oof[holdout_idx] = model.predict_proba(x_train[holdout_idx])[:, 1]
        fold_rows.append({"fold": fold, "fit_rows": len(fit_idx), "holdout_rows": len(holdout_idx)})
    return oof, fold_rows


def train_probability_models(
    prepared: PreparedData,
    lightgbm_pred: dict[str, np.ndarray],
    cfg: ExperimentConfig,
):
    """训练 Platt、Isotonic、logit 残差 LSTM 及其 Platt 后校准版本。"""
    started = time.perf_counter()
    y_train = prepared.train.y_cls.astype(int)
    y_val = prepared.val.y_cls.astype(int)
    x_train, _, _ = _tabular_features(prepared)

    val_score = _logit(lightgbm_pred["val_cls_prob"])
    test_score = _logit(lightgbm_pred["cls_prob"])
    platt = LogisticRegression(solver="lbfgs", random_state=cfg.seed)
    platt.fit(val_score.reshape(-1, 1), y_val)
    isotonic = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    isotonic.fit(val_score, y_val)

    oof_prob, fold_rows = _rolling_oof_probabilities(x_train, y_train, cfg.seed)
    valid = np.flatnonzero(np.isfinite(oof_prob))
    base_idx, aux_idx = _state_layout(prepared)
    train_data = _ResidualDataset(prepared.train, base_idx, aux_idx, oof_prob[valid], y_train, valid)
    val_data = _ResidualDataset(prepared.val, base_idx, aux_idx, lightgbm_pred["val_cls_prob"], y_val)
    test_data = _ResidualDataset(prepared.test, base_idx, aux_idx, lightgbm_pred["cls_prob"], prepared.test.y_cls)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = len(prepared.hmm_prob_cols) + len(aux_idx)
    input_dim = len(base_idx) + state_dim
    model = LGBMLogitResidualLSTM(input_dim, state_dim, cfg.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    positive = max(float(y_train[valid].sum()), 1.0)
    negative = max(float(len(valid) - y_train[valid].sum()), 1.0)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device))
    best_state, best_loss, stale, history = None, float("inf"), 0, []

    for epoch in range(1, 21):
        model.train()
        losses = []
        for short, long, state, base_logit, labels in train_loader:
            short, long, state = short.to(device), long.to(device), state.to(device)
            base_logit, labels = base_logit.to(device), labels.to(device)
            optimizer.zero_grad()
            delta = model(short, long, state, base_logit)
            loss = bce(base_logit + delta, labels) + 0.01 * torch.mean(delta ** 2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_prob, _ = _predict_residual(model, val_loader, device)
        val_loss = float(log_loss(y_val, _clip_probability(val_prob)))
        history.append({
            "model": "LGBM-LR-LSTM",
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_logloss": val_loss,
            "val_pr_auc": float(average_precision_score(y_val, val_prob)),
        })
        if val_loss < best_loss - 1e-4:
            best_loss, stale = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= 4:
                break

    model.load_state_dict(best_state)
    val_lr, val_delta = _predict_residual(model, val_loader, device)
    test_lr, test_delta = _predict_residual(model, test_loader, device)

    post_platt = LogisticRegression(solver="lbfgs", random_state=cfg.seed)
    post_platt.fit(_logit(val_lr).reshape(-1, 1), y_val)
    val_lr_platt = post_platt.predict_proba(_logit(val_lr).reshape(-1, 1))[:, 1]
    test_lr_platt = post_platt.predict_proba(_logit(test_lr).reshape(-1, 1))[:, 1]

    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(platt, cfg.artifacts_dir / "lightgbm_platt_calibrator.pkl")
    joblib.dump(isotonic, cfg.artifacts_dir / "lightgbm_isotonic_calibrator.pkl")
    joblib.dump(post_platt, cfg.artifacts_dir / "lgbm_lr_lstm_platt_calibrator.pkl")
    torch.save(
        {"state_dict": model.state_dict(), "input_dim": input_dim, "state_dim": state_dim},
        cfg.artifacts_dir / "LGBM-LR-LSTM.pt",
    )

    regression = lightgbm_pred["reg"]
    val_regression = lightgbm_pred["val_reg"]
    predictions = {
        "LightGBM + Platt": {"reg": regression, "cls_prob": platt.predict_proba(test_score.reshape(-1, 1))[:, 1]},
        "LightGBM + Isotonic": {"reg": regression, "cls_prob": isotonic.predict(test_score)},
        "LGBM-LR-LSTM": {"reg": regression, "cls_prob": test_lr},
        "LGBM-LR-LSTM + Platt": {"reg": regression, "cls_prob": test_lr_platt},
    }
    val_predictions = {
        "LightGBM + Platt": {"reg": val_regression, "cls_prob": platt.predict_proba(val_score.reshape(-1, 1))[:, 1]},
        "LightGBM + Isotonic": {"reg": val_regression, "cls_prob": isotonic.predict(val_score)},
        "LGBM-LR-LSTM": {"reg": val_regression, "cls_prob": val_lr},
        "LGBM-LR-LSTM + Platt": {"reg": val_regression, "cls_prob": val_lr_platt},
    }
    diagnostics = {
        "model": "LGBM-LR-LSTM",
        "delta_logit_mean": float(test_delta.mean()),
        "delta_logit_std": float(test_delta.std()),
        "delta_logit_min": float(test_delta.min()),
        "delta_logit_max": float(test_delta.max()),
        "mean_absolute_probability_change": float(np.mean(np.abs(test_lr - lightgbm_pred["cls_prob"]))),
        "oof_training_rows": int(len(valid)),
        "epochs": int(len(history)),
        "best_val_logloss": float(best_loss),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    runtime = {
        "model": "LGBM-LR-LSTM",
        "seed": cfg.seed,
        "parameters": diagnostics["parameters"],
        "train_seconds": time.perf_counter() - started,
    }
    return predictions, val_predictions, pd.DataFrame(history), diagnostics, runtime, fold_rows
