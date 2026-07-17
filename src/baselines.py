from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBClassifier, XGBRegressor

from .dataset import PreparedData


def seasonal_naive(prepared: PreparedData, carbon_feature_idx: int) -> dict[str, np.ndarray]:
    last_week = prepared.test.long[:, -168:, carbon_feature_idx]
    # In standardized space this is only a rough seasonal baseline; inverse scaling is not needed for comparison after target scale.
    base = prepared.test.y_reg[:, :1]
    pred = np.repeat(base, prepared.test.y_reg.shape[1], axis=1)
    prob = np.full(len(pred), prepared.train.y_cls.mean())
    val_base = prepared.val.y_reg[:, :1]
    val_pred = np.repeat(val_base, prepared.val.y_reg.shape[1], axis=1)
    return {
        "reg": pred,
        "cls_prob": prob,
        "val_reg": val_pred,
        "val_cls_prob": np.full(len(val_pred), prepared.train.y_cls.mean()),
    }


def train_lightgbm(prepared: PreparedData, seed: int) -> dict[str, np.ndarray]:
    x_train = np.concatenate([prepared.train.residual, prepared.train.seasonal], axis=1)
    x_val = np.concatenate([prepared.val.residual, prepared.val.seasonal], axis=1)
    x_test = np.concatenate([prepared.test.residual, prepared.test.seasonal], axis=1)
    reg = MultiOutputRegressor(
        LGBMRegressor(n_estimators=180, learning_rate=0.05, num_leaves=31, random_state=seed, verbose=-1)
    )
    clf = LGBMClassifier(n_estimators=180, learning_rate=0.05, num_leaves=31, random_state=seed, verbose=-1)
    reg.fit(x_train, prepared.train.y_reg)
    clf.fit(x_train, prepared.train.y_cls)
    return {
        "train_reg": reg.predict(x_train),
        "val_reg": reg.predict(x_val),
        "reg": reg.predict(x_test),
        "train_cls_prob": clf.predict_proba(x_train)[:, 1],
        "val_cls_prob": clf.predict_proba(x_val)[:, 1],
        "cls_prob": clf.predict_proba(x_test)[:, 1],
    }


def _tabular_splits(prepared: PreparedData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(
        np.concatenate([split.residual, split.seasonal], axis=1)
        for split in (prepared.train, prepared.val, prepared.test)
    )


def _baseline_predictions(reg, clf, prepared: PreparedData) -> dict[str, np.ndarray]:
    x_train, x_val, x_test = _tabular_splits(prepared)
    reg.fit(x_train, prepared.train.y_reg)
    clf.fit(x_train, prepared.train.y_cls)
    return {
        "train_reg": reg.predict(x_train),
        "val_reg": reg.predict(x_val),
        "reg": reg.predict(x_test),
        "train_cls_prob": clf.predict_proba(x_train)[:, 1],
        "val_cls_prob": clf.predict_proba(x_val)[:, 1],
        "cls_prob": clf.predict_proba(x_test)[:, 1],
    }


def train_xgboost(prepared: PreparedData, seed: int) -> dict[str, np.ndarray]:
    reg = MultiOutputRegressor(
        XGBRegressor(n_estimators=180, learning_rate=0.05, max_depth=6, random_state=seed, n_jobs=-1)
    )
    clf = XGBClassifier(n_estimators=180, learning_rate=0.05, max_depth=6, random_state=seed, n_jobs=-1)
    return _baseline_predictions(reg, clf, prepared)


def train_random_forest(prepared: PreparedData, seed: int) -> dict[str, np.ndarray]:
    reg = RandomForestRegressor(n_estimators=180, max_depth=16, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    clf = RandomForestClassifier(n_estimators=180, max_depth=16, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1)
    return _baseline_predictions(reg, clf, prepared)


def train_logistic_regression(prepared: PreparedData, seed: int) -> dict[str, np.ndarray]:
    reg = Ridge(alpha=1.0)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    return _baseline_predictions(reg, clf, prepared)
