from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

from .dataset import PreparedData


def seasonal_naive(prepared: PreparedData, carbon_feature_idx: int) -> dict[str, np.ndarray]:
    last_week = prepared.test.long[:, -168:, carbon_feature_idx]
    # In standardized space this is only a rough seasonal baseline; inverse scaling is not needed for comparison after target scale.
    base = prepared.test.y_reg[:, :1]
    pred = np.repeat(base, prepared.test.y_reg.shape[1], axis=1)
    prob = np.full(len(pred), prepared.train.y_cls.mean())
    return {"reg": pred, "cls_prob": prob}


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
