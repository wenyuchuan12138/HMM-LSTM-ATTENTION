from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class SequenceData:
    # Standardized input arrays used by the original B0--B6 experiments.
    short: np.ndarray
    long: np.ndarray
    # Raw causal probabilities used only by the new Markovian experiment.
    short_hmm: np.ndarray
    long_hmm: np.ndarray
    long_static: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    base: np.ndarray
    y_reg: np.ndarray
    y_cls: np.ndarray
    rows: np.ndarray


@dataclass
class PreparedData:
    train: SequenceData
    val: SequenceData
    test: SequenceData
    feature_cols: list[str]                 # Original B0--B6 feature list: unchanged.
    hmm_prob_cols: list[str]
    markov_feature_cols: list[str]          # Removes every hmm_* feature for the new experiment.
    seasonal_cols: list[str]
    residual_cols: list[str]
    long_static_cols: list[str]
    scaler: StandardScaler
    high_threshold: float


def split_indices_by_time(df: pd.DataFrame, train_end_time: str, val_end_time: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return (
        np.flatnonzero(ts <= pd.Timestamp(train_end_time)),
        np.flatnonzero((ts > pd.Timestamp(train_end_time)) & (ts <= pd.Timestamp(val_end_time))),
        np.flatnonzero(ts > pd.Timestamp(val_end_time)),
    )


def prepare_sequences(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    horizon: int,
    short_window: int,
    long_window: int,
    high_quantile: float,
) -> PreparedData:
    hmm_prob_cols = [c for c in df.columns if c.startswith("hmm_prob_")]
    seasonal_cols = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "year_sin", "year_cos", "same_hour_last_year"]
    residual_cols = ["carbon_intensity_gCO2eq_per_kWh", "ci_lag_1h", "ci_lag_24h", "ci_lag_168h"]
    long_static_cols = ["ci_lag_24h", "ci_lag_48h", "ci_lag_72h", "ci_lag_168h", "same_hour_7d_mean", "same_hour_7d_std", "ci_rolling_mean_720h", "ci_rolling_std_720h", "year_sin", "year_cos"]
    optional = ["renewable_percentage", "carbon_free_percentage", "total_load_MW", "fossil_share", "coal_share", "gas_share", "wind_share", "solar_share", "hydro_share", "nuclear_share", "ci_rolling_mean_6h", "ci_rolling_mean_24h", "ci_rolling_mean_168h", "ci_rolling_std_24h", "ci_rolling_std_168h", "ci_diff_1h", "ci_diff_6h", "hmm_high_prob", "hmm_transition_to_high", "hmm_entropy", "hmm_state_duration"]

    # Kept exactly for legacy B0--B6 comparability.
    feature_cols = list(dict.fromkeys([c for c in residual_cols + seasonal_cols + long_static_cols + [c for c in optional if c in df.columns] + hmm_prob_cols if c in df.columns]))
    # New Markovian experiment receives probabilities through its switching
    # mechanism only; it must not see them again as ordinary numeric features.
    markov_feature_cols = [c for c in feature_cols if not c.startswith("hmm_")]
    seasonal_cols = [c for c in seasonal_cols if c in df.columns]
    residual_cols = [c for c in residual_cols if c in df.columns]
    long_static_cols = [c for c in long_static_cols if c in df.columns]

    scaler = StandardScaler()
    x_scaled = np.zeros((len(df), len(feature_cols)), dtype=np.float32)
    x_scaled[train_idx] = scaler.fit_transform(df.loc[train_idx, feature_cols])
    future_idx = np.concatenate([val_idx, test_idx])
    x_scaled[future_idx] = scaler.transform(df.loc[future_idx, feature_cols])

    # HMM probabilities deliberately stay in [0, 1] and sum to 1.  They are not
    # standardized because they are mixture weights, not ordinary covariates.
    hmm_raw = df[hmm_prob_cols].to_numpy(dtype=np.float32)
    seasonal_idx = [feature_cols.index(c) for c in seasonal_cols]
    residual_idx = [feature_cols.index(c) for c in residual_cols]
    long_static_idx = [feature_cols.index(c) for c in long_static_cols]
    ci = df["carbon_intensity_gCO2eq_per_kWh"].to_numpy(dtype=np.float32)
    high_threshold = float(df.loc[train_idx, "carbon_intensity_gCO2eq_per_kWh"].quantile(high_quantile))

    def build(indices: np.ndarray) -> SequenceData:
        xs, xl, xhs, xhl, ls, seas, resid, bases, yr, yc, rows = [], [], [], [], [], [], [], [], [], [], []
        start, end = int(indices[0]), int(indices[-1])
        for row in range(max(start, long_window - 1), end - horizon + 1):
            # Windows do not cross split boundaries; this preserves the original experiment.
            if row - long_window + 1 < start:
                continue
            future = ci[row + 1 : row + horizon + 1]
            xs.append(x_scaled[row - short_window + 1 : row + 1])
            xl.append(x_scaled[row - long_window + 1 : row + 1])
            xhs.append(hmm_raw[row - short_window + 1 : row + 1])
            xhl.append(hmm_raw[row - long_window + 1 : row + 1])
            ls.append(x_scaled[row, long_static_idx])
            seas.append(x_scaled[row, seasonal_idx])
            resid.append(x_scaled[row, residual_idx])
            bases.append(ci[row - 23 : row - 23 + horizon])
            yr.append(future)
            yc.append(float((ci[row] < high_threshold) and (future.max() >= high_threshold)))
            rows.append(row)
        return SequenceData(
            np.asarray(xs, dtype=np.float32), np.asarray(xl, dtype=np.float32),
            np.asarray(xhs, dtype=np.float32), np.asarray(xhl, dtype=np.float32),
            np.asarray(ls, dtype=np.float32), np.asarray(seas, dtype=np.float32),
            np.asarray(resid, dtype=np.float32), np.asarray(bases, dtype=np.float32),
            np.asarray(yr, dtype=np.float32), np.asarray(yc, dtype=np.float32),
            np.asarray(rows, dtype=np.int64),
        )

    return PreparedData(build(train_idx), build(val_idx), build(test_idx), feature_cols, hmm_prob_cols, markov_feature_cols, seasonal_cols, residual_cols, long_static_cols, scaler, high_threshold)
