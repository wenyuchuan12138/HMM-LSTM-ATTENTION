from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["hour"] = out["timestamp"].dt.hour
    out["weekday"] = out["timestamp"].dt.dayofweek
    out["day_of_year"] = out["timestamp"].dt.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["weekday_sin"] = np.sin(2 * np.pi * out["weekday"] / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * out["weekday"] / 7)
    out["year_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["year_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)

    production_cols = [c for c in out.columns if c.startswith("production_") and c.endswith("_MW")]
    if production_cols:
        out["production_total_MW"] = out[production_cols].sum(axis=1).replace(0, np.nan)
        for fuel in ["coal", "gas", "wind", "solar", "hydro", "nuclear", "biomass", "unknown"]:
            col = f"production_{fuel}_MW"
            if col in out.columns:
                out[f"{fuel}_share"] = out[col] / out["production_total_MW"]
        fossil = [c for c in ["production_coal_MW", "production_gas_MW", "production_oil_MW"] if c in out.columns]
        if fossil:
            out["fossil_share"] = out[fossil].sum(axis=1) / out["production_total_MW"]

    ci = out["carbon_intensity_gCO2eq_per_kWh"]
    for lag in [1, 6, 24, 48, 72, 168]:
        out[f"ci_lag_{lag}h"] = ci.shift(lag)
    for win in [6, 24, 168, 720]:
        out[f"ci_rolling_mean_{win}h"] = ci.shift(1).rolling(win).mean()
    for win in [24, 168, 720]:
        out[f"ci_rolling_std_{win}h"] = ci.shift(1).rolling(win).std()
    out["ci_diff_1h"] = ci.diff(1)
    out["ci_diff_6h"] = ci.diff(6)
    same_hour_values = pd.concat([ci.shift(24 * d) for d in range(1, 8)], axis=1)
    out["same_hour_7d_mean"] = same_hour_values.mean(axis=1)
    out["same_hour_7d_std"] = same_hour_values.std(axis=1)

    # 用时间键逐行查找去年同小时，避免闰年2月29日映射产生重复时间戳。
    carbon_by_time = pd.Series(ci.to_numpy(), index=out["timestamp"])
    previous_year = out["timestamp"] - pd.DateOffset(years=1)
    out["same_hour_last_year"] = carbon_by_time.reindex(previous_year).to_numpy()
    out["same_hour_last_year"] = out["same_hour_last_year"].fillna(out["ci_lag_168h"])

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    # 只允许使用过去值填补；bfill会把未来信息带入初始滞后窗口。
    out[numeric_cols] = out[numeric_cols].ffill()
    return out.dropna(subset=["ci_lag_168h", "ci_rolling_mean_720h"]).reset_index(drop=True)
