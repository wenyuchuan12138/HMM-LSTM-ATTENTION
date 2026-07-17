from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAW_FILES = {
    "carbon": "carbon_intensity.csv",
    "renewable": "renewable_percentage.csv",
    "carbon_free": "carbon_free_percentage.csv",
    "load": "total_load.csv",
    "mix": "electricity_mix.csv",
}


def _find_time_col(columns: list[str]) -> str:
    for candidate in ["timestamp", "datetime", "dateTime", "time"]:
        if candidate in columns:
            return candidate
    raise ValueError(f"Cannot identify time column. Available columns: {columns}")


def _dedupe_by_timestamp(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if "updated_at" in df.columns:
        df["_updated_sort"] = pd.to_datetime(df["updated_at"], errors="coerce", utc=True)
        df = df.sort_values([time_col, "_updated_sort"])
        df = df.drop(columns=["_updated_sort"])
    else:
        df = df.sort_values(time_col)
    return df.drop_duplicates(time_col, keep="last")


def profile_csvs(raw_dir: Path, output_path: Path) -> pd.DataFrame:
    rows = []
    for source, filename in RAW_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            rows.append({"source": source, "file": filename, "exists": False})
            continue
        df = pd.read_csv(path)
        time_col = _find_time_col(df.columns.tolist())
        ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        rows.append(
            {
                "source": source,
                "file": filename,
                "exists": True,
                "rows": len(df),
                "columns": "|".join(df.columns),
                "time_col": time_col,
                "start_time": ts.min(),
                "end_time": ts.max(),
                "missing_ratio_max": float(df.isna().mean().max()),
                "duplicate_timestamps": int(ts.duplicated().sum()),
            }
        )
    profile = pd.DataFrame(rows)
    profile.to_csv(output_path, index=False)
    print(profile.to_string(index=False))
    return profile


def _quality_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keep = [c for c in ["is_estimated", "estimation_method", "temporal_granularity", "updated_at", "created_at"] if c in df.columns]
    renamed = {c: f"{prefix}_{c}" for c in keep}
    return df[["timestamp"] + keep].rename(columns=renamed)


def load_and_merge(raw_dir: Path, start_time: str, end_time: str, output_path: Path) -> pd.DataFrame:
    loaded: dict[str, pd.DataFrame] = {}
    for source, filename in RAW_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            if source == "carbon":
                raise FileNotFoundError(f"Required carbon intensity file missing: {path}")
            print(f"[WARN] Optional file missing and skipped: {path}")
            continue
        df = pd.read_csv(path)
        time_col = _find_time_col(df.columns.tolist())
        df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df = _dedupe_by_timestamp(df, "timestamp")
        df = df[(df["timestamp"] >= pd.Timestamp(start_time)) & (df["timestamp"] <= pd.Timestamp(end_time))]
        loaded[source] = df.sort_values("timestamp").reset_index(drop=True)

    carbon = loaded["carbon"]
    if "carbon_intensity_gCO2eq_per_kWh" not in carbon.columns:
        raise ValueError(f"Core carbon intensity column missing. Available columns: {carbon.columns.tolist()}")

    base_cols = ["timestamp", "zone_id", "carbon_intensity_gCO2eq_per_kWh"]
    merged = carbon[base_cols].merge(_quality_columns(carbon, "carbon"), on="timestamp", how="left")

    for source, value_col in [
        ("renewable", "renewable_percentage"),
        ("carbon_free", "carbon_free_percentage"),
        ("load", "total_load_MW"),
    ]:
        if source not in loaded:
            continue
        df = loaded[source]
        cols = ["timestamp"] + ([value_col] if value_col in df.columns else [])
        merged = merged.merge(df[cols], on="timestamp", how="left")
        merged = merged.merge(_quality_columns(df, source), on="timestamp", how="left")

    if "mix" in loaded:
        mix = loaded["mix"]
        numeric_mix = [c for c in mix.columns if c.startswith("production_") or c.startswith("consumption_")]
        merged = merged.merge(mix[["timestamp"] + numeric_mix], on="timestamp", how="left")
        merged = merged.merge(_quality_columns(mix, "mix"), on="timestamp", how="left")

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    merged[numeric_cols] = merged[numeric_cols].interpolate(limit_direction="both").ffill().bfill()
    merged.to_csv(output_path, index=False)
    return merged


def profile_combined_csv(path: Path, output_path: Path) -> pd.DataFrame:
    """检查2017-2025宽表的时间范围、重复和缺失情况。"""
    df = pd.read_csv(path, low_memory=False)
    time_col = _find_time_col(df.columns.tolist())
    timestamp = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    profile = pd.DataFrame(
        [
            {
                "source": "combined_17_25",
                "file": path.name,
                "exists": True,
                "rows": len(df),
                "columns": "|".join(df.columns),
                "time_col": time_col,
                "start_time": timestamp.min(),
                "end_time": timestamp.max(),
                "missing_ratio_max": float(df.isna().mean().max()),
                "duplicate_timestamps": int(timestamp.duplicated().sum()),
            }
        ]
    )
    profile.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(profile.to_string(index=False))
    return profile


def load_combined_csv(path: Path, start_time: str, end_time: str, output_path: Path) -> pd.DataFrame:
    """读取已合并的宽表，保留当前模型需要的全部原始字段。"""
    df = pd.read_csv(path, low_memory=False)
    time_col = _find_time_col(df.columns.tolist())
    df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = _dedupe_by_timestamp(df, "timestamp")
    df = df[(df["timestamp"] >= pd.Timestamp(start_time)) & (df["timestamp"] <= pd.Timestamp(end_time))]
    df = df.sort_values("timestamp").reset_index(drop=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(limit_direction="both").ffill().bfill()
    df.to_csv(output_path, index=False)
    return df
