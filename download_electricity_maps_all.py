from __future__ import annotations

"""
Electricity Maps 中国小时级多信号历史数据下载器
========================================

默认下载 2022-01-01 至 2026-01-01 的中国 CN 小时级数据：

核心信号：
- Carbon Intensity
- Renewable Percentage
- Carbon-Free Percentage
- Total Load
- Net Load
- Electricity Mix / Power Breakdown

特点：
- 自动尝试兼容 endpoint；
- 3~5 天分段；
- 网络重试与指数退避；
- 每段立即保存；
- 中断后自动续传；
- 最终按 timestamp 合并；
- 自动生成质量报告。

运行前：
    pip install -r requirements.txt

PowerShell：
    $env:ELECTRICITY_MAPS_API_KEY="你的 API Key"
    python download_electricity_maps_all.py
"""

import argparse
import json
import os
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://api.electricitymaps.com/v4"

SIGNALS: dict[str, dict[str, Any]] = {
    "carbon_intensity": {
        "endpoints": ["carbon-intensity/past-range"],
        "keys": ["carbonIntensity", "carbon_intensity", "value"],
        "column": "carbon_intensity_gCO2eq_per_kWh",
        "required": True,
        "params": {"emissionFactorType": "lifecycle"},
        "kind": "scalar",
    },
    "renewable_percentage": {
        "endpoints": [
            "renewable-energy/past-range",
            "renewable-percentage/past-range",
        ],
        "keys": [
            "renewableEnergyPercentage",
            "renewablePercentage",
            "renewable_percentage",
            "value",
        ],
        "column": "renewable_percentage",
        "required": False,
        "params": {},
        "kind": "scalar",
    },
    "carbon_free_percentage": {
        "endpoints": [
            "carbon-free-energy/past-range",
            "carbon-free-percentage/past-range",
        ],
        "keys": [
            "carbonFreeEnergyPercentage",
            "carbonFreePercentage",
            "carbon_free_percentage",
            "value",
        ],
        "column": "carbon_free_percentage",
        "required": False,
        "params": {},
        "kind": "scalar",
    },
    "total_load": {
        "endpoints": ["total-load/past-range", "load/past-range"],
        "keys": ["totalLoad", "load", "value"],
        "column": "total_load_MW",
        "required": False,
        "params": {},
        "kind": "scalar",
    },
    "net_load": {
        "endpoints": ["net-load/past-range"],
        "keys": ["netLoad", "net_load", "value"],
        "column": "net_load_MW",
        "required": False,
        "params": {},
        "kind": "scalar",
    },
    "electricity_mix": {
        "endpoints": [
            "power-breakdown/past-range",
            "electricity-mix/past-range",
        ],
        "keys": [],
        "column": None,
        "required": False,
        "params": {},
        "kind": "mix",
    },
    "fossil_only_carbon_intensity": {
        "endpoints": [
            "fossil-only-carbon-intensity/past-range",
        ],
        "keys": [
            "fossilOnlyCarbonIntensity",
            "fossil_only_carbon_intensity",
            "value",
        ],
        "column": "fossil_only_carbon_intensity_gCO2eq_per_kWh",
        "required": False,
        "params": {"emissionFactorType": "lifecycle"},
        "kind": "scalar",
    },
}


class EndpointUnavailable(RuntimeError):
    pass


def parse_iso(value: str) -> datetime:
    value = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def make_session(api_key: str) -> requests.Session:
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        status=8,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=2, pool_maxsize=2)
    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update(
        {
            "auth-token": api_key,
            "Accept": "application/json",
            "User-Agent": "academic-electricity-maps-downloader/1.0",
            "Connection": "close",
        }
    )
    return session


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def atomic_save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp, index=False)
    temp.replace(path)


def append_rows(path: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    old = read_csv(path)
    if not rows:
        return old

    new = pd.DataFrame(rows)
    new["timestamp"] = pd.to_datetime(new["timestamp"], utc=True, errors="coerce")
    df = pd.concat([old, new], ignore_index=True)
    df = (
        df.dropna(subset=["timestamp"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    atomic_save(df, path)
    return df


def write_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(f"{datetime.now().isoformat(timespec='seconds')} {message}\n")


def request_json(
    session: requests.Session,
    endpoint: str,
    params: dict[str, Any],
    max_retries: int = 8,
) -> dict[str, Any]:
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, params=params, timeout=(25, 150))

            if response.status_code in (403, 404):
                raise EndpointUnavailable(
                    f"HTTP {response.status_code}: {response.text[:300]}"
                )

            if response.status_code >= 400:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:500]}"
                )

            return response.json()

        except EndpointUnavailable:
            raise

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
            json.JSONDecodeError,
            RuntimeError,
        ) as exc:
            last_error = exc
            wait = min(180.0, 2 ** (attempt - 1) + random.uniform(0.5, 2.5))
            print(f"请求失败 {attempt}/{max_retries}: {exc}")
            if attempt < max_retries:
                print(f"等待 {wait:.1f} 秒后重试")
                time.sleep(wait)

    raise RuntimeError(f"多次重试仍失败：{endpoint}, {params}") from last_error


def payload_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data: Any = payload.get("data", payload)

    if isinstance(data, dict):
        for key in ("history", "values", "data"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]

    if isinstance(data, list):
        return data

    raise ValueError(f"无法识别 API 响应结构：{type(data)}")


def extract_scalar(payload: dict[str, Any], keys: list[str]) -> list[dict[str, Any]]:
    rows = []

    for item in payload_items(payload):
        value = next((item[key] for key in keys if key in item), None)
        rows.append(
            {
                "timestamp": item.get("datetime") or item.get("timestamp"),
                "value": value,
                "zone_id": item.get("zone"),
                "is_estimated": item.get("isEstimated"),
                "estimation_method": item.get("estimationMethod"),
                "temporal_granularity": item.get("temporalGranularity"),
                "updated_at": item.get("updatedAt"),
                "created_at": item.get("createdAt"),
                "emission_factor_type": item.get("emissionFactorType"),
            }
        )

    return rows


def safe_name(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def extract_mix(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []

    for item in payload_items(payload):
        row: dict[str, Any] = {
            "timestamp": item.get("datetime") or item.get("timestamp"),
            "zone_id": item.get("zone"),
            "is_estimated": item.get("isEstimated"),
            "estimation_method": item.get("estimationMethod"),
            "temporal_granularity": item.get("temporalGranularity"),
            "updated_at": item.get("updatedAt"),
            "created_at": item.get("createdAt"),
        }

        for parent, prefix in (
            ("powerProductionBreakdown", "production"),
            ("powerConsumptionBreakdown", "consumption"),
            ("electricityMix", "mix"),
            ("powerBreakdown", "mix"),
        ):
            block = item.get(parent)
            if isinstance(block, dict):
                for key, value in block.items():
                    row[f"{prefix}_{safe_name(str(key))}_MW"] = value

        rows.append(row)

    return rows


def select_endpoint(
    session: requests.Session,
    cfg: dict[str, Any],
    zone: str,
    start: datetime,
    disable_estimations: bool,
) -> str | None:
    test_end = start + timedelta(days=1)

    for endpoint in cfg["endpoints"]:
        params = {
            "zone": zone,
            "start": iso_z(start),
            "end": iso_z(test_end),
            "temporalGranularity": "hourly",
            "disableEstimations": str(disable_estimations).lower(),
            **cfg["params"],
        }

        try:
            payload = request_json(session, endpoint, params)
            if payload_items(payload):
                return endpoint
        except EndpointUnavailable:
            continue

    return None


def download_signal(
    session: requests.Session,
    name: str,
    cfg: dict[str, Any],
    zone: str,
    start: datetime,
    end: datetime,
    raw_path: Path,
    chunk_days: int,
    sleep_min: float,
    sleep_max: float,
    disable_estimations: bool,
    log_path: Path,
) -> pd.DataFrame:
    endpoint = select_endpoint(
        session,
        cfg,
        zone,
        start,
        disable_estimations,
    )

    if endpoint is None:
        message = f"{name}: 当前账户或 CN 区域未找到可用历史 endpoint"
        print(message)
        write_log(log_path, message)

        if cfg["required"]:
            raise RuntimeError(message)

        return pd.DataFrame()

    existing = read_csv(raw_path)

    if not existing.empty:
        current = max(
            start,
            existing["timestamp"].max().to_pydatetime() + timedelta(hours=1),
        )
        print(f"{name}: 从 {iso_z(current)} 续传")
    else:
        current = start

    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)

        params = {
            "zone": zone,
            "start": iso_z(current),
            "end": iso_z(chunk_end),
            "temporalGranularity": "hourly",
            "disableEstimations": str(disable_estimations).lower(),
            **cfg["params"],
        }

        payload = request_json(session, endpoint, params)

        if cfg["kind"] == "mix":
            rows = extract_mix(payload)
        else:
            rows = extract_scalar(payload, cfg["keys"])

        append_rows(raw_path, rows)

        print(
            f"{name}: {iso_z(current)} -> {iso_z(chunk_end)} "
            f"| {len(rows)} 条 | 已保存"
        )
        write_log(
            log_path,
            f"{name}: {iso_z(current)} -> {iso_z(chunk_end)}, rows={len(rows)}",
        )

        current = chunk_end
        time.sleep(random.uniform(sleep_min, sleep_max))

    df = read_csv(raw_path)

    if cfg["kind"] == "scalar" and not df.empty:
        df = df.rename(columns={"value": cfg["column"]})
        atomic_save(df, raw_path)

    return df


def prepare_scalar(
    df: pd.DataFrame,
    value_col: str,
    prefix: str,
) -> pd.DataFrame:
    if df.empty:
        return df

    keep = ["timestamp", value_col]
    rename = {}

    for col in (
        "is_estimated",
        "estimation_method",
        "temporal_granularity",
        "updated_at",
        "created_at",
        "emission_factor_type",
    ):
        if col in df.columns:
            keep.append(col)
            rename[col] = f"{prefix}_{col}"

    return df[keep].rename(columns=rename)


def merge_signals(
    downloaded: dict[str, pd.DataFrame],
    zone: str,
    start: datetime,
    end: datetime,
    output_path: Path,
) -> pd.DataFrame:
    carbon = downloaded["carbon_intensity"]

    result = prepare_scalar(
        carbon,
        SIGNALS["carbon_intensity"]["column"],
        "carbon",
    )

    for name in (
        "renewable_percentage",
        "carbon_free_percentage",
        "total_load",
        "net_load",
        "fossil_only_carbon_intensity",
    ):
        df = downloaded.get(name, pd.DataFrame())

        if df.empty:
            continue

        part = prepare_scalar(
            df,
            SIGNALS[name]["column"],
            name,
        )
        result = result.merge(part, on="timestamp", how="left")

    mix = downloaded.get("electricity_mix", pd.DataFrame())

    if not mix.empty:
        drop_cols = [
            col
            for col in (
                "zone_id",
                "is_estimated",
                "estimation_method",
                "temporal_granularity",
                "updated_at",
                "created_at",
            )
            if col in mix.columns
        ]
        result = result.merge(
            mix.drop(columns=drop_cols),
            on="timestamp",
            how="left",
        )

    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result = result[
        (result["timestamp"] >= pd.Timestamp(start))
        & (result["timestamp"] < pd.Timestamp(end))
    ].copy()

    result["zone_id"] = zone
    result["hour"] = result["timestamp"].dt.hour.astype("int16")
    result["weekday"] = result["timestamp"].dt.dayofweek.astype("int16")
    result["month"] = result["timestamp"].dt.month.astype("int16")
    result["day_of_year"] = result["timestamp"].dt.dayofyear.astype("int16")

    first = [
        "timestamp",
        "zone_id",
        "carbon_intensity_gCO2eq_per_kWh",
        "hour",
        "weekday",
        "month",
        "day_of_year",
    ]

    result = result[first + [c for c in result.columns if c not in first]]
    result = (
        result.drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    atomic_save(result, output_path)
    return result


def quality_report(
    df: pd.DataFrame,
    start: datetime,
    end: datetime,
    path: Path,
) -> None:
    expected = int((end - start).total_seconds() // 3600)

    lines = [
        "# Electricity Maps 数据质量报告",
        "",
        f"- 理论小时数：{expected}",
        f"- 实际记录数：{len(df)}",
        f"- 起始时间：{df['timestamp'].min()}",
        f"- 结束时间：{df['timestamp'].max()}",
        f"- 重复时间戳：{df['timestamp'].duplicated().sum()}",
        "",
        "## 各字段缺失值",
        "",
    ]

    for col in df.columns:
        count = int(df[col].isna().sum())
        lines.append(
            f"- `{col}`：{count}（{count / max(len(df), 1):.2%}）"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--zone", default="CN")
    parser.add_argument("--start", default="2022-01-01T00:00:00Z")
    parser.add_argument("--end", default="2026-01-01T00:00:00Z")
    parser.add_argument("--output-dir", default="data/electricity_maps")
    parser.add_argument(
        "--api-key",
        default=os.getenv("ELECTRICITY_MAPS_API_KEY"),
    )
    parser.add_argument("--chunk-days", type=int, default=5)
    parser.add_argument("--sleep-min", type=float, default=1.5)
    parser.add_argument("--sleep-max", type=float, default=3.0)
    parser.add_argument("--disable-estimations", action="store_true")
    parser.add_argument(
        "--signals",
        nargs="*",
        choices=list(SIGNALS),
        default=list(SIGNALS),
    )

    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("请先设置 ELECTRICITY_MAPS_API_KEY。")

    if not 1 <= args.chunk_days <= 10:
        raise SystemExit("--chunk-days 必须在 1 到 10 之间。")

    start = parse_iso(args.start)
    end = parse_iso(args.end)

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "download.log"

    session = make_session(args.api_key)
    downloaded: dict[str, pd.DataFrame] = {}

    for name in args.signals:
        cfg = SIGNALS[name]

        try:
            downloaded[name] = download_signal(
                session=session,
                name=name,
                cfg=cfg,
                zone=args.zone,
                start=start,
                end=end,
                raw_path=raw_dir / f"{name}.csv",
                chunk_days=args.chunk_days,
                sleep_min=args.sleep_min,
                sleep_max=args.sleep_max,
                disable_estimations=args.disable_estimations,
                log_path=log_path,
            )
        except Exception as exc:
            print(f"{name} 下载失败：{exc}")
            write_log(log_path, f"{name} failed: {exc}")

            if cfg["required"]:
                raise

            downloaded[name] = pd.DataFrame()

    if (
        "carbon_intensity" not in downloaded
        or downloaded["carbon_intensity"].empty
    ):
        raise RuntimeError("碳强度数据为空，无法建立主数据集。")

    merged_path = output_dir / "electricity_maps_CN_2022_2025_all_signals.csv"

    merged = merge_signals(
        downloaded,
        args.zone,
        start,
        end,
        merged_path,
    )

    quality_report(
        merged,
        start,
        end,
        output_dir / "quality_report.md",
    )

    print("\n全部完成")
    print("合并文件：", merged_path.resolve())
    print("数据形状：", merged.shape)


if __name__ == "__main__":
    main()
