from __future__ import annotations

"""Download Electricity Maps signals not included in the previous project run.

Signals downloaded by this script only:
  1. total load
  2. total reported load
  3. day-ahead price
  4. electricity flows
  5. electricity source data (coal, gas, wind, solar, hydro, nuclear, etc.)
  6. fossil-only carbon intensity
  7. net load

The script does NOT download ordinary carbon intensity, renewable percentage,
carbon-free percentage or whole electricity mix again.

Important: a successful endpoint request can still return zero rows for CN.
That means Electricity Maps does not currently provide that historical signal
for the selected zone/date range or that the API token lacks permission.
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

# These source types are documented by Electricity Maps. Unsupported sources
# for CN are automatically recorded as empty instead of stopping the run.
SOURCE_TYPES = [
    "solar", "wind", "hydro", "nuclear", "coal", "gas", "oil",
    "biomass", "geothermal", "hydro-discharge", "battery-discharge",
]

SCALAR_SIGNALS: dict[str, dict[str, Any]] = {
    "total_load": {
        "endpoint": "total-load/past-range",
        "keys": ["totalLoad", "total_load", "load", "value"],
        "column": "total_load_MW",
    },
    "total_reported_load": {
        "endpoint": "total-reported-load/past-range",
        "keys": ["totalReportedLoad", "total_reported_load", "load", "value"],
        "column": "total_reported_load_MW",
    },
    "net_load": {
        "endpoint": "net-load/past-range",
        "keys": ["netLoad", "net_load", "load", "value"],
        "column": "net_load_MW",
    },
    "day_ahead_price": {
        "endpoint": "price-day-ahead/past-range",
        "keys": ["price", "dayAheadPrice", "day_ahead_price", "value"],
        "column": "day_ahead_price_local_per_MWh",
    },
    "fossil_only_carbon_intensity": {
        # Correct v4 endpoint.  It is not "fossil-only-carbon-intensity".
        "endpoint": "carbon-intensity-fossil-only/past-range",
        "keys": ["fossilOnlyCarbonIntensity", "carbonIntensity", "value"],
        "column": "fossil_only_carbon_intensity_gCO2eq_per_kWh",
        "params": {"emissionFactorType": "lifecycle"},
    },
}


def to_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def make_session(api_key: str) -> requests.Session:
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"auth-token": api_key, "Accept": "application/json"})
    return session


def api_get(session: requests.Session, endpoint: str, params: dict[str, Any]) -> tuple[int, Any]:
    """Return HTTP status and decoded payload; do not hide unsupported signals."""
    response = session.get(f"{BASE_URL}/{endpoint}", params=params, timeout=(20, 120))
    try:
        payload: Any = response.json()
    except ValueError:
        payload = {"raw_text": response.text[:1000]}
    return response.status_code, payload


def items_from_payload(payload: Any) -> list[dict[str, Any]]:
    """Accept the documented response shapes used by different v4 signals."""
    if isinstance(payload, dict):
        data = payload.get("data", payload)
        if isinstance(data, dict):
            for key in ("history", "values", "data"):
                if isinstance(data.get(key), list):
                    return data[key]
            return [data]
        if isinstance(data, list):
            return data
    if isinstance(payload, list):
        return payload
    return []


def timestamp_of(item: dict[str, Any]) -> Any:
    return item.get("datetime") or item.get("timestamp") or item.get("time")


def scalar_rows(payload: Any, keys: list[str], column: str) -> list[dict[str, Any]]:
    rows = []
    for item in items_from_payload(payload):
        value = next((item[key] for key in keys if key in item), None)
        rows.append({
            "timestamp": timestamp_of(item),
            column: value,
            "zone_id": item.get("zone"),
            "currency": item.get("currency"),
            "is_estimated": item.get("isEstimated"),
            "estimation_method": item.get("estimationMethod"),
            "temporal_granularity": item.get("temporalGranularity"),
        })
    return rows


def source_rows(payload: Any, source: str) -> list[dict[str, Any]]:
    # Source responses may call the value "power", "production" or "value".
    return scalar_rows(payload, ["power", "production", "value", "electricity"], f"source_{source}_MW")


def flow_rows(payload: Any) -> list[dict[str, Any]]:
    """Keep every reported inter-zone flow as JSON without losing neighbour IDs.

    Flow responses can vary in nesting by API plan/version. Storing the raw flow
    object makes this downloader schema-safe; the resulting CSV can later be
    expanded after inspecting the actual CN response.
    """
    rows = []
    for item in items_from_payload(payload):
        flow = item.get("electricityFlows") or item.get("flows") or item.get("flow")
        rows.append({
            "timestamp": timestamp_of(item),
            "zone_id": item.get("zone"),
            "flows_json": json.dumps(flow, ensure_ascii=False),
            "is_estimated": item.get("isEstimated"),
            "temporal_granularity": item.get("temporalGranularity"),
        })
    return rows


def append_and_save(path: Path, rows: list[dict[str, Any]]) -> int:
    """Append one time chunk, deduplicate by timestamp and preserve restartability."""
    if not rows:
        return 0
    new = pd.DataFrame(rows)
    new["timestamp"] = pd.to_datetime(new["timestamp"], utc=True, errors="coerce")
    new = new.dropna(subset=["timestamp"])
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not old.empty:
        old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True, errors="coerce")
    result = pd.concat([old, new], ignore_index=True)
    result = result.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    result.to_csv(path, index=False)
    return len(new)


def download_range(
    session: requests.Session,
    name: str,
    endpoint: str,
    parser,
    output_path: Path,
    zone: str,
    start: datetime,
    end: datetime,
    chunk_days: int,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Download one signal in small chunks; v4 range APIs cap some hourly calls."""
    current = start
    status_counts: dict[int, int] = {}
    saved_rows = 0
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        params = {
            "zone": zone,
            "start": iso_z(current),
            "end": iso_z(chunk_end),
            "temporalGranularity": "hourly",
            **(extra_params or {}),
        }
        try:
            status, payload = api_get(session, endpoint, params)
        except requests.RequestException as exc:
            print(f"{name}: network error {exc}")
            status_counts[-1] = status_counts.get(-1, 0) + 1
            current = chunk_end
            continue
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == 200:
            saved_rows += append_and_save(output_path, parser(payload))
            print(f"{name}: {iso_z(current)} -> {iso_z(chunk_end)}")
        else:
            # 403/404/422 are useful evidence that this signal is unavailable
            # for CN or not included in the current Electricity Maps API plan.
            print(f"{name}: HTTP {status}; first response: {str(payload)[:180]}")
        current = chunk_end
        time.sleep(random.uniform(0.8, 1.6))
    return {"signal": name, "endpoint": endpoint, "rows_saved": saved_rows, "http_status_counts": json.dumps(status_counts, ensure_ascii=False)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download additional Electricity Maps CN signals only.")
    parser.add_argument("--zone", default="CN")
    parser.add_argument("--start", default="2022-01-01T00:00:00Z")
    parser.add_argument("--end", default="2026-01-01T00:00:00Z")
    parser.add_argument("--output-dir", default="data/electricity_maps_additional")
    parser.add_argument("--chunk-days", type=int, default=5, help="Use <=10 for hourly past-range calls.")
    parser.add_argument("--api-key", default=os.getenv("ELECTRICITY_MAPS_API_KEY"))
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("请先设置环境变量 ELECTRICITY_MAPS_API_KEY。")
    if not 1 <= args.chunk_days <= 10:
        raise SystemExit("--chunk-days 应在 1 到 10 之间。")

    start, end = to_utc(args.start), to_utc(args.end)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session = make_session(args.api_key)
    report = []

    # Download scalar signals requested by the user.
    for name, cfg in SCALAR_SIGNALS.items():
        report.append(download_range(
            session, name, cfg["endpoint"],
            lambda payload, c=cfg: scalar_rows(payload, c["keys"], c["column"]),
            output_dir / f"{name}.csv", args.zone, start, end, args.chunk_days,
            cfg.get("params"),
        ))

    # Electricity flows are stored as raw JSON per timestamp, because neighbour
    # structure can differ by zone and API response version.
    report.append(download_range(
        session, "electricity_flows", "electricity-flows/past-range", flow_rows,
        output_dir / "electricity_flows.csv", args.zone, start, end, args.chunk_days,
    ))

    # Download each source separately.  These are intentionally not the whole
    # electricity-mix signal previously downloaded in your original script.
    for source in SOURCE_TYPES:
        report.append(download_range(
            session, f"electricity_source_{source}", f"electricity-mix/{source}/past-range",
            lambda payload, s=source: source_rows(payload, s),
            output_dir / f"electricity_source_{source}.csv", args.zone, start, end, args.chunk_days,
            {"flowTraced": "false"},
        ))

    pd.DataFrame(report).to_csv(output_dir / "download_availability_report.csv", index=False)
    print("\n完成。请优先查看：", (output_dir / "download_availability_report.csv").resolve())


if __name__ == "__main__":
    main()
