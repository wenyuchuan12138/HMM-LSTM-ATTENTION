from __future__ import annotations

"""
Electricity Maps 合并数据质量检查
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/electricity_maps/electricity_maps_CN_2022_2025_all_signals.csv",
    )
    parser.add_argument(
        "--output",
        default="data/electricity_maps/final_quality_report.txt",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    expected = pd.date_range(
        df["timestamp"].min(),
        df["timestamp"].max(),
        freq="h",
        tz="UTC",
    )

    actual = pd.DatetimeIndex(df["timestamp"])
    missing_timestamps = expected.difference(actual)

    lines = [
        "Electricity Maps 数据质量报告",
        "=" * 40,
        f"记录数：{len(df)}",
        f"字段数：{len(df.columns)}",
        f"最早时间：{df['timestamp'].min()}",
        f"最晚时间：{df['timestamp'].max()}",
        f"重复时间戳：{df['timestamp'].duplicated().sum()}",
        f"缺失小时数：{len(missing_timestamps)}",
        "",
        "各字段缺失值：",
    ]

    for col in df.columns:
        lines.append(
            f"{col}: {df[col].isna().sum()} "
            f"({df[col].isna().mean():.2%})"
        )

    if "carbon_is_estimated" in df.columns:
        lines.extend(
            [
                "",
                "碳强度估算值分布：",
                str(df["carbon_is_estimated"].value_counts(dropna=False)),
            ]
        )

    report = "\n".join(lines)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")

    print(report)
    print("\n已保存：", output.resolve())


if __name__ == "__main__":
    main()
