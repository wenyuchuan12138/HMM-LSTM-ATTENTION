# Electricity Maps 中国小时级研究数据下载包

这个版本只使用 Electricity Maps，不包含 Carbon Monitor-Power 或其他日级数据。

## 下载信号

脚本会依次尝试下载：

1. Carbon Intensity
2. Renewable Percentage
3. Carbon-Free Percentage
4. Total Load
5. Net Load
6. Electricity Mix / Power Breakdown
7. Fossil-Only Carbon Intensity

其中 Carbon Intensity 是必需数据；其他信号如果中国区域没有历史数据，脚本会记录并跳过。

## 安装

```powershell
pip install -r requirements.txt
```

## 设置 API Key

```powershell
$env:ELECTRICITY_MAPS_API_KEY="你的 API Key"
```

## 下载全部数据

```powershell
python download_electricity_maps_all.py
```

默认时间范围：

```text
2022-01-01T00:00:00Z
至
2026-01-01T00:00:00Z
```

即完整覆盖 2022—2025 年。

网络不稳定时建议：

```powershell
python download_electricity_maps_all.py `
  --chunk-days 3 `
  --sleep-min 2 `
  --sleep-max 4
```

脚本每个时间块都会保存，中断后重新运行相同命令即可续传。

## 分阶段下载

先下载核心碳强度：

```powershell
python download_electricity_maps_all.py `
  --signals carbon_intensity
```

再补充其他信号：

```powershell
python download_electricity_maps_all.py `
  --signals renewable_percentage carbon_free_percentage total_load net_load electricity_mix fossil_only_carbon_intensity
```

## 输出

```text
data/electricity_maps/
├─ raw/
│  ├─ carbon_intensity.csv
│  ├─ renewable_percentage.csv
│  ├─ carbon_free_percentage.csv
│  ├─ total_load.csv
│  ├─ net_load.csv
│  ├─ electricity_mix.csv
│  └─ fossil_only_carbon_intensity.csv
├─ electricity_maps_CN_2022_2025_all_signals.csv
├─ quality_report.md
└─ download.log
```

## 数据检查

```powershell
python validate_electricity_maps_data.py
```

## 最终模型建议使用的变量

### 核心目标变量

```text
carbon_intensity_gCO2eq_per_kWh
```

### HMM 参数估计建议变量

```text
carbon_intensity_gCO2eq_per_kWh
carbon intensity difference
rolling volatility
renewable_percentage
carbon_free_percentage
total_load_MW
net_load_MW
```

其中差分与滚动波动率在模型代码中生成，不需要从 API 下载。

### 深度模型补充变量

```text
renewable_percentage
carbon_free_percentage
total_load_MW
net_load_MW
production_coal_MW
production_gas_MW
production_wind_MW
production_solar_MW
production_hydro_MW
production_nuclear_MW
```

这些电源结构字段是否存在，取决于中国区域 Electricity Maps 的历史覆盖情况。

## 重要说明

中国碳强度可能大量为估算值：

```text
isEstimated = true
estimationMethod = GENERAL_PURPOSE_ZONE_MODEL
```

论文中应称为：

> Electricity Maps 提供的中国区域小时级电力碳排放强度估算数据。

不要称为官方逐小时实测数据。
