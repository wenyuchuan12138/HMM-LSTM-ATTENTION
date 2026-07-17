from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    zone_id: str = "CN"
    start_time: str = "2017-01-01 00:00:00+00:00"
    end_time: str = "2025-12-31 23:00:00+00:00"
    train_end_time: str = "2024-06-30 23:00:00+00:00"
    val_end_time: str = "2024-12-31 23:00:00+00:00"
    raw_dir: Path = Path("data/electricity_maps/raw")
    combined_data_path: Path = Path("data/17-25.csv")
    results_dir: Path = Path("results")
    artifacts_dir: Path = Path("artifacts")
    figures_dir: Path = Path("results/figures")
    horizon: int = 24
    short_window: int = 24
    long_window: int = 168
    high_carbon_quantile: float = 0.80
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    batch_size: int = 512
    max_epochs: int = 3
    patience: int = 2
    hidden_dim: int = 32
    train_stride: int = 3
    learning_rate: float = 1e-3
    lambda_reg: float = 0.70
    lambda_cls: float = 0.30
    hmm_components: tuple[int, ...] = (3, 4, 5)
    hmm_seeds: tuple[int, ...] = (1, 7, 13, 21, 29, 42, 55, 68, 81, 99, 111, 123, 135, 147, 159, 171, 183, 195, 207, 219)
    hmm_max_iter: int = 50

    def ensure_dirs(self) -> None:
        for path in [self.results_dir, self.artifacts_dir, self.figures_dir]:
            path.mkdir(parents=True, exist_ok=True)
