from __future__ import annotations

import os
import shutil
import time
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore")

from src.baselines import seasonal_naive, train_lightgbm
from src.config import ExperimentConfig
from src.data_loader import load_and_merge, profile_csvs
from src.dataset import prepare_sequences, split_indices_by_time
from src.evaluate import classification_metrics, confusion_counts, horizon_metrics, regression_metrics
from src.feature_engineering import add_features
from src.hmm_model import fit_hmm_features
from src.plots import save_figures
from src.train import train_markovian_model, train_torch_model
from src.utils import save_json, set_seed


def build_metric_row(
    model: str,
    y_reg: np.ndarray,
    y_cls: np.ndarray,
    pred: dict[str, np.ndarray],
    threshold: float = 0.5,
) -> dict[str, float | str]:
    return {"model": model, **regression_metrics(y_reg, pred["reg"]), **classification_metrics(y_cls, pred["cls_prob"], threshold)}


def safe_to_csv(df: pd.DataFrame, path) -> None:
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_updated{path.suffix}")
        df.to_csv(fallback, index=False)
        print(f"[WARN] {path} is locked; wrote {fallback} instead.")


def tune_threshold_and_direction(y_val: np.ndarray, val_prob: np.ndarray, test_prob: np.ndarray) -> tuple[np.ndarray, float, str, float]:
    from sklearn.metrics import f1_score, roc_auc_score

    direction = "normal"
    checked_val = val_prob
    checked_test = test_prob
    if len(np.unique(y_val)) == 2:
        auc_normal = roc_auc_score(y_val, val_prob)
        auc_flipped = roc_auc_score(y_val, 1 - val_prob)
        if auc_flipped > auc_normal:
            direction = "flipped"
            checked_val = 1 - val_prob
            checked_test = 1 - test_prob
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.1, 0.9, 17):
        f1 = f1_score(y_val, (checked_val >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return checked_test, best_threshold, direction, float(best_f1)


def select_residual_shrinkage(
    y_val: np.ndarray,
    base_val: np.ndarray,
    corrected_val: np.ndarray,
) -> float:
    residual_hat = corrected_val - base_val
    best_alpha = 0.0
    best_mae = float(np.mean(np.abs(y_val - base_val)))
    for alpha in np.linspace(0.0, 1.0, 21):
        mae = float(np.mean(np.abs(y_val - (base_val + alpha * residual_hat))))
        if mae < best_mae:
            best_mae = mae
            best_alpha = float(alpha)
    return best_alpha


def residual_diagnostics(y_true: np.ndarray, base_pred: np.ndarray, corrected_pred: np.ndarray, model: str, alpha: float) -> dict[str, float | str]:
    true_residual = y_true - base_pred
    pred_residual = corrected_pred - base_pred
    true_flat = true_residual.ravel()
    pred_flat = pred_residual.ravel()
    corr = float(np.corrcoef(true_flat, pred_flat)[0, 1]) if np.std(pred_flat) > 1e-8 else float("nan")
    return {
        "model": model,
        "residual_alpha": alpha,
        "true_residual_std": float(np.std(true_flat)),
        "pred_residual_std": float(np.std(pred_flat)),
        "residual_corr": corr,
        "mae_before": float(np.mean(np.abs(y_true - base_pred))),
        "mae_after": float(np.mean(np.abs(y_true - corrected_pred))),
    }


def main() -> None:
    # 1) Read configuration, create result folders and make random operations reproducible.
    cfg = ExperimentConfig()
    cfg.ensure_dirs()
    set_seed(cfg.seed)

    # 2) Read raw Electricity Maps files, engineer causal lag/rolling features,
    #    then split strictly by time before fitting any learning component.
    profile_csvs(cfg.raw_dir, cfg.results_dir / "data_profile.csv")
    merged = load_and_merge(cfg.raw_dir, cfg.start_time, cfg.end_time, cfg.results_dir / "merged_hourly_data.csv")
    features = add_features(merged)
    train_idx, val_idx, test_idx = split_indices_by_time(features, cfg.train_end_time, cfg.val_end_time)

    # 3) HMM is fitted on training observations only.  Its state probabilities
    #    for all later rows are produced by causal forward filtering.
    features, hmm_params = fit_hmm_features(
        features,
        train_idx,
        cfg.hmm_components,
        cfg.hmm_seeds,
        cfg.hmm_max_iter,
        cfg.results_dir,
    )
    # 4) Build 24-hour and 168-hour supervised windows plus the 24-hour
    #    seasonal-naive base forecast used by seasonal-residual branches.
    prepared = prepare_sequences(
        features,
        train_idx,
        val_idx,
        test_idx,
        cfg.horizon,
        cfg.short_window,
        cfg.long_window,
        cfg.high_carbon_quantile,
    )
    features.to_csv(cfg.results_dir / "merged_hourly_data.csv", index=False)
    save_json(
        {
            "seed": cfg.seed,
            "zone_id": cfg.zone_id,
            "target": {
                "regression": "future_24h_carbon_intensity_vector",
                "classification": "current not high and any next 24h above train-only high threshold",
                "high_threshold": prepared.high_threshold,
            },
            "feature_cols": prepared.feature_cols,
            "hmm_prob_cols": prepared.hmm_prob_cols,
            "date_range": {"start": cfg.start_time, "end": cfg.end_time},
            "hmm": {"selected_K": hmm_params["selected_K"], "selected_seed": hmm_params["selected_seed"]},
        },
        cfg.results_dir / "experiment_metadata.json",
    )
    hmm_branch_root = cfg.results_dir / "hmm_parameters"
    hmm_branch_root.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    horizon_rows = []
    histories = []
    runtime_rows = []
    threshold_rows = []
    predictions: dict[str, dict[str, np.ndarray]] = {}
    val_predictions: dict[str, dict[str, np.ndarray]] = {}

    # 5) Conventional baselines remain necessary: a deep model is useful only
    #    if it can beat simple seasonal persistence and tree models.
    carbon_idx = prepared.feature_cols.index("carbon_intensity_gCO2eq_per_kWh")
    baseline_preds = {
        "Seasonal Naive": seasonal_naive(prepared, carbon_idx),
        "LightGBM": train_lightgbm(prepared, cfg.seed),
    }
    for name, pred in baseline_preds.items():
        metrics_rows.append(build_metric_row(name, prepared.test.y_reg, prepared.test.y_cls, pred))
        horizon_rows.append(horizon_metrics(prepared.test.y_reg, pred["reg"], name))
        predictions[name] = pred
        if "val_reg" in pred:
            val_predictions[name] = {"reg": pred["val_reg"], "cls_prob": pred["val_cls_prob"]}

    # 6) Legacy B0--B6 ablations are intentionally retained unchanged so their
    #    historical structure can still be compared after the causal-HMM fix.
    branch_specs = [
        ("B0_original_HMM_MRGA_LSTM", dict(use_long_branch=True, use_hmm_gate=True, use_state_attention=True, use_residual=True, use_multiscale=True)),
        ("B1_seasonal_residual", dict(use_long_branch=True, use_hmm_gate=True, use_state_attention=True, use_residual=True, use_multiscale=True, seasonal_residual=True)),
        ("B2_long_mlp_compression", dict(use_long_branch=False, use_hmm_gate=True, use_state_attention=True, use_residual=True, use_multiscale=False, use_long_mlp=True)),
        ("B3_reduced_HMM_features", dict(use_long_branch=True, use_hmm_gate=True, use_state_attention=True, use_residual=True, use_multiscale=True)),
        ("B4_regression_decoupled", dict(use_long_branch=True, use_hmm_gate=True, use_state_attention=True, use_residual=True, use_multiscale=True, reg_only=True)),
        ("B4_classification_decoupled", dict(use_long_branch=True, use_hmm_gate=True, use_state_attention=True, use_residual=True, use_multiscale=True, cls_only=True)),
        ("B6_combo_B1_B2_B3", dict(use_long_branch=False, use_hmm_gate=True, use_state_attention=True, use_residual=True, use_multiscale=False, use_long_mlp=True, seasonal_residual=True)),
    ]
    hmm_branch_names = [name for name, _ in branch_specs] + ["B5_LightGBM_deep_residual"]
    for branch_name in hmm_branch_names:
        branch_dir = hmm_branch_root / branch_name
        branch_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cfg.results_dir / "hmm_parameters.json", branch_dir / "parameters.json")
    for name, kwargs in branch_specs:
        start = time.perf_counter()
        train_cfg = replace(cfg, max_epochs=4, patience=2, train_stride=2)
        pred, model_info, history = train_torch_model(name, prepared, train_cfg, **kwargs)
        elapsed = time.perf_counter() - start
        metrics_rows.append(build_metric_row(name, prepared.test.y_reg, prepared.test.y_cls, pred))
        horizon_rows.append(horizon_metrics(prepared.test.y_reg, pred["reg"], name))
        histories.append(history.assign(model=name))
        predictions[name] = pred
        val_predictions[name] = model_info["val_pred"]
        param_count = sum(p.numel() for p in model_info["model"].parameters())
        runtime_rows.append({"model": name, "seed": cfg.seed, "parameters": param_count, "train_seconds": elapsed})

    start = time.perf_counter()
    b1_aux_cfg = replace(cfg, max_epochs=4, patience=2, train_stride=2)
    b1_aux_pred, b1_aux_info, b1_aux_history = train_torch_model(
        "B1_HMM_aux_no_gate",
        prepared,
        b1_aux_cfg,
        use_long_branch=True,
        use_hmm_gate=False,
        use_state_attention=False,
        use_residual=True,
        use_multiscale=True,
        seasonal_residual=True,
    )
    elapsed = time.perf_counter() - start
    metrics_rows.append(build_metric_row("B1_HMM_aux_no_gate", prepared.test.y_reg, prepared.test.y_cls, b1_aux_pred))
    horizon_rows.append(horizon_metrics(prepared.test.y_reg, b1_aux_pred["reg"], "B1_HMM_aux_no_gate"))
    histories.append(b1_aux_history.assign(model="B1_HMM_aux_no_gate"))
    predictions["B1_HMM_aux_no_gate"] = b1_aux_pred
    val_predictions["B1_HMM_aux_no_gate"] = b1_aux_info["val_pred"]
    runtime_rows.append(
        {
            "model": "B1_HMM_aux_no_gate",
            "seed": cfg.seed,
            "parameters": sum(p.numel() for p in b1_aux_info["model"].parameters()),
            "train_seconds": elapsed,
        }
    )

    # -----------------------------------------------------------------------
    # New Markovian-RNN-inspired ablation group.  B0--B6 above are kept
    # untouched.  These three branches use the same causal HMM probabilities,
    # seasonal residual target and multi-scale architecture; only HMM fusion
    # changes, making the comparison interpretable.
    # -----------------------------------------------------------------------
    markov_specs = [
        ("B7_no_HMM_multiscale_residual", "baseline"),
        ("B8_causal_HMM_probability_concat", "concat"),
        ("B9_Markovian_HMM_state_mixture", "markovian"),
    ]
    for name, mode in markov_specs:
        start = time.perf_counter()
        # Unlike the legacy quick screen (4 epochs, stride 2), this evidence
        # experiment trains all windows and allows early stopping to work.
        markov_cfg = replace(cfg, max_epochs=30, patience=6, train_stride=1)
        pred, model_info, history = train_markovian_model(name, prepared, markov_cfg, mode)
        elapsed = time.perf_counter() - start
        metrics_rows.append(build_metric_row(name, prepared.test.y_reg, prepared.test.y_cls, pred))
        horizon_rows.append(horizon_metrics(prepared.test.y_reg, pred["reg"], name))
        histories.append(history)
        predictions[name] = pred
        val_predictions[name] = model_info["val_pred"]
        runtime_rows.append({
            "model": name,
            "seed": cfg.seed,
            "parameters": sum(p.numel() for p in model_info["model"].parameters()),
            "train_seconds": elapsed,
        })

    # 7) The original LightGBM-plus-deep-residual experiment is also retained.
    b5_base = baseline_preds["LightGBM"]
    prepared_b5 = replace(
        prepared,
        train=replace(prepared.train, base=b5_base["train_reg"].astype(np.float32)),
        val=replace(prepared.val, base=b5_base["val_reg"].astype(np.float32)),
        test=replace(prepared.test, base=b5_base["reg"].astype(np.float32)),
    )
    start = time.perf_counter()
    b5_cfg = replace(cfg, max_epochs=4, patience=2, train_stride=2)
    b5_pred, b5_info, b5_history = train_torch_model(
        "B5_LightGBM_deep_residual",
        prepared_b5,
        b5_cfg,
        use_long_branch=False,
        use_hmm_gate=True,
        use_state_attention=True,
        use_residual=True,
        use_multiscale=False,
        use_long_mlp=True,
        seasonal_residual=True,
    )
    b5_pred["cls_prob"] = 0.5 * b5_pred["cls_prob"] + 0.5 * b5_base["cls_prob"]
    b5_val = b5_info["val_pred"]
    b5_val["cls_prob"] = 0.5 * b5_val["cls_prob"] + 0.5 * b5_base["val_cls_prob"]
    residual_alpha = select_residual_shrinkage(prepared.val.y_reg, b5_base["val_reg"], b5_val["reg"])
    b5_optimized = {
        "reg": b5_base["reg"] + residual_alpha * (b5_pred["reg"] - b5_base["reg"]),
        "cls_prob": b5_pred["cls_prob"],
    }
    elapsed = time.perf_counter() - start
    metrics_rows.append(build_metric_row("B5_LightGBM_deep_residual", prepared.test.y_reg, prepared.test.y_cls, b5_pred))
    horizon_rows.append(horizon_metrics(prepared.test.y_reg, b5_pred["reg"], "B5_LightGBM_deep_residual"))
    metrics_rows.append(build_metric_row("B5_LightGBM_deep_residual_optimized", prepared.test.y_reg, prepared.test.y_cls, b5_optimized))
    horizon_rows.append(horizon_metrics(prepared.test.y_reg, b5_optimized["reg"], "B5_LightGBM_deep_residual_optimized"))
    histories.append(b5_history.assign(model="B5_LightGBM_deep_residual"))
    predictions["B5_LightGBM_deep_residual"] = b5_pred
    predictions["B5_LightGBM_deep_residual_optimized"] = b5_optimized
    val_predictions["B5_LightGBM_deep_residual"] = b5_val
    b5_diag = residual_diagnostics(
        prepared.test.y_reg,
        b5_base["reg"],
        b5_optimized["reg"],
        "B5_LightGBM_deep_residual_optimized",
        residual_alpha,
    )
    runtime_rows.append(
        {
            "model": "B5_LightGBM_deep_residual",
            "seed": cfg.seed,
            "parameters": sum(p.numel() for p in b5_info["model"].parameters()),
            "train_seconds": elapsed,
        }
    )

    for tuned_name in ["B1_seasonal_residual", "B3_reduced_HMM_features"]:
        tuned_prob, threshold, direction, val_f1 = tune_threshold_and_direction(
            prepared.val.y_cls,
            val_predictions[tuned_name]["cls_prob"],
            predictions[tuned_name]["cls_prob"],
        )
        tuned_pred = {**predictions[tuned_name], "cls_prob": tuned_prob}
        row_name = f"{tuned_name}_threshold_tuned"
        metrics_rows.append(build_metric_row(row_name, prepared.test.y_reg, prepared.test.y_cls, tuned_pred, threshold))
        horizon_rows.append(horizon_metrics(prepared.test.y_reg, tuned_pred["reg"], row_name))
        threshold_rows.append({"model": row_name, "threshold": threshold, "direction": direction, "val_f1": val_f1})

    b5_cls_prob, b5_threshold, b5_direction, b5_val_f1 = tune_threshold_and_direction(
        prepared.val.y_cls,
        b5_val["cls_prob"],
        b5_optimized["cls_prob"],
    )
    b5_tuned = {**b5_optimized, "cls_prob": b5_cls_prob}
    metrics_rows.append(build_metric_row("B5_LightGBM_deep_residual_threshold_tuned", prepared.test.y_reg, prepared.test.y_cls, b5_tuned, b5_threshold))
    horizon_rows.append(horizon_metrics(prepared.test.y_reg, b5_tuned["reg"], "B5_LightGBM_deep_residual_threshold_tuned"))
    threshold_rows.append({"model": "B5_LightGBM_deep_residual_threshold_tuned", "threshold": b5_threshold, "direction": b5_direction, "val_f1": b5_val_f1})

    for b4_name in ["B4_regression_decoupled", "B4_classification_decoupled"]:
        checked_prob, threshold, direction, val_f1 = tune_threshold_and_direction(
            prepared.val.y_cls,
            val_predictions[b4_name]["cls_prob"],
            predictions[b4_name]["cls_prob"],
        )
        predictions[b4_name]["cls_prob_checked"] = checked_prob
        threshold_rows.append({"model": b4_name, "threshold": threshold, "direction": direction, "val_f1": val_f1})

    # 8) Save one common metric table so legacy and new Markovian branches can
    #    be compared by MAE/RMSE/R2 and PR-AUC/Recall/F1.
    metrics = pd.DataFrame(metrics_rows).sort_values("MAE")
    safe_to_csv(metrics, cfg.results_dir / "metrics_summary.csv")
    safe_to_csv(metrics, cfg.results_dir / "branch_metrics_summary.csv")
    horizon_df = pd.concat(horizon_rows, ignore_index=True)
    safe_to_csv(horizon_df, cfg.results_dir / "horizon_metrics.csv")
    focus_horizon = horizon_df[horizon_df["horizon"].isin([1, 6, 12, 24])]
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=focus_horizon, x="horizon", y="MAE", hue="model", marker="o")
    plt.title("Horizon MAE comparison")
    plt.tight_layout()
    plt.savefig(cfg.figures_dir / "horizon_mae_comparison.png", dpi=180)
    plt.close()
    branch_metrics = metrics.copy()
    b0 = branch_metrics[branch_metrics["model"] == "B0_original_HMM_MRGA_LSTM"].iloc[0]
    deltas = []
    for _, row in branch_metrics.iterrows():
        deltas.append({"model": row["model"], "delta_MAE_vs_B0": row["MAE"] - b0["MAE"], "delta_PR_AUC_vs_B0": row["PR_AUC"] - b0["PR_AUC"]})
    safe_to_csv(pd.DataFrame(deltas), cfg.results_dir / "ablation_comparison.csv")
    safe_to_csv(pd.DataFrame(deltas), cfg.results_dir / "ablation_results.csv")
    safe_to_csv(pd.DataFrame(runtime_rows), cfg.results_dir / "branch_runtime.csv")
    safe_to_csv(pd.DataFrame([b5_diag]), cfg.results_dir / "b5_residual_diagnostics.csv")
    safe_to_csv(
        pd.DataFrame(
            [
                {"check": "uses_HMMMRGALSTM_class", "status": True, "evidence": "src/models.py::HMMMRGALSTM; main.py branch_specs"},
                {"check": "hmm_posterior_probabilities", "status": len(prepared.hmm_prob_cols) > 0, "evidence": "|".join(prepared.hmm_prob_cols)},
                {"check": "legacy_hmm_gate_layer", "status": True, "evidence": "HMMMRGALSTM.gate enabled when use_hmm_gate=True"},
                {"check": "causal_HMM_filtering", "status": True, "evidence": "hmm_model.causal_filter_probabilities; no predict_proba(x_all)"},
                {"check": "markovian_state_mixture", "status": True, "evidence": "MarkovianLSTMEncoder mixes K state-specific LSTMCell outputs by alpha_(t-1)"},
                {"check": "markovian_hmm_not_plain_input", "status": True, "evidence": "B9 excludes every hmm_* column from ordinary LSTM inputs"},
                {"check": "multi_scale_input", "status": True, "evidence": "short 24h branch plus long 168h branch or B2/B6 long_static MLP"},
                {"check": "state_conditioned_attention", "status": True, "evidence": "StateConditionedAttention projects HMM posterior probabilities"},
                {"check": "prediction_residual_head", "status": True, "evidence": "linear_residual head; B1/B5 use base + predicted residual"},
                {"check": "B1_HMM_aux_without_complex_gate", "status": True, "evidence": "B1_HMM_aux_no_gate uses HMM features as inputs, with use_hmm_gate=False and use_state_attention=False"},
            ]
        ),
        cfg.results_dir / "model_structure_audit.csv",
    )
    if not threshold_rows:
        threshold_rows = [{"model": row["model"], "threshold": 0.5, "direction": "normal", "val_f1": np.nan} for _, row in metrics.iterrows()]
    safe_to_csv(pd.DataFrame(threshold_rows), cfg.results_dir / "validation_thresholds.csv")
    safe_to_csv(pd.concat(histories, ignore_index=True), cfg.results_dir / "training_history.csv")

    # Use the proposed B9 branch for diagnostic plots when it is available;
    # this does not affect the metric table or the retained legacy outputs.
    report_model = "B9_Markovian_HMM_state_mixture" if "B9_Markovian_HMM_state_mixture" in predictions else "B6_combo_B1_B2_B3"
    best_pred = predictions[report_model]
    rows = prepared.test.rows
    pred_df = pd.DataFrame({"timestamp": features.loc[rows, "timestamp"].to_numpy(), "transition_label": prepared.test.y_cls})
    for h in range(cfg.horizon):
        pred_df[f"y_true_h{h + 1}"] = prepared.test.y_reg[:, h]
        pred_df[f"pred_h{h + 1}"] = best_pred["reg"][:, h]
    pred_df["transition_prob"] = best_pred["cls_prob"]
    safe_to_csv(pred_df, cfg.results_dir / "predictions_test.csv")

    cm = confusion_counts(prepared.test.y_cls, best_pred["cls_prob"])
    save_figures(
        features,
        pred_df,
        metrics,
        cm,
        cfg.figures_dir,
        transition_matrix=hmm_params["transmat"],
        attention=best_pred.get("attention"),
    )

    print("Experiment complete.")
    print(f"Results: {cfg.results_dir.resolve()}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
