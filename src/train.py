from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import ExperimentConfig
from .dataset import PreparedData, SequenceData
from .models import HMMMRGALSTM, MarkovianHMMMultiScaleLSTM


def _stride_data(data: SequenceData, stride: int) -> SequenceData:
    """Keep the original speed-oriented subsampling behaviour for B0--B6."""
    if stride <= 1:
        return data
    idx = np.arange(0, len(data.y_cls), stride)
    return SequenceData(data.short[idx], data.long[idx], data.short_hmm[idx], data.long_hmm[idx], data.long_static[idx], data.seasonal[idx], data.residual[idx], data.base[idx], data.y_reg[idx], data.y_cls[idx], data.rows[idx])


def _tensor_dataset(data: SequenceData, hmm_idx: list[int]) -> TensorDataset:
    short, long = torch.tensor(data.short), torch.tensor(data.long)
    if hmm_idx:
        short_hmm, long_hmm = short[:, :, hmm_idx], long[:, :, hmm_idx]
    else:
        short_hmm = torch.zeros(short.shape[0], short.shape[1], 0)
        long_hmm = torch.zeros(long.shape[0], long.shape[1], 0)
    return TensorDataset(short, long, torch.tensor(data.long_static), torch.tensor(data.seasonal), torch.tensor(data.residual), torch.tensor(data.base), short_hmm, long_hmm, torch.tensor(data.y_reg), torch.tensor(data.y_cls))


def _markov_tensor_dataset(data: SequenceData, base_idx: list[int]) -> TensorDataset:
    """Dataset for the new fair experiment: base covariates and raw HMM beliefs are separate."""
    return TensorDataset(torch.tensor(data.short[:, :, base_idx]), torch.tensor(data.long[:, :, base_idx]), torch.tensor(data.seasonal), torch.tensor(data.residual), torch.tensor(data.short_hmm), torch.tensor(data.long_hmm), torch.tensor(data.y_reg), torch.tensor(data.y_cls))


def _fit_model(model, train_loader, val_loader, cfg, seasonal_residual, train_base, val_y, val_base, mode_name):
    """Shared multi-task training loop with validation-MAE early stopping."""
    device = next(model.parameters()).device
    reg_loss = nn.HuberLoss()
    # Positive weight prevents the rarer high-carbon-transition class from being ignored.
    train_labels = train_loader.dataset.tensors[-1]
    pos = max(float(train_labels.sum()), 1.0)
    neg = max(float(len(train_labels) - train_labels.sum()), 1.0)
    cls_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    target = train_loader.dataset.tensors[-2].numpy() - train_base if seasonal_residual else train_loader.dataset.tensors[-2].numpy()
    y_mean, y_std = float(target.mean()), float(target.std() + 1e-6)
    best_state, best_val, stale, history = None, float("inf"), 0, []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train(); losses = []
        for batch in train_loader:
            batch = [b.to(device) for b in batch]
            *inputs, y_reg, y_cls = batch
            # Scale only the regression target; classification labels remain 0/1.
            y_reg = (y_reg - y_mean) / y_std
            opt.zero_grad()
            pred_reg, pred_cls, _ = model(*inputs)
            loss = cfg.lambda_reg * reg_loss(pred_reg, y_reg) + cfg.lambda_cls * cls_loss(pred_cls, y_cls)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(loss.detach().cpu()))
        val_pred = _predict(model, val_loader, device)
        val_reg = val_pred["reg"] * y_std + y_mean
        if seasonal_residual:
            val_reg = val_reg + val_base
        val_mae = float(np.mean(np.abs(val_y - val_reg)))
        history.append({"model": mode_name, "epoch": epoch, "loss": float(np.mean(losses)), "val_mae": val_mae})
        if val_mae < best_val:
            best_val, stale = val_mae, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return y_mean, y_std, pd.DataFrame(history)


def _predict(model, loader, device):
    regs, probs, attns = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = [b.to(device) for b in batch]
            # Legacy data has an additional ``base`` tensor used outside the
            # neural network; Markovian data does not.
            inputs = batch[:-2]
            if len(batch) == 10:
                inputs = [batch[0], batch[1], batch[2], batch[3], batch[4], batch[6], batch[7]]
            reg, cls, attn = model(*inputs)
            regs.append(reg.cpu().numpy()); probs.append(torch.sigmoid(cls).cpu().numpy()); attns.append(attn.cpu().numpy())
    return {"reg": np.vstack(regs), "cls_prob": np.concatenate(probs), "attention": np.vstack(attns)}


def train_torch_model(name: str, prepared: PreparedData, cfg: ExperimentConfig, use_long_branch: bool, use_hmm_gate: bool, use_state_attention: bool, use_residual: bool, use_multiscale: bool, use_long_mlp: bool = False, seasonal_residual: bool = False, cls_only: bool = False, reg_only: bool = False):
    """Original B0--B6 trainer.  Retained so the prior experiment suite still runs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hmm_idx = [prepared.feature_cols.index(c) for c in prepared.hmm_prob_cols if c in prepared.feature_cols]
    model = HMMMRGALSTM(len(prepared.feature_cols), len(prepared.seasonal_cols), len(prepared.residual_cols), len(prepared.long_static_cols), len(hmm_idx), cfg.horizon, cfg.hidden_dim, use_long_branch, use_hmm_gate, use_state_attention, use_residual, use_multiscale, use_long_mlp).to(device)
    train_target = prepared.train.y_reg - prepared.train.base if seasonal_residual else prepared.train.y_reg
    y_mean, y_std = float(train_target.mean()), float(train_target.std() + 1e-6)
    train_scaled = replace(prepared.train, y_reg=(train_target - y_mean) / y_std)
    train_loader = DataLoader(_tensor_dataset(_stride_data(train_scaled, cfg.train_stride), hmm_idx), batch_size=cfg.batch_size, shuffle=False)
    val_loader = DataLoader(_tensor_dataset(prepared.val, hmm_idx), batch_size=512, shuffle=False)
    # Preserve the legacy loss choices, including its single-task switches.
    reg_loss, opt = nn.HuberLoss(), torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    labels = train_loader.dataset.tensors[-1]; pos = max(float(labels.sum()), 1.0); neg = max(float(len(labels) - labels.sum()), 1.0)
    cls_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
    best_state, best_val, stale, history = None, float("inf"), 0, []
    for epoch in range(1, cfg.max_epochs + 1):
        model.train(); losses=[]
        for batch in train_loader:
            short, long, long_static, seasonal, residual, base, sh, lh, y_reg, y_cls = [b.to(device) for b in batch]
            opt.zero_grad(); p_reg, p_cls, _ = model(short, long, long_static, seasonal, residual, sh, lh)
            loss_reg, loss_cls = reg_loss(p_reg, y_reg), cls_loss(p_cls, y_cls)
            loss = loss_cls if cls_only else loss_reg if reg_only else cfg.lambda_reg * loss_reg + cfg.lambda_cls * loss_cls
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.detach().cpu()))
        val_pred = _predict(model, val_loader, device); val_pred["reg"] = val_pred["reg"] * y_std + y_mean
        if seasonal_residual: val_pred["reg"] += prepared.val.base
        val_mae = float(np.mean(np.abs(prepared.val.y_reg - val_pred["reg"])))
        history.append({"model": name, "epoch": epoch, "loss": float(np.mean(losses)), "val_mae": val_mae})
        if val_mae < best_val:
            best_val, stale = val_mae, 0; best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience: break
    model.load_state_dict(best_state); cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "feature_cols": prepared.feature_cols}, cfg.artifacts_dir / f"{name}.pt")
    val_pred = _predict(model, val_loader, device); test_pred = _predict(model, DataLoader(_tensor_dataset(prepared.test, hmm_idx), batch_size=512), device)
    for pred, data in [(val_pred, prepared.val), (test_pred, prepared.test)]:
        pred["reg"] = pred["reg"] * y_std + y_mean
        if seasonal_residual: pred["reg"] += data.base
    return test_pred, {"model": model, "hmm_idx": hmm_idx, "val_pred": val_pred}, pd.DataFrame(history)


def train_markovian_model(name: str, prepared: PreparedData, cfg: ExperimentConfig, mode: str):
    """Train one new fair ablation branch with seasonal-residual targets.

    Existing experiments use their original features.  This new family excludes
    every hmm_* feature from ordinary inputs and receives only raw causal HMM
    posterior probabilities through the designated mechanism.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_idx = [prepared.feature_cols.index(c) for c in prepared.markov_feature_cols]
    n_states = len(prepared.hmm_prob_cols)
    model = MarkovianHMMMultiScaleLSTM(len(base_idx), len(prepared.seasonal_cols), len(prepared.residual_cols), n_states, cfg.horizon, cfg.hidden_dim, mode).to(device)
    # New experiment uses every training window and longer training than the
    # retained fast legacy branches.
    train_data = _stride_data(prepared.train, cfg.train_stride)
    train_loader = DataLoader(_markov_tensor_dataset(train_data, base_idx), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(_markov_tensor_dataset(prepared.val, base_idx), batch_size=512, shuffle=False)
    y_mean, y_std, history = _fit_model(model, train_loader, val_loader, cfg, True, train_data.base, prepared.val.y_reg, prepared.val.base, name)
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "markov_feature_cols": prepared.markov_feature_cols, "hmm_prob_cols": prepared.hmm_prob_cols, "mode": mode}, cfg.artifacts_dir / f"{name}.pt")
    val_pred = _predict(model, val_loader, device)
    test_pred = _predict(model, DataLoader(_markov_tensor_dataset(prepared.test, base_idx), batch_size=512, shuffle=False), device)
    for pred, data in [(val_pred, prepared.val), (test_pred, prepared.test)]:
        pred["reg"] = pred["reg"] * y_std + y_mean + data.base
    return test_pred, {"model": model, "val_pred": val_pred, "mode": mode}, history
