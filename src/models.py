from __future__ import annotations

import torch
import torch.nn as nn


class StateConditionedAttention(nn.Module):
    """Temporal attention whose scores can depend on HMM state beliefs."""

    def __init__(self, hidden_dim: int, hmm_dim: int):
        super().__init__()
        self.h_proj = nn.Linear(hidden_dim, hidden_dim)
        self.p_proj = nn.Linear(hmm_dim, hidden_dim) if hmm_dim > 0 else None
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor, p: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        score = self.h_proj(h)
        if self.p_proj is not None and p is not None:
            score = score + self.p_proj(p)
        alpha = torch.softmax(self.v(torch.tanh(score)).squeeze(-1), dim=1)
        return torch.sum(h * alpha.unsqueeze(-1), dim=1), alpha


class HMMMRGALSTM(nn.Module):
    """Original architecture retained unchanged for the user's B0--B6 runs."""

    def __init__(self, input_dim: int, seasonal_dim: int, residual_dim: int, long_static_dim: int, hmm_dim: int, horizon: int, hidden_dim: int = 64, use_long_branch: bool = True, use_hmm_gate: bool = True, use_state_attention: bool = True, use_residual: bool = True, use_multiscale: bool = True, use_long_mlp: bool = False):
        super().__init__()
        self.horizon = horizon
        self.use_long_branch = use_long_branch and use_multiscale
        self.use_hmm_gate = use_hmm_gate and hmm_dim > 0
        self.use_state_attention = use_state_attention
        self.short_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.long_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True) if self.use_long_branch else None
        self.gate = nn.Linear(hmm_dim, hidden_dim) if self.use_hmm_gate else None
        attention_hmm_dim = hmm_dim if use_state_attention else 0
        self.short_attn = StateConditionedAttention(hidden_dim, attention_hmm_dim)
        self.long_attn = StateConditionedAttention(hidden_dim, attention_hmm_dim) if self.use_long_branch else None
        self.long_mlp = nn.Sequential(nn.Linear(long_static_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.10)) if use_long_mlp and long_static_dim > 0 else None
        context_dim = hidden_dim * (1 + int(self.use_long_branch or self.long_mlp is not None)) + seasonal_dim
        self.deep = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.15))
        self.reg_head, self.cls_head = nn.Linear(hidden_dim, horizon), nn.Linear(hidden_dim, 1)
        self.linear_residual = nn.Linear(residual_dim, horizon) if use_residual and residual_dim > 0 else None

    def _encode(self, lstm: nn.LSTM, x: torch.Tensor, hmm_p: torch.Tensor | None, attn: StateConditionedAttention):
        h, _ = lstm(x)
        if self.gate is not None and hmm_p is not None:
            h = h * torch.sigmoid(self.gate(hmm_p))
        return attn(h, hmm_p if self.use_state_attention else None)

    def forward(self, short_x, long_x, long_static_x, seasonal_x, residual_x, short_hmm, long_hmm):
        short_context, short_alpha = self._encode(self.short_lstm, short_x, short_hmm, self.short_attn)
        contexts = [short_context]
        if self.long_lstm is not None and self.long_attn is not None:
            long_context, _ = self._encode(self.long_lstm, long_x, long_hmm, self.long_attn)
            contexts.append(long_context)
        elif self.long_mlp is not None:
            contexts.append(self.long_mlp(long_static_x))
        contexts.append(seasonal_x)
        z = self.deep(torch.cat(contexts, dim=1))
        y_reg = self.reg_head(z)
        if self.linear_residual is not None:
            y_reg = y_reg + self.linear_residual(residual_x)
        return y_reg, self.cls_head(z).squeeze(-1), short_alpha


class MarkovianLSTMEncoder(nn.Module):
    """Markovian-RNN-inspired LSTM encoder with soft state switching.

    At each time step every HMM regime has its own LSTMCell.  Their candidate
    hidden/cell states are mixed using the *previous* causal posterior belief.
    This implements h_t=sum_k alpha_(t-1,k) h_t^(k), rather than multiplying
    one already-computed LSTM output by a gate.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_states: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cells = nn.ModuleList([nn.LSTMCell(input_dim, hidden_dim) for _ in range(n_states)])

    def forward(self, x: torch.Tensor, causal_probs: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = x.shape
        h = x.new_zeros(batch, self.hidden_dim)
        c = x.new_zeros(batch, self.hidden_dim)
        outputs = []
        for t in range(steps):
            # alpha_(t-1) is available before the model processes x_t.  For the
            # first point of an extracted window, use its first causal belief.
            belief = causal_probs[:, max(t - 1, 0), :]
            h_candidates, c_candidates = [], []
            for cell in self.cells:
                h_k, c_k = cell(x[:, t, :], (h, c))
                h_candidates.append(h_k)
                c_candidates.append(c_k)
            h_stack = torch.stack(h_candidates, dim=1)
            c_stack = torch.stack(c_candidates, dim=1)
            h = torch.sum(belief.unsqueeze(-1) * h_stack, dim=1)
            c = torch.sum(belief.unsqueeze(-1) * c_stack, dim=1)
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class MarkovianHMMMultiScaleLSTM(nn.Module):
    """New experimental model family.

    mode='baseline' : no HMM information.
    mode='concat'   : causal probabilities are ordinary LSTM input features.
    mode='markovian': causal probabilities softly mix state-specific LSTM cells.
    All variants retain the same multi-scale branches, Attention and residual
    head so the switching mechanism is the only intended difference.
    """

    def __init__(self, input_dim: int, seasonal_dim: int, residual_dim: int, hmm_dim: int, horizon: int, hidden_dim: int, mode: str):
        super().__init__()
        if mode not in {"baseline", "concat", "markovian"}:
            raise ValueError(f"Unknown Markovian experiment mode: {mode}")
        self.mode = mode
        lstm_input_dim = input_dim + hmm_dim if mode == "concat" else input_dim
        if mode == "markovian":
            self.short_encoder = MarkovianLSTMEncoder(input_dim, hidden_dim, hmm_dim)
            self.long_encoder = MarkovianLSTMEncoder(input_dim, hidden_dim, hmm_dim)
        else:
            self.short_encoder = nn.LSTM(lstm_input_dim, hidden_dim, batch_first=True)
            self.long_encoder = nn.LSTM(lstm_input_dim, hidden_dim, batch_first=True)
        # Only the proposed state-switching model lets posterior probabilities
        # alter attention scores.  The concat control is deliberately plain.
        attn_hmm_dim = hmm_dim if mode == "markovian" else 0
        self.short_attn = StateConditionedAttention(hidden_dim, attn_hmm_dim)
        self.long_attn = StateConditionedAttention(hidden_dim, attn_hmm_dim)
        self.deep = nn.Sequential(nn.LayerNorm(hidden_dim * 2 + seasonal_dim), nn.Linear(hidden_dim * 2 + seasonal_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.15))
        self.reg_head, self.cls_head = nn.Linear(hidden_dim, horizon), nn.Linear(hidden_dim, 1)
        self.linear_residual = nn.Linear(residual_dim, horizon)

    def _encode(self, encoder, x, probs, attn):
        if self.mode == "concat":
            h, _ = encoder(torch.cat([x, probs], dim=-1))
            return attn(h, None)
        if self.mode == "markovian":
            h = encoder(x, probs)
            return attn(h, probs)
        h, _ = encoder(x)
        return attn(h, None)

    def forward(self, short_x, long_x, seasonal_x, residual_x, short_probs, long_probs):
        short_context, short_alpha = self._encode(self.short_encoder, short_x, short_probs, self.short_attn)
        long_context, _ = self._encode(self.long_encoder, long_x, long_probs, self.long_attn)
        z = self.deep(torch.cat([short_context, long_context, seasonal_x], dim=1))
        return self.reg_head(z) + self.linear_residual(residual_x), self.cls_head(z).squeeze(-1), short_alpha


class LGBMLogitResidualLSTM(nn.Module):
    """只学习 LightGBM 基线遗漏的分类 logit 修正量。"""

    def __init__(self, input_dim: int, state_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.short_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.long_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.deep = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + state_dim + 1),
            nn.Linear(hidden_dim * 2 + state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        self.delta_head = nn.Linear(hidden_dim, 1)

        # 零初始化使训练起点严格等价于 LightGBM：delta_logit = 0。
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, short_x, long_x, current_state, base_logit):
        _, (short_h, _) = self.short_lstm(short_x)
        _, (long_h, _) = self.long_lstm(long_x)
        context = torch.cat(
            [short_h[-1], long_h[-1], current_state, base_logit.unsqueeze(1)],
            dim=1,
        )
        return self.delta_head(self.deep(context)).squeeze(1)
