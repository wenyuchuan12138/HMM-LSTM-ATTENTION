from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler

from .utils import save_json


def _bic_aic(model: GaussianHMM, x: np.ndarray) -> tuple[float, float, float]:
    """Return likelihood, AIC and BIC computed on the training observations only."""
    n, d = x.shape
    k = model.n_components
    ll = model.score(x)
    # Initial probabilities + transition matrix + Gaussian means/covariances.
    n_params = (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) / 2
    return float(ll), float(-2 * ll + 2 * n_params), float(-2 * ll + n_params * np.log(n))


def causal_filter_probabilities(model: GaussianHMM, x: np.ndarray) -> np.ndarray:
    """Compute p(S_t | x_1, ..., x_t) by the forward algorithm.

    ``GaussianHMM.predict_proba`` uses forward-backward smoothing and therefore
    lets the state probability at time t see future observations.  That is useful
    for retrospective state labelling but invalid as a forecasting feature.  This
    function is intentionally causal: each row uses the current and earlier data
    only.
    """
    log_emission = model._compute_log_likelihood(x)  # [time, K], HMM's fitted Gaussian density
    log_start = np.log(np.clip(model.startprob_, 1e-12, 1.0))
    log_trans = np.log(np.clip(model.transmat_, 1e-12, 1.0))

    filtered = np.zeros((len(x), model.n_components), dtype=np.float64)
    log_alpha = log_start + log_emission[0]
    log_alpha -= logsumexp(log_alpha)
    filtered[0] = np.exp(log_alpha)

    for t in range(1, len(x)):
        # log p(S_t=j | x_1:t-1) = logsumexp_i[log alpha_{t-1,i} + log A_ij]
        log_predict = logsumexp(log_alpha[:, None] + log_trans, axis=0)
        log_alpha = log_emission[t] + log_predict
        log_alpha -= logsumexp(log_alpha)
        filtered[t] = np.exp(log_alpha)
    return filtered


def fit_hmm_features(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    candidate_k: tuple[int, ...],
    seeds: tuple[int, ...],
    max_iter: int,
    results_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    """Fit HMM on training data and append *causal* state features to all rows."""
    hmm_cols = [
        "carbon_intensity_gCO2eq_per_kWh",
        "ci_diff_1h",
        "ci_rolling_std_24h",
        "renewable_percentage",
    ]
    hmm_cols = [c for c in hmm_cols if c in df.columns]
    if not hmm_cols:
        raise ValueError("No HMM input features are available.")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(df.loc[train_idx, hmm_cols])
    x_all = scaler.transform(df[hmm_cols])

    rows, fitted = [], []
    for k in candidate_k:
        for seed in seeds:
            model = GaussianHMM(
                n_components=k,
                covariance_type="full",
                n_iter=max_iter,
                random_state=seed,
                min_covar=1e-3,
            )
            model.fit(x_train)
            states = model.predict(x_train)
            min_share = float(np.bincount(states, minlength=k).min() / len(states))
            ll, aic, bic = _bic_aic(model, x_train)
            rows.append({"K": k, "seed": seed, "log_likelihood": ll, "AIC": aic, "BIC": bic, "min_state_share": min_share})
            fitted.append((bic, -min_share, k, seed, model))

    selection = pd.DataFrame(rows).sort_values(["BIC", "min_state_share"], ascending=[True, False])
    selection.to_csv(results_dir / "hmm_model_selection.csv", index=False)
    _, _, best_k, best_seed, best_model = min(fitted, key=lambda item: (item[0], item[1]))

    # State labels are ordered by training-period mean carbon intensity, so the
    # largest index always means the highest-carbon regime.
    train_viterbi = best_model.predict(x_train)
    state_means = [df.loc[train_idx[train_viterbi == state], "carbon_intensity_gCO2eq_per_kWh"].mean() for state in range(best_k)]
    order = np.argsort(state_means)

    # Crucial forecasting fix: never call predict_proba(x_all) here.
    raw_filtered = causal_filter_probabilities(best_model, x_all)
    probs = raw_filtered[:, order]
    states = probs.argmax(axis=1)
    transmat = best_model.transmat_[order][:, order]

    out = df.copy()
    out["hmm_state"] = states
    for i in range(best_k):
        out[f"hmm_prob_{i}"] = probs[:, i]
    out["hmm_high_prob"] = probs[:, -1]
    # Given the causal belief at t, this is the one-step probability of entering
    # the highest-carbon regime at t+1.
    out["hmm_transition_to_high"] = probs @ transmat[:, -1]
    out["hmm_entropy"] = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
    out["hmm_state_duration"] = out.groupby((out["hmm_state"] != out["hmm_state"].shift()).cumsum()).cumcount() + 1

    summary = (
        out.groupby("hmm_state")
        .agg(
            carbon_mean=("carbon_intensity_gCO2eq_per_kWh", "mean"),
            carbon_std=("carbon_intensity_gCO2eq_per_kWh", "std"),
            mean_duration=("hmm_state_duration", "mean"),
            sample_share=("hmm_state", lambda s: len(s) / len(out)),
        )
        .reset_index()
    )
    summary.to_csv(results_dir / "hmm_state_summary.csv", index=False)

    params = {
        "selected_K": int(best_k),
        "selected_seed": int(best_seed),
        "input_features": hmm_cols,
        "posterior_type": "causal_forward_filtering_p(S_t|x_1:t)",
        "startprob": best_model.startprob_[order].tolist(),
        "transmat": transmat.tolist(),
        "means": best_model.means_[order].tolist(),
        "covars": best_model.covars_[order].tolist(),
    }
    save_json(params, results_dir / "hmm_parameters.json")

    posterior_cols = ["timestamp", "hmm_state"] + [f"hmm_prob_{i}" for i in range(best_k)]
    out.iloc[train_idx][posterior_cols].to_csv(results_dir / "hmm_train_posteriors.csv", index=False)
    out.iloc[~out.index.isin(train_idx)][posterior_cols].to_csv(results_dir / "hmm_future_posteriors.csv", index=False)
    return out, params
