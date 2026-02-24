"""
Bandit Experiment Runner for Specification Selection.

Provides a clean, reusable function for running the bandit competition
on ANY dataset's scores matrix.  Separates concerns:

1. **Scores matrix construction** -- build_scores_matrix_from_series()
   converts a raw univariate time series into a (T, K) interval-score
   matrix using rolling-window forecasters with different lookback windows.

2. **Bandit competition** -- run_bandit_experiment() takes a pre-computed
   (T, K) scores matrix and a (T, D) context matrix and simulates how
   each method (CTS, Fixed(k), Random, Ensemble, Oracle) would select
   specs and what scores they would receive.

This design means the same bandit runner can be called from
run_diagnostic.py, run_gefcom.py, or any other script, and the
interval-score differences between specs emerge naturally from real
forecasting pipelines rather than being manufactured.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BanditExperimentConfig:
    """Configuration for the bandit specification selection experiment."""

    num_specs: int = 4
    feature_dim: int = 8
    warmup_rounds: int = 20
    exploration_variance: float = 5.0
    prior_precision: float = 0.1
    seed: int = 42


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BanditExperimentResult:
    """Results from running the bandit competition on a scores matrix."""

    # Inputs
    scores_matrix: np.ndarray  # (T, K)
    contexts: np.ndarray       # (T, D)

    # Per-method score arrays (T,)
    cts_scores: np.ndarray
    fixed_scores: Dict[int, np.ndarray]   # fixed_scores[k] = scores when always picking spec k
    random_scores: np.ndarray
    ensemble_scores: np.ndarray
    oracle_scores: np.ndarray

    # Selection tracking
    cts_selections: np.ndarray   # (T,) which spec CTS picked
    optimal_specs: np.ndarray    # (T,) which spec was best (oracle)

    def best_fixed_scores(self) -> np.ndarray:
        """Return scores for the best single fixed spec (in hindsight)."""
        best_k = min(self.fixed_scores, key=lambda k: float(np.mean(self.fixed_scores[k])))
        return self.fixed_scores[best_k]

    def best_fixed_spec(self) -> int:
        """Return index of the best single fixed spec (in hindsight)."""
        return min(self.fixed_scores, key=lambda k: float(np.mean(self.fixed_scores[k])))

    def improvement_over_best_fixed(self) -> float:
        """CTS improvement percentage over best fixed spec (lower score is better)."""
        best_fixed_mean = float(np.mean(self.best_fixed_scores()))
        cts_mean = float(np.mean(self.cts_scores))
        if best_fixed_mean == 0:
            return 0.0
        return 100.0 * (best_fixed_mean - cts_mean) / (best_fixed_mean + 1e-12)

    def median_fixed_scores(self) -> np.ndarray:
        """Return scores for the median-performing fixed spec (no hindsight)."""
        ranked = sorted(self.fixed_scores.keys(),
                        key=lambda k: float(np.mean(self.fixed_scores[k])))
        median_k = ranked[len(ranked) // 2]
        return self.fixed_scores[median_k]

    def mean_fixed_scores(self) -> np.ndarray:
        """Average score across all fixed specs (expected perf of random fixed choice)."""
        return np.mean([self.fixed_scores[k] for k in sorted(self.fixed_scores)], axis=0)

    def improvement_over_random(self) -> float:
        """CTS improvement % over random selection."""
        rand_mean = float(np.mean(self.random_scores))
        cts_mean = float(np.mean(self.cts_scores))
        if rand_mean == 0:
            return 0.0
        return 100.0 * (rand_mean - cts_mean) / (rand_mean + 1e-12)

    def summary(self) -> Dict[str, float]:
        """Return a summary dictionary of mean scores for all methods."""
        result = {
            "cts_mean": float(np.mean(self.cts_scores)),
            "oracle_mean": float(np.mean(self.oracle_scores)),
            "random_mean": float(np.mean(self.random_scores)),
            "ensemble_mean": float(np.mean(self.ensemble_scores)),
            "best_fixed_mean": float(np.mean(self.best_fixed_scores())),
            "best_fixed_spec": self.best_fixed_spec(),
            "improvement_over_best_fixed_pct": self.improvement_over_best_fixed(),
            "improvement_over_random_pct": self.improvement_over_random(),
        }
        for k, scores in self.fixed_scores.items():
            result[f"fixed_{k}_mean"] = float(np.mean(scores))
        return result


# ---------------------------------------------------------------------------
# Main bandit experiment runner
# ---------------------------------------------------------------------------

def run_bandit_experiment(
    scores_matrix: np.ndarray,
    contexts: np.ndarray,
    config: BanditExperimentConfig,
) -> BanditExperimentResult:
    """
    Run the bandit specification selection competition.

    Given a pre-computed (T, K) scores matrix and (T, D) context matrix,
    simulates how each method would select specs and what scores they'd
    get.

    Methods
    -------
    - **CTS** : Linear Thompson Sampling with context.  During warmup it
      cycles through specs round-robin; afterwards it samples from the
      posterior.  Reward = -interval_score (higher is better).
    - **Fixed(k)** : Always pick spec *k* (run for every k).
    - **Random** : Uniform random selection each step.
    - **Ensemble** : Average score across all specs each step.
    - **Oracle** : Best spec at each step (lower bound on achievable
      score; requires future knowledge).

    Parameters
    ----------
    scores_matrix : ndarray, shape (T, K)
        Pre-computed interval scores.  Lower is better.
    contexts : ndarray, shape (T, D)
        Context / feature vectors for each timestep.
    config : BanditExperimentConfig
        Experiment hyper-parameters.

    Returns
    -------
    BanditExperimentResult
    """
    from conformal_ts.models.linear_ts import LinearThompsonSampling

    T, K = scores_matrix.shape
    D = contexts.shape[1]

    assert contexts.shape[0] == T, (
        f"scores_matrix has {T} rows but contexts has {contexts.shape[0]}"
    )
    assert K == config.num_specs, (
        f"scores_matrix has {K} specs but config says {config.num_specs}"
    )
    assert D == config.feature_dim, (
        f"contexts has {D} features but config says {config.feature_dim}"
    )

    # --- set up bandit ---
    bandit = LinearThompsonSampling(
        num_actions=K,
        feature_dim=D,
        prior_precision=config.prior_precision,
        exploration_variance=config.exploration_variance,
        seed=config.seed,
    )

    warmup = min(config.warmup_rounds, T // 5)
    rng = np.random.default_rng(config.seed + 999)

    # --- allocate output arrays ---
    cts_scores = np.zeros(T)
    cts_selections = np.zeros(T, dtype=int)
    random_scores = np.zeros(T)
    ensemble_scores = np.zeros(T)
    oracle_scores = np.zeros(T)
    optimal_specs = np.zeros(T, dtype=int)

    # Fixed baselines: one array per spec
    fixed_scores: Dict[int, np.ndarray] = {k: np.zeros(T) for k in range(K)}

    # --- simulation loop ---
    for t in range(T):
        ctx = contexts[t]
        row = scores_matrix[t]

        # Oracle
        best_k = int(np.argmin(row))
        optimal_specs[t] = best_k
        oracle_scores[t] = row[best_k]

        # CTS
        if t < warmup:
            cts_action = t % K
        else:
            cts_action = bandit.select_action(ctx)
        cts_selections[t] = cts_action
        cts_scores[t] = row[cts_action]
        # Reward = negative interval score (higher is better)
        bandit.update(cts_action, ctx, -row[cts_action])

        # Fixed baselines
        for k in range(K):
            fixed_scores[k][t] = row[k]

        # Random
        random_scores[t] = row[rng.integers(0, K)]

        # Ensemble (average across specs)
        ensemble_scores[t] = float(np.mean(row))

    return BanditExperimentResult(
        scores_matrix=scores_matrix,
        contexts=contexts,
        cts_scores=cts_scores,
        fixed_scores=fixed_scores,
        random_scores=random_scores,
        ensemble_scores=ensemble_scores,
        oracle_scores=oracle_scores,
        cts_selections=cts_selections,
        optimal_specs=optimal_specs,
    )


# ---------------------------------------------------------------------------
# Helper: build scores matrix from a univariate time series
# ---------------------------------------------------------------------------

def build_scores_matrix_from_series(
    series: np.ndarray,
    lookback_windows: List[int],
    alpha: float = 0.10,
    min_history: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a (T', K) scores matrix from a univariate time series.

    For each timestep *t* (starting from ``min_history``) and each
    lookback window *w_k*:

    1. Compute rolling mean and std over the last *w_k* observations.
    2. Form a prediction interval::

           [mean - z_{1-alpha/2} * std,  mean + z_{1-alpha/2} * std]

    3. Score against the true observation at *t* using the interval
       score (Gneiting & Raftery, 2007).

    Parameters
    ----------
    series : ndarray, shape (T,)
        Raw univariate time series.
    lookback_windows : list of int
        Each element defines a "spec" -- the lookback window length.
    alpha : float
        Miscoverage rate for the prediction interval (default 0.10
        for 90 % intervals).
    min_history : int
        Minimum number of observations before scoring begins.  The
        returned arrays start at index ``min_history`` of the original
        series.

    Returns
    -------
    scores_matrix : ndarray, shape (T', K)
        Interval scores.  T' = T - min_history.
    contexts : ndarray, shape (T', D)
        Context features derived from the series history.  D = 8.
    """
    from conformal_ts.evaluation.metrics import interval_score

    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)

    T = len(series)
    K = len(lookback_windows)
    T_prime = T - min_history

    if T_prime <= 0:
        raise ValueError(
            f"Series length {T} is <= min_history {min_history}; "
            "nothing to score."
        )

    scores_matrix = np.zeros((T_prime, K))
    feature_dim = 8
    contexts = np.zeros((T_prime, feature_dim))

    for idx in range(T_prime):
        t = min_history + idx
        y_true = series[t]

        # --- build context features from history ---
        ctx = np.zeros(feature_dim)
        ctx[0] = 1.0  # bias

        # Short-window statistics (last 5)
        short_win = series[max(0, t - 5):t]
        if len(short_win) > 0:
            ctx[1] = np.mean(short_win)
            ctx[2] = np.std(short_win) + 1e-8
        # Medium-window statistics (last 20)
        med_win = series[max(0, t - 20):t]
        if len(med_win) > 0:
            ctx[3] = np.mean(med_win)
            ctx[4] = np.std(med_win) + 1e-8
        # Long-window statistics (last 50)
        long_win = series[max(0, t - 50):t]
        if len(long_win) > 0:
            ctx[5] = np.mean(long_win)
            ctx[6] = np.std(long_win) + 1e-8
        # Last value
        ctx[7] = series[t - 1]

        contexts[idx] = ctx

        # --- score each spec ---
        for k, w in enumerate(lookback_windows):
            window = series[max(0, t - w):t]
            mu = float(np.mean(window))
            sigma = float(np.std(window)) + 1e-8

            lower_k = mu - z * sigma
            upper_k = mu + z * sigma

            scores_matrix[idx, k] = interval_score(
                np.array([lower_k]),
                np.array([upper_k]),
                np.array([y_true]),
                alpha=alpha,
            )[0]

    return scores_matrix, contexts


# ---------------------------------------------------------------------------
# Convenience: generate a regime-switching time series
# ---------------------------------------------------------------------------

def generate_regime_switching_series(
    n_steps: int,
    regime_persistence: float = 0.95,
    num_regimes: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a univariate time series with regime-switching dynamics.

    Each regime has its own mean, volatility, and AR(1) coefficient, so
    different lookback windows will naturally excel in different regimes.

    Parameters
    ----------
    n_steps : int
        Length of the output series.
    regime_persistence : float
        Probability of staying in the current regime at each step.
    num_regimes : int
        Number of distinct regimes.
    seed : int
        Random seed.

    Returns
    -------
    series : ndarray, shape (n_steps,)
    """
    rng = np.random.default_rng(seed)

    # Regime parameters -- deliberately varied so that different
    # lookback windows are better in different regimes.
    regime_means = rng.normal(0, 1.0, size=num_regimes)
    regime_vols = 0.3 + rng.uniform(0, 1.5, size=num_regimes)
    regime_ar = rng.uniform(-0.5, 0.8, size=num_regimes)

    series = np.zeros(n_steps)
    current_regime = rng.integers(0, num_regimes)

    for t in range(n_steps):
        mean = regime_means[current_regime]
        vol = regime_vols[current_regime]
        ar = regime_ar[current_regime]

        ar_contrib = ar * series[t - 1] if t > 0 else 0.0
        series[t] = mean + ar_contrib + rng.normal(0, vol)

        # Regime transition
        if rng.random() > regime_persistence:
            new_regime = rng.integers(0, num_regimes)
            while new_regime == current_regime and num_regimes > 1:
                new_regime = rng.integers(0, num_regimes)
            current_regime = new_regime

    return series


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== bandit_experiment self-test ===\n")

    # 1. Generate a regime-switching time series
    series = generate_regime_switching_series(
        n_steps=600, regime_persistence=0.90, seed=123,
    )
    print(f"Series length: {len(series)}, mean={series.mean():.3f}, std={series.std():.3f}")

    # 2. Build scores matrix with 4 lookback windows
    lookbacks = [10, 25, 50, 100]
    scores_matrix, contexts = build_scores_matrix_from_series(
        series, lookback_windows=lookbacks, alpha=0.10, min_history=50,
    )
    print(f"Scores matrix shape: {scores_matrix.shape}")
    print(f"Contexts shape:      {contexts.shape}")
    print(f"Mean scores per spec: {scores_matrix.mean(axis=0).round(3)}")

    # 3. Run bandit experiment
    T, K = scores_matrix.shape
    D = contexts.shape[1]

    config = BanditExperimentConfig(
        num_specs=K,
        feature_dim=D,
        warmup_rounds=20,
        seed=123,
    )
    result = run_bandit_experiment(scores_matrix, contexts, config)

    summary = result.summary()
    print("\n--- Summary ---")
    for key, val in summary.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")

    print(f"\n  CTS improvement over best fixed: "
          f"{result.improvement_over_best_fixed():+.2f}%")
    print("\nSelf-test passed.")
