"""
Interval Scores and Evaluation Metrics.

This module implements:
- Interval scores (proper scoring rule)
- Coverage metrics
- Width metrics
- Diebold-Mariano tests
- Bootstrap confidence intervals

Reference:
    Gneiting & Raftery (2007) - Strictly Proper Scoring Rules
    Bracher et al. (2021) - Evaluating epidemic forecasts
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from scipy import stats
from dataclasses import dataclass


def interval_score(
    lower: np.ndarray,
    upper: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.10
) -> np.ndarray:
    """
    Compute interval score (proper scoring rule for prediction intervals).
    
    IS_alpha(l, u, y) = (u - l) 
                        + (2/alpha) * (l - y) * I(y < l)
                        + (2/alpha) * (y - u) * I(y > u)
    
    The interval score penalizes:
    - Wide intervals (first term)
    - Undercoverage below lower bound (second term)
    - Undercoverage above upper bound (third term)
    
    Lower scores are better.
    
    Args:
        lower: Lower bounds of intervals
        upper: Upper bounds of intervals
        y: True values
        alpha: Miscoverage rate (default: 0.10 for 90% intervals)
    
    Returns:
        Interval scores for each sample
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    y = np.asarray(y)
    
    # Ensure shapes match
    assert lower.shape == upper.shape == y.shape, \
        f"Shape mismatch: {lower.shape}, {upper.shape}, {y.shape}"
    
    # Width penalty
    width = upper - lower
    
    # Undercoverage penalties
    below_mask = y < lower
    above_mask = y > upper
    
    penalty_below = (2.0 / alpha) * (lower - y) * below_mask
    penalty_above = (2.0 / alpha) * (y - upper) * above_mask
    
    scores = width + penalty_below + penalty_above
    return scores


def interval_score_reward(
    lower: float,
    upper: float,
    y: float,
    alpha: float = 0.10,
    scale: float = 1.0,
    clip_min: float = -10.0,
    clip_max: float = 0.0
) -> float:
    """
    Convert interval score to reward for bandit algorithm.
    
    Reward = -scale * IS_alpha(l, u, y)
    
    Optionally clips to prevent extreme values.
    
    Args:
        lower: Lower bound
        upper: Upper bound
        y: True value
        alpha: Miscoverage rate
        scale: Reward scaling factor
        clip_min: Minimum reward
        clip_max: Maximum reward (should be <= 0)
    
    Returns:
        Reward (higher is better)
    """
    score = interval_score(
        np.array([lower]),
        np.array([upper]),
        np.array([y]),
        alpha
    )[0]
    
    reward = -scale * score
    
    if clip_min is not None and clip_max is not None:
        reward = np.clip(reward, clip_min, clip_max)
    
    return float(reward)


def coverage_rate(
    lower: np.ndarray,
    upper: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Compute empirical coverage rate.
    
    Coverage = mean(I(l <= y <= u))
    
    Args:
        lower: Lower bounds
        upper: Upper bounds
        y: True values
    
    Returns:
        Coverage rate in [0, 1]
    """
    covered = (y >= lower) & (y <= upper)
    return float(np.mean(covered))


def mean_interval_width(
    lower: np.ndarray,
    upper: np.ndarray
) -> float:
    """
    Compute mean interval width.
    
    Args:
        lower: Lower bounds
        upper: Upper bounds
    
    Returns:
        Mean width
    """
    return float(np.mean(upper - lower))


def winkler_score(
    lower: np.ndarray,
    upper: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.10
) -> np.ndarray:
    """
    Compute Winkler score (same as interval score, different name).
    
    Included for compatibility with forecasting literature.
    """
    return interval_score(lower, upper, y, alpha)


def quantile_score(
    predicted: np.ndarray,
    y: np.ndarray,
    tau: float
) -> np.ndarray:
    """
    Compute quantile score (pinball loss).
    
    QS_tau(q, y) = (y - q) * (tau - I(y < q))
    
    Args:
        predicted: Predicted quantile values
        y: True values
        tau: Quantile level
    
    Returns:
        Quantile scores
    """
    residual = y - predicted
    return residual * (tau - (y < predicted).astype(float))


def crps(
    lower: np.ndarray,
    upper: np.ndarray,
    y: np.ndarray,
    n_quantiles: int = 99
) -> np.ndarray:
    """
    Approximate CRPS from prediction intervals.
    
    Assumes uniform distribution between lower and upper bounds.
    This is a rough approximation useful for comparison.
    
    Args:
        lower: Lower bounds
        upper: Upper bounds
        y: True values
        n_quantiles: Number of quantiles for approximation
    
    Returns:
        Approximate CRPS values
    """
    # Quantile levels
    taus = np.linspace(0.01, 0.99, n_quantiles)
    
    # Approximate quantile predictions assuming uniform
    crps_values = np.zeros(len(y))
    
    for i, (l, u, yi) in enumerate(zip(lower, upper, y)):
        # Quantile predictions under uniform assumption
        quantiles = l + taus * (u - l)
        
        # Average quantile score
        qs = quantile_score(quantiles, np.full(n_quantiles, yi), taus)
        crps_values[i] = 2 * np.mean(qs)
    
    return crps_values


@dataclass
class IntervalMetrics:
    """Container for interval evaluation metrics."""
    
    mean_interval_score: float
    median_interval_score: float
    coverage_rate: float
    mean_width: float
    median_width: float
    width_90th_percentile: float
    num_samples: int
    
    # Optional: by quantile
    lower_quantile_score: Optional[float] = None
    upper_quantile_score: Optional[float] = None


def compute_interval_metrics(
    lower: np.ndarray,
    upper: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.10
) -> IntervalMetrics:
    """
    Compute comprehensive interval metrics.
    
    Args:
        lower: Lower bounds
        upper: Upper bounds
        y: True values
        alpha: Miscoverage rate
    
    Returns:
        IntervalMetrics object
    """
    scores = interval_score(lower, upper, y, alpha)
    widths = upper - lower
    
    return IntervalMetrics(
        mean_interval_score=float(np.mean(scores)),
        median_interval_score=float(np.median(scores)),
        coverage_rate=coverage_rate(lower, upper, y),
        mean_width=float(np.mean(widths)),
        median_width=float(np.median(widths)),
        width_90th_percentile=float(np.percentile(widths, 90)),
        num_samples=len(y),
    )


def diebold_mariano_test(
    scores_1: np.ndarray,
    scores_2: np.ndarray,
    alternative: str = 'two-sided',
    h: int = 1
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for comparing forecast accuracy.
    
    Tests H0: E[d_t] = 0 where d_t = score_1_t - score_2_t
    
    Uses Newey-West HAC standard errors for time series dependence.
    
    Args:
        scores_1: Scores from first method
        scores_2: Scores from second method
        alternative: 'two-sided', 'less', or 'greater'
        h: Forecast horizon (for HAC bandwidth)
    
    Returns:
        (test_statistic, p_value)
    """
    d = scores_1 - scores_2
    n = len(d)
    
    # Mean difference
    d_bar = np.mean(d)
    
    # Newey-West HAC variance estimator
    # Bandwidth = h - 1 (common choice for h-step ahead forecasts)
    bandwidth = max(h - 1, 0)
    
    # Autocovariances
    gamma_0 = np.var(d, ddof=1)
    gamma = [gamma_0]
    
    for j in range(1, bandwidth + 1):
        gamma_j = np.mean((d[j:] - d_bar) * (d[:-j] - d_bar))
        gamma.append(gamma_j)
    
    # HAC variance
    weights = 1 - np.arange(1, bandwidth + 1) / (bandwidth + 1)
    var_d = gamma_0 + 2 * np.sum(weights * gamma[1:])
    var_d = max(var_d, 1e-10)  # Numerical stability
    
    # Test statistic
    se = np.sqrt(var_d / n)
    t_stat = d_bar / se
    
    # P-value
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    elif alternative == 'less':
        p_value = stats.norm.cdf(t_stat)
    elif alternative == 'greater':
        p_value = 1 - stats.norm.cdf(t_stat)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    
    return float(t_stat), float(p_value)


def bootstrap_confidence_interval(
    scores: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    block_size: int = 5,
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Moving block bootstrap confidence interval for mean score.
    
    Accounts for time series dependence using block bootstrap.
    
    Args:
        scores: Score series
        confidence: Confidence level (e.g., 0.95)
        n_bootstrap: Number of bootstrap samples
        block_size: Size of moving blocks
        seed: Random seed
    
    Returns:
        (mean, lower_ci, upper_ci)
    """
    rng = np.random.default_rng(seed)
    n = len(scores)
    
    # Number of blocks needed
    n_blocks = int(np.ceil(n / block_size))
    
    # Bootstrap means
    bootstrap_means = np.zeros(n_bootstrap)
    
    for b in range(n_bootstrap):
        # Sample block starting positions
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        
        # Construct bootstrap sample
        sample = np.concatenate([
            scores[start:start + block_size] for start in starts
        ])[:n]
        
        bootstrap_means[b] = np.mean(sample)
    
    # Percentile confidence interval
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return float(np.mean(scores)), float(lower), float(upper)


def compute_regret(
    selected_scores: np.ndarray,
    best_scores: np.ndarray,
    cumulative: bool = True
) -> np.ndarray:
    """
    Compute regret relative to oracle.
    
    Regret_t = score_selected_t - score_best_t
    
    Args:
        selected_scores: Scores from selected actions
        best_scores: Scores from oracle (best action at each step)
        cumulative: Whether to return cumulative regret
    
    Returns:
        Regret series
    """
    instant_regret = selected_scores - best_scores
    
    if cumulative:
        return np.cumsum(instant_regret)
    return instant_regret


def compare_methods(
    method_scores: Dict[str, np.ndarray],
    baseline_name: str,
    alpha: float = 0.05
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple methods to a baseline using DM tests.
    
    Args:
        method_scores: Dictionary mapping method names to score arrays
        baseline_name: Name of baseline method
        alpha: Significance level
    
    Returns:
        Comparison results for each method
    """
    baseline_scores = method_scores[baseline_name]
    results = {}
    
    for name, scores in method_scores.items():
        if name == baseline_name:
            continue
        
        # DM test
        t_stat, p_value = diebold_mariano_test(scores, baseline_scores)
        
        # Improvement
        mean_diff = np.mean(scores) - np.mean(baseline_scores)
        pct_improvement = -100 * mean_diff / np.mean(baseline_scores)
        
        # Bootstrap CI for difference
        diff_mean, diff_lower, diff_upper = bootstrap_confidence_interval(
            scores - baseline_scores
        )
        
        results[name] = {
            'mean_score': float(np.mean(scores)),
            'mean_diff': float(mean_diff),
            'pct_improvement': float(pct_improvement),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'diff_ci_lower': diff_lower,
            'diff_ci_upper': diff_upper,
        }
    
    return results


class OnlineMetrics:
    """
    Track metrics online during training.
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize online metrics tracker.
        
        Args:
            window_size: Size of rolling window for statistics
        """
        self.window_size = window_size
        self._scores: List[float] = []
        self._coverages: List[bool] = []
        self._widths: List[float] = []
        self._rewards: List[float] = []
    
    def update(
        self,
        lower: float,
        upper: float,
        y: float,
        reward: float
    ):
        """
        Update metrics with new observation.
        """
        score = interval_score(
            np.array([lower]),
            np.array([upper]),
            np.array([y])
        )[0]
        
        covered = lower <= y <= upper
        width = upper - lower
        
        self._scores.append(score)
        self._coverages.append(covered)
        self._widths.append(width)
        self._rewards.append(reward)
        
        # Trim to window
        if len(self._scores) > self.window_size:
            self._scores = self._scores[-self.window_size:]
            self._coverages = self._coverages[-self.window_size:]
            self._widths = self._widths[-self.window_size:]
            self._rewards = self._rewards[-self.window_size:]
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current statistics."""
        if len(self._scores) == 0:
            return {
                'mean_score': 0.0,
                'coverage': 0.0,
                'mean_width': 0.0,
                'mean_reward': 0.0,
                'num_samples': 0,
            }
        
        return {
            'mean_score': float(np.mean(self._scores)),
            'coverage': float(np.mean(self._coverages)),
            'mean_width': float(np.mean(self._widths)),
            'mean_reward': float(np.mean(self._rewards)),
            'num_samples': len(self._scores),
        }
    
    def reset(self):
        """Reset all metrics."""
        self._scores = []
        self._coverages = []
        self._widths = []
        self._rewards = []


if __name__ == "__main__":
    # Test the implementations
    np.random.seed(42)
    
    # Generate synthetic intervals and observations
    n = 1000
    y_true = np.random.randn(n)
    
    # Method 1: Well-calibrated intervals
    lower_1 = y_true - 1.65
    upper_1 = y_true + 1.65
    
    # Add noise to make realistic
    lower_1 += np.random.randn(n) * 0.3
    upper_1 += np.random.randn(n) * 0.3
    
    # Method 2: Wider but better coverage
    lower_2 = y_true - 2.0 + np.random.randn(n) * 0.2
    upper_2 = y_true + 2.0 + np.random.randn(n) * 0.2
    
    # Compute metrics
    print("Method 1 (narrow):")
    metrics_1 = compute_interval_metrics(lower_1, upper_1, y_true)
    print(f"  Mean IS: {metrics_1.mean_interval_score:.4f}")
    print(f"  Coverage: {metrics_1.coverage_rate:.2%}")
    print(f"  Mean width: {metrics_1.mean_width:.4f}")
    
    print("\nMethod 2 (wide):")
    metrics_2 = compute_interval_metrics(lower_2, upper_2, y_true)
    print(f"  Mean IS: {metrics_2.mean_interval_score:.4f}")
    print(f"  Coverage: {metrics_2.coverage_rate:.2%}")
    print(f"  Mean width: {metrics_2.mean_width:.4f}")
    
    # Diebold-Mariano test
    scores_1 = interval_score(lower_1, upper_1, y_true)
    scores_2 = interval_score(lower_2, upper_2, y_true)
    
    t_stat, p_value = diebold_mariano_test(scores_1, scores_2)
    print(f"\nDiebold-Mariano test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    
    # Bootstrap CI
    mean, lower_ci, upper_ci = bootstrap_confidence_interval(scores_1)
    print(f"\nBootstrap 95% CI for Method 1 mean score:")
    print(f"  {mean:.4f} [{lower_ci:.4f}, {upper_ci:.4f}]")
    
    # Test reward computation
    reward = interval_score_reward(
        lower_1[0], upper_1[0], y_true[0],
        alpha=0.10, scale=1.0
    )
    print(f"\nExample reward: {reward:.4f}")
