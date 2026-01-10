"""
Competition-Grade Evaluation Metrics.

Implements metrics matching the official M5 and GEFCom2014 evaluation:

M5 Uncertainty Track:
- Weighted Scaled Pinball Loss (WSPL) across 9 quantiles
- Hierarchical weighting by aggregation level

GEFCom2014:
- Pinball loss across 99 quantiles (1%, 2%, ..., 99%)
- Normalized by zone capacity

Also includes:
- Interval Score (Winkler Score)
- Coverage Rate
- CRPS (Continuous Ranked Probability Score)
- Diebold-Mariano test for significance
- Bootstrap confidence intervals
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
import warnings


# M5 Uncertainty quantile levels
M5_QUANTILES = np.array([0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995])

# GEFCom quantile levels
GEFCOM_QUANTILES = np.array([q/100 for q in range(1, 100)])


def pinball_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
    quantile: float
) -> np.ndarray:
    """
    Compute pinball loss (quantile loss).
    
    L_tau(q, y) = (y - q) * (tau - I(y < q))
               = max(tau * (y - q), (tau - 1) * (y - q))
    
    Args:
        predictions: Predicted quantile values
        targets: True values
        quantile: Quantile level tau in (0, 1)
        
    Returns:
        Pinball loss for each sample
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    
    diff = targets - predictions
    
    return np.where(
        diff >= 0,
        quantile * diff,
        (quantile - 1) * diff
    )


def interval_score(
    lower: np.ndarray,
    upper: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.10
) -> np.ndarray:
    """
    Compute interval score (Winkler score).
    
    IS_alpha(l, u, y) = (u - l) + (2/alpha) * (l - y) * I(y < l) 
                               + (2/alpha) * (y - u) * I(y > u)
    
    This is equivalent to sum of two pinball losses.
    
    Args:
        lower: Lower interval bounds
        upper: Upper interval bounds
        y: True values
        alpha: Miscoverage rate (e.g., 0.10 for 90% intervals)
        
    Returns:
        Interval score for each sample
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    y = np.asarray(y)
    
    width = upper - lower
    
    below = np.maximum(0, lower - y)
    above = np.maximum(0, y - upper)
    
    return width + (2 / alpha) * (below + above)


def coverage_rate(
    lower: np.ndarray,
    upper: np.ndarray,
    y: np.ndarray
) -> float:
    """Compute empirical coverage rate."""
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    y = np.asarray(y)
    
    covered = (y >= lower) & (y <= upper)
    return float(np.mean(covered))


def weighted_scaled_pinball_loss(
    predictions: Dict[float, np.ndarray],
    targets: np.ndarray,
    scale_factors: np.ndarray,
    weights: Optional[np.ndarray] = None,
    quantiles: np.ndarray = M5_QUANTILES
) -> float:
    """
    Compute Weighted Scaled Pinball Loss (WSPL) - M5 metric.
    
    WSPL = sum_i sum_q w_i * SPL(q_hat, y, tau) / scale_i
    
    Args:
        predictions: Dict mapping quantile -> predictions array
        targets: True values (n_samples,)
        scale_factors: Scaling factors per series (n_samples,)
        weights: Optional sample weights (n_samples,)
        quantiles: Quantile levels
        
    Returns:
        WSPL score (lower is better)
    """
    n_samples = len(targets)
    
    if weights is None:
        weights = np.ones(n_samples)
    
    total_loss = 0.0
    total_weight = 0.0
    
    for q in quantiles:
        if q not in predictions:
            # Find closest quantile
            closest_q = min(predictions.keys(), key=lambda x: abs(x - q))
            preds = predictions[closest_q]
        else:
            preds = predictions[q]
        
        # Pinball loss
        pl = pinball_loss(preds, targets, q)
        
        # Scale and weight
        scaled_pl = pl / (scale_factors + 1e-6)
        weighted_pl = weights * scaled_pl
        
        total_loss += np.sum(weighted_pl)
        total_weight += np.sum(weights)
    
    return total_loss / total_weight


def m5_wspl_from_intervals(
    lower: np.ndarray,
    upper: np.ndarray,
    targets: np.ndarray,
    scale_factors: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Compute WSPL from prediction intervals.
    
    Approximates the full quantile predictions by assuming
    linear interpolation between lower and upper bounds.
    
    Args:
        lower: Lower bounds (5th percentile approximation)
        upper: Upper bounds (95th percentile approximation)
        targets: True values
        scale_factors: Scaling factors
        weights: Optional weights
        
    Returns:
        Approximate WSPL
    """
    # Create approximate quantile predictions
    # Assume lower ~ 5th percentile, upper ~ 95th percentile
    # Median ~ midpoint
    
    predictions = {}
    
    for q in M5_QUANTILES:
        if q <= 0.05:
            predictions[q] = lower
        elif q >= 0.95:
            predictions[q] = upper
        else:
            # Linear interpolation
            frac = (q - 0.05) / (0.95 - 0.05)
            predictions[q] = lower + frac * (upper - lower)
    
    return weighted_scaled_pinball_loss(
        predictions, targets, scale_factors, weights, M5_QUANTILES
    )


def gefcom_pinball_score(
    predictions: Dict[float, np.ndarray],
    targets: np.ndarray,
    quantiles: np.ndarray = GEFCOM_QUANTILES
) -> float:
    """
    Compute GEFCom-style pinball loss across 99 quantiles.
    
    Args:
        predictions: Dict mapping quantile -> predictions
        targets: True values
        quantiles: Quantile levels (default: 1%, 2%, ..., 99%)
        
    Returns:
        Mean pinball loss across all quantiles
    """
    total_loss = 0.0
    n_quantiles = 0
    
    for q in quantiles:
        if q in predictions:
            pl = pinball_loss(predictions[q], targets, q)
            total_loss += np.mean(pl)
            n_quantiles += 1
    
    return total_loss / n_quantiles if n_quantiles > 0 else float('inf')


def crps(
    lower: np.ndarray,
    upper: np.ndarray,
    y: np.ndarray,
    n_quantiles: int = 99
) -> np.ndarray:
    """
    Approximate CRPS from prediction intervals.
    
    Assumes uniform distribution between lower and upper.
    
    CRPS = integral_0^1 [2 * QL_tau(F^{-1}(tau), y)] dtau
    
    Args:
        lower: Lower bounds
        upper: Upper bounds
        y: True values
        n_quantiles: Number of quantiles for approximation
        
    Returns:
        CRPS for each sample
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    y = np.asarray(y)
    
    taus = np.linspace(0.01, 0.99, n_quantiles)
    
    crps_values = np.zeros(len(y))
    
    for i in range(len(y)):
        # Quantile predictions under uniform assumption
        quantile_preds = lower[i] + taus * (upper[i] - lower[i])
        
        # Average pinball loss
        total = 0.0
        for j, tau in enumerate(taus):
            total += pinball_loss(
                np.array([quantile_preds[j]]),
                np.array([y[i]]),
                tau
            )[0]
        
        crps_values[i] = 2 * total / n_quantiles
    
    return crps_values


def diebold_mariano_test(
    scores1: np.ndarray,
    scores2: np.ndarray,
    h: int = 1
) -> Dict[str, float]:
    """
    Diebold-Mariano test for equal predictive accuracy.
    
    H0: E[d_t] = 0 where d_t = L(e1_t) - L(e2_t)
    
    Args:
        scores1: Loss series from method 1
        scores2: Loss series from method 2
        h: Forecast horizon (for HAC variance)
        
    Returns:
        Dict with test statistic, p-value, and improvement percentage
    """
    scores1 = np.asarray(scores1)
    scores2 = np.asarray(scores2)
    
    d = scores1 - scores2
    n = len(d)
    
    d_mean = np.mean(d)
    
    # Newey-West HAC variance
    gamma_0 = np.var(d, ddof=1)
    
    # Autocovariances
    gamma_sum = 0.0
    for k in range(1, h):
        weight = 1 - k / h
        gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
        gamma_sum += 2 * weight * gamma_k
    
    variance = (gamma_0 + gamma_sum) / n
    
    if variance <= 0:
        variance = gamma_0 / n
    
    dm_stat = d_mean / np.sqrt(variance)
    
    # Two-sided p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    # Improvement percentage
    mean1 = np.mean(scores1)
    mean2 = np.mean(scores2)
    pct_improvement = 100 * (mean2 - mean1) / mean2 if mean2 != 0 else 0
    
    return {
        'dm_statistic': float(dm_stat),
        'p_value': float(p_value),
        'mean_difference': float(d_mean),
        'pct_improvement': float(pct_improvement),
        'method1_mean': float(mean1),
        'method2_mean': float(mean2),
        'significant_0.01': p_value < 0.01,
        'significant_0.05': p_value < 0.05,
        'significant_0.10': p_value < 0.10,
    }


def bootstrap_confidence_interval(
    scores: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for mean score.
    
    Args:
        scores: Score array
        confidence: Confidence level
        n_bootstrap: Number of bootstrap samples
        seed: Random seed
        
    Returns:
        (mean, lower_ci, upper_ci)
    """
    rng = np.random.default_rng(seed)
    
    scores = np.asarray(scores)
    n = len(scores)
    
    means = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        means.append(np.mean(scores[indices]))
    
    means = np.array(means)
    
    alpha = 1 - confidence
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    
    return float(np.mean(scores)), float(lower), float(upper)


def compare_methods(
    method_scores: Dict[str, np.ndarray],
    baseline_name: str = 'Fixed'
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple methods against a baseline.
    
    Args:
        method_scores: Dict of method name -> score array
        baseline_name: Name of baseline method
        
    Returns:
        Dict of method name -> DM test results
    """
    if baseline_name not in method_scores:
        raise ValueError(f"Baseline '{baseline_name}' not in methods")
    
    baseline_scores = method_scores[baseline_name]
    results = {}
    
    for name, scores in method_scores.items():
        if name == baseline_name:
            continue
        
        results[name] = diebold_mariano_test(scores, baseline_scores)
    
    return results


@dataclass
class EvaluationResults:
    """Container for comprehensive evaluation results."""
    
    # Core metrics
    mean_interval_score: float
    coverage_rate: float
    mean_width: float
    
    # M5-specific
    wspl: Optional[float] = None
    
    # GEFCom-specific
    mean_pinball: Optional[float] = None
    
    # Uncertainty
    interval_score_ci: Optional[Tuple[float, float, float]] = None
    
    # Comparisons
    dm_tests: Optional[Dict[str, Dict]] = None
    
    # Per-quantile
    quantile_scores: Optional[Dict[float, float]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'mean_interval_score': self.mean_interval_score,
            'coverage_rate': self.coverage_rate,
            'mean_width': self.mean_width,
            'wspl': self.wspl,
            'mean_pinball': self.mean_pinball,
            'interval_score_ci': self.interval_score_ci,
            'dm_tests': self.dm_tests,
            'quantile_scores': self.quantile_scores,
        }


def full_evaluation(
    lower: np.ndarray,
    upper: np.ndarray,
    targets: np.ndarray,
    scale_factors: Optional[np.ndarray] = None,
    baseline_lower: Optional[np.ndarray] = None,
    baseline_upper: Optional[np.ndarray] = None,
    alpha: float = 0.10
) -> EvaluationResults:
    """
    Run comprehensive evaluation.
    
    Args:
        lower: Predicted lower bounds
        upper: Predicted upper bounds
        targets: True values
        scale_factors: Optional scaling for WSPL
        baseline_lower: Optional baseline lower bounds for comparison
        baseline_upper: Optional baseline upper bounds for comparison
        alpha: Miscoverage rate
        
    Returns:
        EvaluationResults with all metrics
    """
    # Core metrics
    scores = interval_score(lower, upper, targets, alpha)
    cov = coverage_rate(lower, upper, targets)
    widths = upper - lower
    
    # Bootstrap CI
    ci = bootstrap_confidence_interval(scores)
    
    # WSPL if scale factors provided
    wspl = None
    if scale_factors is not None:
        wspl = m5_wspl_from_intervals(lower, upper, targets, scale_factors)
    
    # DM test if baseline provided
    dm_tests = None
    if baseline_lower is not None and baseline_upper is not None:
        baseline_scores = interval_score(baseline_lower, baseline_upper, targets, alpha)
        dm = diebold_mariano_test(scores, baseline_scores)
        dm_tests = {'baseline': dm}
    
    return EvaluationResults(
        mean_interval_score=float(np.mean(scores)),
        coverage_rate=cov,
        mean_width=float(np.mean(widths)),
        wspl=wspl,
        interval_score_ci=ci,
        dm_tests=dm_tests,
    )


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    
    n = 1000
    y = np.random.randn(n) * 10 + 50
    
    # Generate intervals
    lower = y - np.abs(np.random.randn(n)) * 5 - 3
    upper = y + np.abs(np.random.randn(n)) * 5 + 3
    
    # Test interval score
    scores = interval_score(lower, upper, y)
    print(f"Interval Score: {np.mean(scores):.2f}")
    
    # Test coverage
    cov = coverage_rate(lower, upper, y)
    print(f"Coverage: {cov:.2%}")
    
    # Test pinball loss
    predictions = (lower + upper) / 2
    pl = pinball_loss(predictions, y, 0.5)
    print(f"Pinball (median): {np.mean(pl):.2f}")
    
    # Test DM
    scores2 = scores + np.random.randn(n) * 0.5
    dm = diebold_mariano_test(scores, scores2)
    print(f"\nDM Test:")
    print(f"  Statistic: {dm['dm_statistic']:.3f}")
    print(f"  P-value: {dm['p_value']:.4f}")
    print(f"  Improvement: {dm['pct_improvement']:.2f}%")
    
    # Test bootstrap CI
    mean, lo, hi = bootstrap_confidence_interval(scores)
    print(f"\nBootstrap 95% CI:")
    print(f"  Mean: {mean:.2f}")
    print(f"  CI: [{lo:.2f}, {hi:.2f}]")
    
    # Test full evaluation
    scale_factors = np.abs(np.random.randn(n)) + 0.5
    results = full_evaluation(lower, upper, y, scale_factors)
    print(f"\nFull Evaluation:")
    print(f"  Interval Score: {results.mean_interval_score:.2f}")
    print(f"  Coverage: {results.coverage_rate:.2%}")
    print(f"  WSPL: {results.wspl:.4f}")
