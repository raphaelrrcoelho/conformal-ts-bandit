"""
Non-Stationarity Diagnostics for Conformal Thompson Sampling.

This module implements the diagnostic framework for research contribution #2:
"When does adaptive specification selection help?"

The core idea: compute a composite non-stationarity index from the matrix of
per-timestep, per-specification interval scores. A high index indicates that
the optimal specification changes frequently and unpredictably, so adaptive
methods like CTS should significantly outperform fixed baselines. A low index
means fixed baselines are sufficient.

Components of the composite index:
    1. Switch frequency  -- how often the argmin specification changes
    2. Selection entropy  -- how spread out are optimal spec selections
    3. Performance spread -- how different are specs from each other
    4. Changepoint count  -- how many regime shifts in the best-spec sequence

References:
    Page (1954) - CUSUM test for changepoint detection
    Gneiting & Raftery (2007) - Interval scores (the input metric)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class NonStationarityReport:
    """
    Container for non-stationarity diagnostic metrics.

    All component scores are normalized to [0, 1] so they can be averaged
    into a single composite index.

    Attributes:
        switch_frequency: Fraction of timesteps where the argmin spec changes.
            0 = the same spec is always best; 1 = the best spec changes every step.
        selection_entropy: Normalized Shannon entropy of the empirical distribution
            of optimal specifications.  0 = one spec dominates; 1 = uniform.
        performance_spread: Mean pairwise absolute difference in interval scores
            across specifications, normalized by the overall score range.
            0 = all specs are equally good; 1 = maximal spread.
        changepoint_count: Number of detected regime shifts in the best-spec
            sequence, normalized by the theoretical maximum (T - 1).
        composite_index: Weighted average of the four components (default:
            equal weights).
        num_timesteps: Number of evaluation timesteps (T).
        num_specs: Number of specifications (K).
        best_spec_sequence: The argmin spec index at each timestep, shape (T,).
        component_weights: Weights used to form the composite index.
        changepoint_locations: Timestep indices of detected changepoints.
        spec_selection_counts: How many times each spec was optimal, shape (K,).
        mean_scores_per_spec: Mean interval score for each spec, shape (K,).
    """

    # Core component scores (all in [0, 1])
    switch_frequency: float
    selection_entropy: float
    performance_spread: float
    changepoint_count: float

    # Composite
    composite_index: float

    # Dimensions
    num_timesteps: int
    num_specs: int

    # Detailed arrays
    best_spec_sequence: np.ndarray
    component_weights: np.ndarray
    changepoint_locations: np.ndarray
    spec_selection_counts: np.ndarray
    mean_scores_per_spec: np.ndarray

    def to_dict(self) -> Dict[str, object]:
        """Convert scalar fields to a plain dictionary (for serialization)."""
        return {
            'switch_frequency': self.switch_frequency,
            'selection_entropy': self.selection_entropy,
            'performance_spread': self.performance_spread,
            'changepoint_count': self.changepoint_count,
            'composite_index': self.composite_index,
            'num_timesteps': self.num_timesteps,
            'num_specs': self.num_specs,
            'num_changepoints': len(self.changepoint_locations),
            'mean_scores_per_spec': self.mean_scores_per_spec.tolist(),
            'spec_selection_counts': self.spec_selection_counts.tolist(),
        }


# ---------------------------------------------------------------------------
# Component computations
# ---------------------------------------------------------------------------

def _compute_switch_frequency(best_specs: np.ndarray) -> float:
    """
    Fraction of consecutive timesteps where the optimal spec changes.

    Args:
        best_specs: Integer array of shape (T,) with the argmin spec at each step.

    Returns:
        Switch frequency in [0, 1].  Returns 0.0 when T <= 1.
    """
    T = len(best_specs)
    if T <= 1:
        return 0.0
    switches = np.sum(best_specs[1:] != best_specs[:-1])
    return float(switches / (T - 1))


def _compute_selection_entropy(best_specs: np.ndarray, K: int) -> float:
    """
    Normalized Shannon entropy of the empirical distribution of optimal specs.

    Entropy is divided by log(K) so the result lies in [0, 1].
    When K = 1 the entropy is trivially 0.

    Args:
        best_specs: Integer array of shape (T,) with the argmin spec at each step.
        K: Total number of specifications.

    Returns:
        Normalized entropy in [0, 1].
    """
    if K <= 1:
        return 0.0

    counts = np.bincount(best_specs, minlength=K).astype(float)
    probs = counts / counts.sum()

    # Filter zeros to avoid log(0)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(K)

    return float(max(0.0, entropy / max_entropy))


def _compute_performance_spread(scores_matrix: np.ndarray) -> float:
    """
    Normalized mean pairwise difference in interval scores across specs.

    For each timestep, compute the mean absolute difference across all
    pairs of specs, then average over timesteps and normalize by the
    global score range.

    Args:
        scores_matrix: Array of shape (T, K) with interval scores.

    Returns:
        Performance spread in [0, 1].  Returns 0.0 when K <= 1 or when
        the global score range is zero.
    """
    T, K = scores_matrix.shape
    if K <= 1:
        return 0.0

    # Global range for normalization
    global_range = float(np.max(scores_matrix) - np.min(scores_matrix))
    if global_range <= 0:
        return 0.0

    # Mean pairwise absolute difference per timestep.
    # For each row, the mean |s_i - s_j| over all i < j pairs equals
    # (2 / (K*(K-1))) * sum_{i<j} |s_i - s_j|.
    # An efficient computation: for each timestep sort the row, then
    # use the identity sum_{i<j} |s_i - s_j| = sum_k (2k - K + 1) * s_sorted[k].
    sorted_scores = np.sort(scores_matrix, axis=1)  # (T, K)
    coefficients = 2 * np.arange(K) - K + 1  # shape (K,)
    pairwise_sums = sorted_scores @ coefficients  # (T,), sum_{i<j} |s_i - s_j|
    n_pairs = K * (K - 1) / 2
    mean_pairwise = np.mean(pairwise_sums / n_pairs)  # scalar

    return float(np.clip(mean_pairwise / global_range, 0.0, 1.0))


def _detect_changepoints_cusum(
    best_specs: np.ndarray,
    threshold: float = 4.0,
) -> np.ndarray:
    """
    Detect changepoints in the best-spec sequence using a CUSUM-style method.

    The sequence is mapped to a numeric signal (the spec indices), centered,
    and the cumulative sum is tracked. A changepoint is flagged whenever the
    absolute CUSUM exceeds *threshold* standard deviations of the underlying
    differences; the statistic is then reset.

    This is a simple, assumption-light detector suitable for an ordinal
    categorical sequence. It intentionally avoids exotic dependencies.

    Args:
        best_specs: Integer array of shape (T,).
        threshold: Number of standard deviations for the CUSUM alarm.

    Returns:
        Integer array of changepoint timestep indices (0-indexed).
    """
    T = len(best_specs)
    if T <= 2:
        return np.array([], dtype=int)

    signal = best_specs.astype(float)
    diffs = np.diff(signal)

    if np.std(diffs) < 1e-12:
        # Constant sequence -- no changepoints
        return np.array([], dtype=int)

    # Standardize differences
    mu = np.mean(diffs)
    sigma = np.std(diffs)
    z = (diffs - mu) / sigma

    # Two-sided CUSUM
    s_pos = 0.0
    s_neg = 0.0
    changepoints: List[int] = []

    for t in range(len(z)):
        s_pos = max(0.0, s_pos + z[t])
        s_neg = max(0.0, s_neg - z[t])

        if s_pos > threshold or s_neg > threshold:
            changepoints.append(t + 1)  # +1 because diffs are lagged by 1
            s_pos = 0.0
            s_neg = 0.0

    return np.array(changepoints, dtype=int)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_nonstationarity_index(
    scores_matrix: np.ndarray,
    weights: Optional[np.ndarray] = None,
    cusum_threshold: float = 4.0,
) -> NonStationarityReport:
    """
    Compute the composite non-stationarity index from a scores matrix.

    The scores matrix holds interval scores (lower is better) for T timesteps
    and K specifications. The function extracts four component metrics, each
    normalized to [0, 1], and combines them into a single composite index.

    Args:
        scores_matrix: Array of shape (T, K) containing interval scores.
            T = number of evaluation timesteps, K = number of specifications.
            Lower scores are better (consistent with interval score convention).
        weights: Optional array of shape (4,) giving the relative importance
            of [switch_frequency, selection_entropy, performance_spread,
            changepoint_count].  Weights are normalized to sum to 1.
            Default: equal weights (0.25 each).
        cusum_threshold: Sensitivity parameter for CUSUM changepoint detector.
            Lower values detect more changepoints; higher values are more
            conservative.  Default: 4.0.

    Returns:
        NonStationarityReport with all component scores and the composite index.

    Raises:
        ValueError: If scores_matrix has fewer than 2 timesteps or 1 specification.
    """
    scores_matrix = np.asarray(scores_matrix, dtype=float)

    if scores_matrix.ndim != 2:
        raise ValueError(
            f"scores_matrix must be 2-dimensional, got shape {scores_matrix.shape}"
        )

    T, K = scores_matrix.shape

    if T < 2:
        raise ValueError(
            f"Need at least 2 timesteps, got T={T}"
        )
    if K < 1:
        raise ValueError(
            f"Need at least 1 specification, got K={K}"
        )

    # Default equal weights
    if weights is None:
        w = np.array([0.25, 0.25, 0.25, 0.25])
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (4,):
            raise ValueError(f"weights must have shape (4,), got {w.shape}")
        w = w / w.sum()

    # Best spec at each timestep (argmin because lower IS is better)
    best_specs = np.argmin(scores_matrix, axis=1)  # (T,)

    # Components
    sf = _compute_switch_frequency(best_specs)
    se = _compute_selection_entropy(best_specs, K)
    ps = _compute_performance_spread(scores_matrix)

    cp_locations = _detect_changepoints_cusum(best_specs, threshold=cusum_threshold)
    # Normalize: fraction of possible changepoints
    cp_normalized = float(len(cp_locations) / (T - 1)) if T > 1 else 0.0
    cp_normalized = min(cp_normalized, 1.0)

    # Composite
    composite = float(w[0] * sf + w[1] * se + w[2] * ps + w[3] * cp_normalized)

    # Auxiliary arrays
    spec_counts = np.bincount(best_specs, minlength=K)
    mean_per_spec = np.mean(scores_matrix, axis=0)

    return NonStationarityReport(
        switch_frequency=sf,
        selection_entropy=se,
        performance_spread=ps,
        changepoint_count=cp_normalized,
        composite_index=composite,
        num_timesteps=T,
        num_specs=K,
        best_spec_sequence=best_specs,
        component_weights=w,
        changepoint_locations=cp_locations,
        spec_selection_counts=spec_counts,
        mean_scores_per_spec=mean_per_spec,
    )


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def diagnostic_summary(report: NonStationarityReport) -> str:
    """
    Produce a human-readable summary of a NonStationarityReport.

    Args:
        report: A NonStationarityReport object.

    Returns:
        Multi-line formatted string.
    """
    bar_width = 20

    def bar(value: float) -> str:
        filled = int(round(value * bar_width))
        return "[" + "#" * filled + "." * (bar_width - filled) + "]"

    lines = [
        "=" * 64,
        "  Non-Stationarity Diagnostic Report",
        "=" * 64,
        f"  Timesteps (T) : {report.num_timesteps}",
        f"  Specs     (K) : {report.num_specs}",
        "-" * 64,
        "  Component Scores (0 = stationary, 1 = highly non-stationary)",
        "-" * 64,
        f"  Switch frequency  : {report.switch_frequency:.4f}  {bar(report.switch_frequency)}",
        f"  Selection entropy  : {report.selection_entropy:.4f}  {bar(report.selection_entropy)}",
        f"  Performance spread : {report.performance_spread:.4f}  {bar(report.performance_spread)}",
        f"  Changepoint count  : {report.changepoint_count:.4f}  {bar(report.changepoint_count)}",
        "-" * 64,
        f"  COMPOSITE INDEX    : {report.composite_index:.4f}  {bar(report.composite_index)}",
        "-" * 64,
        f"  Weights: sf={report.component_weights[0]:.2f}  "
        f"se={report.component_weights[1]:.2f}  "
        f"ps={report.component_weights[2]:.2f}  "
        f"cp={report.component_weights[3]:.2f}",
        f"  Changepoints detected: {len(report.changepoint_locations)}",
    ]

    # Spec-level summary
    lines.append("-" * 64)
    lines.append("  Per-Specification Summary")
    lines.append(f"  {'Spec':>6s}  {'MeanIS':>10s}  {'OptimalCount':>13s}  {'OptimalPct':>10s}")
    for k in range(report.num_specs):
        pct = 100.0 * report.spec_selection_counts[k] / report.num_timesteps
        lines.append(
            f"  {k:>6d}  {report.mean_scores_per_spec[k]:>10.4f}"
            f"  {report.spec_selection_counts[k]:>13d}  {pct:>9.1f}%"
        )

    # Interpretation hint
    lines.append("-" * 64)
    ci = report.composite_index
    if ci < 0.2:
        hint = "LOW  -- Fixed baselines likely sufficient."
    elif ci < 0.5:
        hint = "MODERATE -- Adaptive methods may offer modest gains."
    elif ci < 0.8:
        hint = "HIGH -- Adaptive specification selection (e.g. CTS) recommended."
    else:
        hint = "VERY HIGH -- Strong regime-switching; CTS should dominate fixed methods."
    lines.append(f"  Interpretation: {hint}")
    lines.append("=" * 64)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cross-dataset comparison
# ---------------------------------------------------------------------------

def compare_datasets(
    reports: Dict[str, NonStationarityReport],
) -> "pd.DataFrame":
    """
    Create a comparison table of non-stationarity diagnostics across datasets.

    Each row corresponds to one dataset; columns are the component scores,
    composite index, and dimensional metadata.

    Args:
        reports: Dictionary mapping dataset name to its NonStationarityReport.

    Returns:
        pandas DataFrame with one row per dataset, sorted by composite_index
        descending.

    Raises:
        ImportError: If pandas is not installed.
    """
    if not _HAS_PANDAS:
        raise ImportError(
            "pandas is required for compare_datasets(). "
            "Install it with: pip install pandas"
        )

    rows = []
    for name, report in reports.items():
        rows.append({
            'dataset': name,
            'T': report.num_timesteps,
            'K': report.num_specs,
            'switch_freq': report.switch_frequency,
            'sel_entropy': report.selection_entropy,
            'perf_spread': report.performance_spread,
            'changepoints': report.changepoint_count,
            'composite': report.composite_index,
            'n_changepoints': len(report.changepoint_locations),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('composite', ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    'NonStationarityReport',
    'compute_nonstationarity_index',
    'diagnostic_summary',
    'compare_datasets',
]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    # --- Test 1: Stationary scenario (one spec is always best) ---
    T, K = 500, 4
    base_scores = np.random.rand(T, K) * 10 + 5
    # Make spec 0 always the best by a large margin
    base_scores[:, 0] = 1.0
    report_stationary = compute_nonstationarity_index(base_scores)
    print(diagnostic_summary(report_stationary))
    print()

    # --- Test 2: Regime-switching scenario ---
    scores_regime = np.random.rand(T, K) * 10 + 5
    # Regime 1: spec 0 best  (t < 150)
    scores_regime[:150, 0] = 0.5
    # Regime 2: spec 2 best  (150 <= t < 350)
    scores_regime[150:350, 2] = 0.5
    # Regime 3: spec 1 best  (t >= 350)
    scores_regime[350:, 1] = 0.5
    report_regime = compute_nonstationarity_index(scores_regime)
    print(diagnostic_summary(report_regime))
    print()

    # --- Test 3: Fully chaotic (uniform random best spec) ---
    scores_chaotic = np.random.rand(T, K) * 10
    report_chaotic = compute_nonstationarity_index(scores_chaotic)
    print(diagnostic_summary(report_chaotic))
    print()

    # --- Test 4: Cross-dataset comparison ---
    reports = {
        'Stationary': report_stationary,
        'Regime-Switching': report_regime,
        'Chaotic': report_chaotic,
    }
    try:
        df = compare_datasets(reports)
        print("Cross-Dataset Comparison:")
        print(df.to_string(index=False))
    except ImportError:
        print("(pandas not available, skipping compare_datasets test)")
