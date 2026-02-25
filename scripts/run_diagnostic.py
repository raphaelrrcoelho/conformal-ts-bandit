#!/usr/bin/env python3
"""
Multi-Dataset Diagnostic Comparison.

Runs CTS experiments across multiple datasets/configurations, computes the
non-stationarity index for each, and correlates non-stationarity with CTS
improvement over baselines.  Outputs a summary table.

This script answers the central diagnostic question of the pivot:
  "When does adaptive specification selection help?"

Usage:
    # Run all available datasets
    python scripts/run_diagnostic.py

    # Specific datasets only
    python scripts/run_diagnostic.py --datasets synthetic_high synthetic_low gefcom_solar

    # Quick run (fewer steps for fast iteration)
    python scripts/run_diagnostic.py --quick

    # Custom output directory
    python scripts/run_diagnostic.py --output ./results/diagnostic
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple

# Ensure the project root is on sys.path so `conformal_ts` is importable
# regardless of where the script is invoked from.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("run_diagnostic")


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

AVAILABLE_DATASETS = [
    "synthetic_high",    # High non-stationarity (frequent regime switches)
    "synthetic_medium",  # Medium non-stationarity
    "synthetic_low",     # Low non-stationarity (very persistent regimes)
    "gefcom_solar",      # GEFCom2014 solar track (synthetic fallback)
    "gefcom_wind",       # GEFCom2014 wind track (synthetic fallback)
    "ETTh1",             # Electricity Transformer Temperature (station 1, hourly)
    "ETTh2",             # Electricity Transformer Temperature (station 2, hourly)
    "ETTm1",             # Electricity Transformer Temperature (station 1, 15-min)
    "ETTm2",             # Electricity Transformer Temperature (station 2, 15-min)
    "ExchangeRate",      # Daily exchange rates of 8 countries
    "Traffic",           # California freeway occupancy rates
]

# Real datasets handled by the RealDatasetLoader
_REAL_DATASET_NAMES = {
    "ETTh1", "ETTh2", "ETTm1", "ETTm2",
    "Electricity", "AustralianElecDemand",
    "ExchangeRate", "Traffic",
}


# ---------------------------------------------------------------------------
# Data class for per-dataset results
# ---------------------------------------------------------------------------

@dataclass
class DatasetResult:
    """Results from running CTS on a single dataset."""

    dataset_name: str
    nonstationarity_index: float = 0.0
    cts_mean_score: float = 0.0
    fixed_mean_score: float = 0.0          # CV-Fixed baseline (primary)
    ensemble_mean_score: float = 0.0
    random_mean_score: float = 0.0
    cts_coverage: float = 0.0
    improvement_over_fixed_pct: float = 0.0
    improvement_over_ensemble_pct: float = 0.0
    n_steps: int = 0
    elapsed_seconds: float = 0.0
    nonstationarity_details: Dict[str, float] = field(default_factory=dict)

    # New baselines
    aci_mean_score: float = 0.0
    best_fixed_mean_score: float = 0.0     # Best single spec in hindsight

    # Multi-seed confidence intervals and significance (None when n_seeds == 1)
    cts_score_ci: Optional[Tuple[float, float]] = None
    fixed_score_ci: Optional[Tuple[float, float]] = None
    improvement_ci: Optional[Tuple[float, float]] = None
    dm_statistic: Optional[float] = None
    dm_pvalue: Optional[float] = None
    n_seeds: int = 1


# ---------------------------------------------------------------------------
# Helpers: run a single experiment and collect scores
# ---------------------------------------------------------------------------

def _run_synthetic_experiment(
    regime_persistence: float,
    n_steps: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Run CTS + baselines on synthetic data with a given regime persistence.

    Uses CQR-based conformal intervals, a proper train/test split, and
    returns scores for all baselines including CV-Fixed and ACI.
    """
    from conformal_ts.experiments.bandit_experiment import (
        BanditExperimentConfig,
        build_scores_matrix_with_cqr,
        generate_regime_switching_series,
        run_bandit_experiment,
    )

    lookback_windows = [10, 25, 50, 100]
    num_specs = len(lookback_windows)
    min_history = 100
    train_steps = 50

    # Extra data for min_history + train warmup + evaluation steps
    total_len = n_steps + min_history + train_steps

    # 1. Generate a raw regime-switching time series
    series = generate_regime_switching_series(
        n_steps=total_len,
        regime_persistence=regime_persistence,
        num_regimes=3,
        seed=seed,
    )

    # 2. Build scores matrix with CQR-based conformal intervals
    scores_matrix, contexts, targets, intervals = build_scores_matrix_with_cqr(
        series,
        lookback_windows=lookback_windows,
        alpha=0.10,
        min_history=min_history,
        calibration_window=50,
    )

    feature_dim = contexts.shape[1]

    # 3. Run the bandit competition with train/test split and coverage
    config = BanditExperimentConfig(
        num_specs=num_specs,
        feature_dim=feature_dim,
        warmup_rounds=5,
        exploration_variance=1.0,
        prior_precision=1.0,
        seed=seed,
        train_steps=train_steps,
        cv_window=50,
        window_size=100,
    )
    result = run_bandit_experiment(
        scores_matrix, contexts, config,
        targets=targets, intervals_matrix=intervals,
    )

    return _build_result_dict(result, num_specs)


def _build_result_dict(result, num_specs: int) -> Dict[str, Any]:
    """Convert BanditExperimentResult to the dict format used downstream."""
    d = {
        "cts_scores": result.cts_scores,
        "fixed_scores": result.cv_fixed_scores,  # CV-Fixed is the primary baseline
        "ensemble_scores": result.ensemble_scores,
        "random_scores": result.random_scores,
        "optimal_specs": result.optimal_specs,
        "scores_matrix": result.scores_matrix,
        "contexts": result.contexts,
        "num_specs": num_specs,
        "oracle_scores": result.oracle_scores,
        "cv_fixed_scores": result.cv_fixed_scores,
        "aci_scores": result.aci_scores,
        "best_fixed_scores": result.best_fixed_scores(),
    }
    if result.cts_covered is not None:
        d["cts_covered"] = result.cts_covered
    return d


def _run_gefcom_experiment(
    track: str,
    n_steps: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Run CTS + baselines on GEFCom2014 (synthetic fallback).

    Pre-builds a (T, K) scores matrix and context matrix from the GEFCom
    data, then delegates to run_bandit_experiment for the competition.

    Each specification is a combination of lookback window, forecast
    horizon, and feature subset.  For each timestep and each spec, we
    compute a rolling-mean-based prediction interval and score it against
    the true target at the spec's forecast horizon.

    Returns the same structure as _run_synthetic_experiment.
    """
    from conformal_ts.data.gefcom2014 import GEFCom2014Loader, GEFComConfig
    from conformal_ts.evaluation.metrics import interval_score
    from conformal_ts.experiments.bandit_experiment import (
        BanditExperimentConfig,
        run_bandit_experiment,
    )

    np.random.seed(seed)

    gefcom_config = GEFComConfig(track=track)
    loader = GEFCom2014Loader(gefcom_config)

    # Use synthetic fallback to avoid requiring real data files
    power_df = loader.create_synthetic_gefcom_data(num_zones=2, num_days=90, seed=seed)
    specs = loader.build_specification_space()
    num_specs = len(specs)

    # Feature dim from one sample
    zone_cols = [c for c in power_df.columns if c.startswith("zone")]
    zone_col = zone_cols[0]
    sample_ts = power_df.index[500]
    sample_features = loader.compute_features(
        power_df[zone_col], sample_ts, 1, "full", 168
    )
    feature_dim = len(sample_features)

    max_horizon = max(s["forecast_horizon"] for s in specs)

    # Start after enough history is available
    start_idx = max(s["lookback_hours"] for s in specs) + 10
    end_idx = min(start_idx + n_steps, len(power_df) - max_horizon - 1)
    actual_steps = end_idx - start_idx

    # --- Pre-build the scores matrix and contexts ---
    scores_matrix = np.zeros((actual_steps, num_specs))
    contexts = np.zeros((actual_steps, feature_dim))

    for step, i in enumerate(range(start_idx, end_idx)):
        ts = power_df.index[i]
        contexts[step] = loader.compute_features(
            power_df[zone_col], ts, 1, "full", 168,
        )

        for sp_idx, sp in enumerate(specs):
            h = sp["forecast_horizon"]
            lookback = sp["lookback_hours"]
            t_idx = min(i + h, len(power_df) - 1)
            target_val = float(power_df.iloc[t_idx][zone_col])

            lookback_vals = power_df[zone_col].iloc[max(0, i - lookback):i].values
            if len(lookback_vals) > 1:
                mu = float(np.mean(lookback_vals))
                sigma = float(np.std(lookback_vals)) + 1e-6
            else:
                mu = target_val
                sigma = 1.0
            lower_k = mu - 1.645 * sigma
            upper_k = mu + 1.645 * sigma
            scores_matrix[step, sp_idx] = interval_score(
                np.array([lower_k]), np.array([upper_k]), np.array([target_val])
            )[0]

    # --- Run the bandit competition with train/test split ---
    train_steps = min(50, actual_steps // 5)
    config = BanditExperimentConfig(
        num_specs=num_specs,
        feature_dim=feature_dim,
        warmup_rounds=5,
        exploration_variance=1.0,
        prior_precision=1.0,
        seed=seed,
        train_steps=train_steps,
        cv_window=50,
        window_size=100,
    )
    result = run_bandit_experiment(scores_matrix, contexts, config)

    return _build_result_dict(result, num_specs)


# ---------------------------------------------------------------------------
# Non-stationarity computation
# ---------------------------------------------------------------------------

def compute_nonstationarity(
    optimal_specs: np.ndarray,
    num_specs: int,
    scores_matrix: Optional[np.ndarray] = None,
    contexts: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute non-stationarity index.

    When contexts are available, uses the predictable NS index (how well
    context predicts the best spec).  Otherwise falls back to the raw NS
    index.
    """
    result_dict: Dict[str, float] = {}

    # Try predictable NS index first (requires contexts)
    if contexts is not None and scores_matrix is not None:
        try:
            from conformal_ts.diagnostics import compute_predictable_ns_index
            pns = compute_predictable_ns_index(scores_matrix, contexts)
            result_dict["composite_index"] = pns.composite_index
            result_dict["predictable_ns"] = pns.predictable_ns_index
            result_dict["context_predictability"] = pns.context_predictability
            result_dict["score_gap"] = pns.score_gap
            result_dict["spec_diversity"] = pns.spec_diversity
            return result_dict
        except (ImportError, Exception) as e:
            logger.debug(f"Predictable NS failed, falling back to raw: {e}")

    # Fallback to raw NS index
    try:
        from conformal_ts.diagnostics import compute_nonstationarity_index, NonStationarityReport

        if scores_matrix is not None:
            sm = scores_matrix
        else:
            n = len(optimal_specs)
            sm = np.ones((n, num_specs))
            for t in range(n):
                sm[t, optimal_specs[t]] = 0.0

        raw = compute_nonstationarity_index(sm)
        if isinstance(raw, NonStationarityReport):
            result_dict["composite_index"] = raw.composite_index
            result_dict["switch_frequency"] = getattr(raw, "switch_frequency", 0.0)
            result_dict["selection_entropy"] = getattr(raw, "selection_entropy", 0.0)
            result_dict["performance_spread"] = getattr(raw, "performance_spread", 0.0)
        return result_dict

    except ImportError:
        return _compute_nonstationarity_fallback(optimal_specs, num_specs)


def _compute_nonstationarity_fallback(
    optimal_specs: np.ndarray, num_specs: int
) -> Dict[str, float]:
    """Compute non-stationarity index without the diagnostics module."""
    n = len(optimal_specs)
    if n < 2:
        return {"composite_index": 0.0, "switch_frequency": 0.0,
                "selection_entropy": 0.0, "performance_spread": 0.0}

    # 1. Switch frequency: fraction of timesteps where optimal spec changes
    switches = np.sum(np.diff(optimal_specs) != 0)
    switch_freq = switches / (n - 1)

    # 2. Selection entropy (normalised)
    counts = np.bincount(optimal_specs.astype(int), minlength=num_specs)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(num_specs) if num_specs > 1 else 1.0
    norm_entropy = entropy / max_entropy

    # 3. Performance spread: std of per-spec selection frequency
    spread = float(np.std(counts / counts.sum()))

    # Composite (simple average, clamped to [0, 1])
    composite = np.clip((switch_freq + norm_entropy + spread) / 3, 0.0, 1.0)

    return {
        "composite_index": float(composite),
        "switch_frequency": float(switch_freq),
        "selection_entropy": float(norm_entropy),
        "performance_spread": float(spread),
    }


# ---------------------------------------------------------------------------
# Per-dataset dispatcher
# ---------------------------------------------------------------------------

def _run_real_dataset_experiment(
    dataset_name: str,
    n_steps: Optional[int],
    seed: int,
) -> Dict[str, Any]:
    """
    Run CTS + baselines on a real dataset with CQR-based conformal intervals.

    Uses diverse forecasters (Naive, SeasonalNaive, RollingMean, SES,
    LinearTrend) instead of 5 rolling-mean lookback windows, giving CTS
    genuinely different failure modes to exploit.
    """
    from conformal_ts.data.real_datasets import RealDatasetLoader
    from conformal_ts.forecasters import make_default_forecasters
    from conformal_ts.experiments.bandit_experiment import (
        BanditExperimentConfig,
        build_scores_matrix_with_cqr,
        run_bandit_experiment,
    )

    loader = RealDatasetLoader()
    raw = loader.prepare_raw_series(dataset_name, num_specs=5)
    if raw is None:
        raise RuntimeError(f"Failed to load dataset: {dataset_name}")

    target = raw["target"]
    warmup = raw["warmup"]
    df = raw["df"]
    config_ds = raw["config"]

    # Build diverse forecasters from dataset's seasonal period.
    # Use horizon = seasonal_period (1 full cycle ahead) so that
    # Naive no longer dominates on smooth high-frequency series.
    seasonal_period = config_ds.seasonal_period or 24
    horizon = seasonal_period  # e.g. 24 for hourly, 96 for 15-min
    forecasters = make_default_forecasters(
        seasonal_period=seasonal_period,
        rolling_window=50,
        trend_window=20,
        horizon=horizon,
    )
    logger.info(
        f"  {dataset_name}: horizon={horizon}, seasonal_period={seasonal_period}"
    )

    # Ensure min_history is sufficient for all forecasters
    min_history = max(warmup, max(fc.min_history for fc in forecasters))

    # Build CQR scores matrix with diverse forecasters
    scores_matrix, _cqr_ctx, targets, intervals = build_scores_matrix_with_cqr(
        target,
        forecasters=forecasters,
        alpha=0.10,
        min_history=min_history,
        calibration_window=50,
    )

    T = scores_matrix.shape[0]

    # Build rich 13-dim context features
    contexts = RealDatasetLoader.build_context_features(
        df, target, min_history, T, config_ds,
    )

    # Truncate to n_steps if requested
    if n_steps is not None and n_steps < T:
        scores_matrix = scores_matrix[:n_steps]
        contexts = contexts[:n_steps]
        targets = targets[:n_steps]
        intervals = intervals[:n_steps]
        T = n_steps

    num_specs = scores_matrix.shape[1]
    feature_dim = contexts.shape[1]
    train_steps = min(50, T // 5)

    config = BanditExperimentConfig(
        num_specs=num_specs,
        feature_dim=feature_dim,
        warmup_rounds=min(5, T // 10),
        exploration_variance=1.0,
        prior_precision=1.0,
        seed=seed,
        train_steps=train_steps,
        cv_window=50,
        window_size=100,
    )
    result = run_bandit_experiment(
        scores_matrix, contexts, config,
        targets=targets, intervals_matrix=intervals,
    )

    return _build_result_dict(result, num_specs)


def _dispatch_experiment(
    dataset_name: str,
    n_steps: int,
    seed: int,
    max_real_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Dispatch to the right experiment runner and return raw results dict."""
    if dataset_name == "synthetic_high":
        return _run_synthetic_experiment(regime_persistence=0.85, n_steps=n_steps, seed=seed)
    elif dataset_name == "synthetic_medium":
        return _run_synthetic_experiment(regime_persistence=0.95, n_steps=n_steps, seed=seed)
    elif dataset_name == "synthetic_low":
        return _run_synthetic_experiment(regime_persistence=0.995, n_steps=n_steps, seed=seed)
    elif dataset_name == "gefcom_solar":
        return _run_gefcom_experiment(track="solar", n_steps=n_steps, seed=seed)
    elif dataset_name == "gefcom_wind":
        return _run_gefcom_experiment(track="wind", n_steps=n_steps, seed=seed)
    elif dataset_name in _REAL_DATASET_NAMES:
        # Real datasets use full length by default; only truncate when
        # explicitly requested via --quick or --max-real-steps.
        real_steps = max_real_steps  # None = full dataset
        return _run_real_dataset_experiment(dataset_name, n_steps=real_steps, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def run_single_dataset(
    dataset_name: str,
    n_steps: int,
    seed: int,
    max_real_steps: Optional[int] = None,
) -> DatasetResult:
    """Run CTS experiment on one dataset (single seed) and compute diagnostics."""
    t0 = time.perf_counter()
    logger.info(f"Running dataset: {dataset_name}")

    raw = _dispatch_experiment(dataset_name, n_steps, seed, max_real_steps=max_real_steps)
    elapsed = time.perf_counter() - t0

    # Compute non-stationarity (with contexts for predictable NS when available)
    ns = compute_nonstationarity(
        raw["optimal_specs"], raw["num_specs"],
        scores_matrix=raw.get("scores_matrix"),
        contexts=raw.get("contexts"),
    )

    # Mean scores
    cts_mean = float(np.mean(raw["cts_scores"])) if len(raw["cts_scores"]) > 0 else 0.0
    fixed_mean = float(np.mean(raw["fixed_scores"])) if len(raw["fixed_scores"]) > 0 else 0.0
    ens_mean = float(np.mean(raw["ensemble_scores"])) if len(raw["ensemble_scores"]) > 0 else 0.0
    rand_mean = float(np.mean(raw["random_scores"])) if len(raw["random_scores"]) > 0 else 0.0
    aci_mean = float(np.mean(raw.get("aci_scores", [0.0])))
    best_fixed_mean = float(np.mean(raw.get("best_fixed_scores", raw["fixed_scores"])))

    # Coverage
    covered = raw.get("cts_covered")
    coverage = float(np.mean(covered)) if covered is not None else 0.0

    # Improvement percentages (lower score is better)
    imp_fixed = 100.0 * (fixed_mean - cts_mean) / (fixed_mean + 1e-12)
    imp_ens = 100.0 * (ens_mean - cts_mean) / (ens_mean + 1e-12)

    result = DatasetResult(
        dataset_name=dataset_name,
        nonstationarity_index=ns["composite_index"],
        cts_mean_score=cts_mean,
        fixed_mean_score=fixed_mean,
        ensemble_mean_score=ens_mean,
        random_mean_score=rand_mean,
        cts_coverage=coverage,
        improvement_over_fixed_pct=imp_fixed,
        improvement_over_ensemble_pct=imp_ens,
        n_steps=len(raw["cts_scores"]),
        elapsed_seconds=elapsed,
        nonstationarity_details=ns,
        aci_mean_score=aci_mean,
        best_fixed_mean_score=best_fixed_mean,
        n_seeds=1,
    )

    logger.info(
        f"  {dataset_name}: NS_index={ns['composite_index']:.3f}  "
        f"CTS={cts_mean:.3f}  CV-Fixed={fixed_mean:.3f}  "
        f"Imp={imp_fixed:+.1f}%  ({elapsed:.1f}s)"
    )

    return result


# ---------------------------------------------------------------------------
# Multi-seed statistical helpers
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute bootstrap percentile confidence interval for the mean.

    Uses the moving-block bootstrap from conformal_ts.evaluation.metrics
    when available, falling back to a simple i.i.d. percentile bootstrap.

    Args:
        values: 1-D array of observations (e.g. per-seed means).
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (default 0.95 for 95% CI).
        seed: Optional RNG seed for reproducibility.

    Returns:
        (lower, upper) bounds of the CI.
    """
    try:
        from conformal_ts.evaluation.metrics import bootstrap_confidence_interval
        _, lo, hi = bootstrap_confidence_interval(
            values,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            block_size=max(1, len(values) // 10),
            seed=seed,
        )
        return (lo, hi)
    except ImportError:
        pass

    # Fallback: simple percentile bootstrap
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1 - confidence
    return (
        float(np.percentile(boot_means, 100 * alpha / 2)),
        float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
    )


def _dm_test(
    cts_scores: np.ndarray,
    fixed_scores: np.ndarray,
) -> Tuple[float, float]:
    """
    Diebold-Mariano test: H0: E[CTS] = E[Fixed] vs H1: E[CTS] < E[Fixed].

    Uses the HAC-based implementation in conformal_ts.evaluation.metrics.
    Returns (dm_statistic, p_value).  A negative statistic with small p-value
    means CTS is significantly better (lower scores).
    """
    try:
        from conformal_ts.evaluation.metrics import diebold_mariano_test
        stat, pval = diebold_mariano_test(
            cts_scores, fixed_scores, alternative="less", h=1,
        )
        return (stat, pval)
    except (ImportError, TypeError):
        pass

    # Fallback using the competition_metrics version (returns dict)
    try:
        from conformal_ts.evaluation.competition_metrics import (
            diebold_mariano_test as dm_dict,
        )
        result = dm_dict(cts_scores, fixed_scores, h=1)
        stat = result["dm_statistic"]
        # Convert two-sided p-value to one-sided (less)
        from scipy import stats as _sp
        pval = _sp.norm.cdf(stat)
        return (stat, pval)
    except ImportError:
        pass

    # Minimal inline implementation
    from scipy import stats as _sp
    d = cts_scores - fixed_scores
    n = len(d)
    d_bar = np.mean(d)
    var_d = np.var(d, ddof=1)
    se = np.sqrt(var_d / n) if var_d > 0 else 1e-10
    stat = d_bar / se
    pval = _sp.norm.cdf(stat)
    return (float(stat), float(pval))


def _significance_stars(pvalue: float) -> str:
    """Return significance stars for a p-value."""
    if pvalue < 0.001:
        return "***"
    elif pvalue < 0.01:
        return "**"
    elif pvalue < 0.05:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def run_dataset_multi_seed(
    dataset_name: str,
    n_steps: int,
    base_seed: int,
    n_seeds: int,
    max_real_steps: Optional[int] = None,
) -> DatasetResult:
    """
    Run dataset across multiple seeds and aggregate with CIs + significance.

    For each seed in [base_seed, base_seed + n_seeds - 1]:
      1. Run the experiment and collect per-step score arrays.
      2. Record the per-seed mean scores.

    Then:
      - Report the grand mean across seeds as the point estimate.
      - Compute 95% bootstrap CIs on per-seed means.
      - Run a Diebold-Mariano test (CTS vs Fixed) on the *concatenated*
        per-step score series across all seeds for maximum power.
      - Non-stationarity index is averaged across seeds.
    """
    t0 = time.perf_counter()
    logger.info(f"Running dataset: {dataset_name}  ({n_seeds} seeds)")

    # -- Collect per-seed results --
    per_seed_cts_means: List[float] = []
    per_seed_fixed_means: List[float] = []
    per_seed_ens_means: List[float] = []
    per_seed_rand_means: List[float] = []
    per_seed_aci_means: List[float] = []
    per_seed_best_fixed_means: List[float] = []
    per_seed_coverage: List[float] = []
    per_seed_ns: List[float] = []
    per_seed_imp: List[float] = []
    ns_details_accum: Dict[str, List[float]] = {}

    # Concatenated per-step scores (for DM test)
    all_cts_scores: List[np.ndarray] = []
    all_fixed_scores: List[np.ndarray] = []

    for s in range(n_seeds):
        seed = base_seed + s
        logger.info(f"  Seed {seed} ({s + 1}/{n_seeds})")

        raw = _dispatch_experiment(dataset_name, n_steps, seed, max_real_steps=max_real_steps)

        # Per-seed means
        cts_m = float(np.mean(raw["cts_scores"])) if len(raw["cts_scores"]) > 0 else 0.0
        fixed_m = float(np.mean(raw["fixed_scores"])) if len(raw["fixed_scores"]) > 0 else 0.0
        ens_m = float(np.mean(raw["ensemble_scores"])) if len(raw["ensemble_scores"]) > 0 else 0.0
        rand_m = float(np.mean(raw["random_scores"])) if len(raw["random_scores"]) > 0 else 0.0
        aci_m = float(np.mean(raw.get("aci_scores", [0.0])))
        best_fixed_m = float(np.mean(raw.get("best_fixed_scores", raw["fixed_scores"])))

        per_seed_cts_means.append(cts_m)
        per_seed_fixed_means.append(fixed_m)
        per_seed_ens_means.append(ens_m)
        per_seed_rand_means.append(rand_m)
        per_seed_aci_means.append(aci_m)
        per_seed_best_fixed_means.append(best_fixed_m)

        # Coverage
        covered = raw.get("cts_covered")
        per_seed_coverage.append(float(np.mean(covered)) if covered is not None else 0.0)

        imp = 100.0 * (fixed_m - cts_m) / (fixed_m + 1e-12)
        per_seed_imp.append(imp)

        # Concatenated step-level scores for DM test
        all_cts_scores.append(raw["cts_scores"])
        all_fixed_scores.append(raw["fixed_scores"])

        # Non-stationarity (with contexts for predictable NS)
        ns = compute_nonstationarity(
            raw["optimal_specs"], raw["num_specs"],
            scores_matrix=raw.get("scores_matrix"),
            contexts=raw.get("contexts"),
        )
        per_seed_ns.append(ns["composite_index"])
        for k, v in ns.items():
            ns_details_accum.setdefault(k, []).append(v)

    elapsed = time.perf_counter() - t0

    # -- Grand means --
    cts_mean = float(np.mean(per_seed_cts_means))
    fixed_mean = float(np.mean(per_seed_fixed_means))
    ens_mean = float(np.mean(per_seed_ens_means))
    rand_mean = float(np.mean(per_seed_rand_means))
    aci_mean = float(np.mean(per_seed_aci_means))
    best_fixed_mean = float(np.mean(per_seed_best_fixed_means))
    coverage_mean = float(np.mean(per_seed_coverage))
    imp_fixed = float(np.mean(per_seed_imp))
    imp_ens = 100.0 * (ens_mean - cts_mean) / (ens_mean + 1e-12)
    ns_index = float(np.mean(per_seed_ns))

    # -- Bootstrap 95% CIs on per-seed means --
    arr_cts = np.array(per_seed_cts_means)
    arr_fixed = np.array(per_seed_fixed_means)
    arr_imp = np.array(per_seed_imp)

    cts_ci = _bootstrap_ci(arr_cts, seed=base_seed)
    fixed_ci = _bootstrap_ci(arr_fixed, seed=base_seed + 1)
    imp_ci = _bootstrap_ci(arr_imp, seed=base_seed + 2)

    # -- Diebold-Mariano test on concatenated step-level scores --
    cat_cts = np.concatenate(all_cts_scores)
    cat_fixed = np.concatenate(all_fixed_scores)
    dm_stat, dm_pval = _dm_test(cat_cts, cat_fixed)

    # -- Average non-stationarity details --
    ns_details_mean = {k: float(np.mean(v)) for k, v in ns_details_accum.items()}

    result = DatasetResult(
        dataset_name=dataset_name,
        nonstationarity_index=ns_index,
        cts_mean_score=cts_mean,
        fixed_mean_score=fixed_mean,
        ensemble_mean_score=ens_mean,
        random_mean_score=rand_mean,
        cts_coverage=coverage_mean,
        improvement_over_fixed_pct=imp_fixed,
        improvement_over_ensemble_pct=imp_ens,
        n_steps=len(all_cts_scores[0]) if all_cts_scores else 0,
        elapsed_seconds=elapsed,
        nonstationarity_details=ns_details_mean,
        aci_mean_score=aci_mean,
        best_fixed_mean_score=best_fixed_mean,
        cts_score_ci=cts_ci,
        fixed_score_ci=fixed_ci,
        improvement_ci=imp_ci,
        dm_statistic=dm_stat,
        dm_pvalue=dm_pval,
        n_seeds=n_seeds,
    )

    stars = _significance_stars(dm_pval)
    logger.info(
        f"  {dataset_name}: NS_index={ns_index:.3f}  "
        f"CTS={cts_mean:.3f} ({cts_ci[0]:.3f}, {cts_ci[1]:.3f})  "
        f"CV-Fixed={fixed_mean:.3f} ({fixed_ci[0]:.3f}, {fixed_ci[1]:.3f})  "
        f"Imp={imp_fixed:+.1f}%{stars}  DM p={dm_pval:.4f}  "
        f"Cov={coverage_mean:.1%}  ({elapsed:.1f}s, {n_seeds} seeds)"
    )

    return result


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def analyze_correlation(results: List[DatasetResult]) -> Dict[str, Any]:
    """Correlate non-stationarity index with CTS improvement."""
    from scipy import stats as sp_stats

    ns_indices = np.array([r.nonstationarity_index for r in results])
    improvements = np.array([r.improvement_over_fixed_pct for r in results])

    if len(ns_indices) < 3:
        return {
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
            "note": "Too few datasets for reliable correlation",
        }

    pr, pp = sp_stats.pearsonr(ns_indices, improvements)
    sr, sp = sp_stats.spearmanr(ns_indices, improvements)

    return {
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt_score_ci(mean: float, ci: Optional[Tuple[float, float]]) -> str:
    """Format a score with optional CI, e.g. '0.413 (0.39, 0.44)'."""
    if ci is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} ({ci[0]:.2f}, {ci[1]:.2f})"


def print_summary_table(results: List[DatasetResult], correlation: Dict[str, Any]):
    """Print a formatted summary table to stdout."""
    has_ci = any(r.cts_score_ci is not None for r in results)
    has_coverage = any(r.cts_coverage > 0 for r in results)

    if has_ci:
        header = (
            f"{'Dataset':<18s} {'NS':>6s} "
            f"{'CTS (95% CI)':>22s} "
            f"{'CV-Fixed (95% CI)':>22s} "
            f"{'ACI':>9s} "
            f"{'Imp%':>10s} {'DM p':>8s}"
        )
        if has_coverage:
            header += f" {'Cov':>6s}"
    else:
        header = (
            f"{'Dataset':<18s} {'NS':>6s} {'CTS':>9s} {'CV-Fixed':>9s} "
            f"{'ACI':>9s} {'Ensemble':>9s} {'Imp%':>10s} {'Steps':>6s} {'Time':>6s}"
        )
        if has_coverage:
            header += f" {'Cov':>6s}"

    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("  Diagnostic Summary: Non-Stationarity vs CTS Improvement")
    if has_ci:
        n_seeds = results[0].n_seeds if results else 0
        print(f"  ({n_seeds} seeds per dataset, 95% bootstrap CIs)")
    print(f"  Baseline: CV-Fixed (rolling cross-validated best spec)")
    print("=" * len(header))
    print(header)
    print(sep)

    for r in sorted(results, key=lambda x: -x.nonstationarity_index):
        if has_ci:
            cts_str = _fmt_score_ci(r.cts_mean_score, r.cts_score_ci)
            fixed_str = _fmt_score_ci(r.fixed_mean_score, r.fixed_score_ci)

            # Improvement with significance stars
            stars = _significance_stars(r.dm_pvalue) if r.dm_pvalue is not None else ""
            imp_str = f"{r.improvement_over_fixed_pct:+.1f}%{stars}"

            dm_p_str = f"{r.dm_pvalue:.3f}" if r.dm_pvalue is not None else "N/A"

            line = (
                f"{r.dataset_name:<18s} "
                f"{r.nonstationarity_index:>6.3f} "
                f"{cts_str:>22s} "
                f"{fixed_str:>22s} "
                f"{r.aci_mean_score:>9.3f} "
                f"{imp_str:>10s} "
                f"{dm_p_str:>8s}"
            )
            if has_coverage:
                line += f" {r.cts_coverage:>5.1%}"
            print(line)
        else:
            line = (
                f"{r.dataset_name:<18s} "
                f"{r.nonstationarity_index:>6.3f} "
                f"{r.cts_mean_score:>9.3f} "
                f"{r.fixed_mean_score:>9.3f} "
                f"{r.aci_mean_score:>9.3f} "
                f"{r.ensemble_mean_score:>9.3f} "
                f"{r.improvement_over_fixed_pct:>+10.1f}% "
                f"{r.n_steps:>6d} "
                f"{r.elapsed_seconds:>5.1f}s"
            )
            if has_coverage:
                line += f" {r.cts_coverage:>5.1%}"
            print(line)

    print(sep)

    # Print significance legend when CIs are present
    if has_ci:
        print("  Significance: * p<0.05  ** p<0.01  *** p<0.001 (one-sided DM test)")
        print()

    print()
    print("Correlation (NS Index vs CTS Improvement over CV-Fixed):")
    print(f"  Pearson  r = {correlation.get('pearson_r', float('nan')):.3f}  "
          f"(p = {correlation.get('pearson_p', float('nan')):.4f})")
    print(f"  Spearman r = {correlation.get('spearman_r', float('nan')):.3f}  "
          f"(p = {correlation.get('spearman_p', float('nan')):.4f})")
    if "note" in correlation:
        print(f"  Note: {correlation['note']}")
    print()


def save_results(
    results: List[DatasetResult],
    correlation: Dict[str, Any],
    output_dir: Path,
):
    """Save results as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "diagnostic_summary.json"

    payload = {
        "datasets": [asdict(r) for r in results],
        "correlation": correlation,
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-dataset diagnostic comparison for CTS."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=AVAILABLE_DATASETS,
        help=f"Datasets to include (default: all). Choices: {AVAILABLE_DATASETS}",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/diagnostic",
        help="Output directory (default: ./results/diagnostic)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run with fewer steps (100 instead of 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="Number of seeds per dataset (default: 10). "
             "Seeds will be [seed, seed+1, ..., seed+n_seeds-1].",
    )
    parser.add_argument(
        "--max-real-steps",
        type=int,
        default=None,
        help="Max steps for real datasets (default: None = full dataset). "
             "Overridden to n_steps when --quick is used.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    np.random.seed(args.seed)

    datasets = args.datasets or AVAILABLE_DATASETS
    n_steps = 100 if args.quick else 500
    n_seeds = args.n_seeds
    output_dir = Path(args.output)

    # Real dataset step limit: --quick forces truncation, otherwise use
    # --max-real-steps (default None = full dataset).
    max_real_steps = n_steps if args.quick else args.max_real_steps

    logger.info("=" * 60)
    logger.info("  Diagnostic Comparison: Non-Stationarity vs CTS Benefit")
    logger.info("=" * 60)
    logger.info(f"  Datasets : {datasets}")
    logger.info(f"  Steps    : {n_steps}")
    logger.info(f"  Seed     : {args.seed}")
    logger.info(f"  N seeds  : {n_seeds}")
    logger.info(f"  Output   : {output_dir}")
    logger.info(f"  Max real : {max_real_steps or 'full'}")
    logger.info("=" * 60)

    results: List[DatasetResult] = []

    for ds_name in datasets:
        try:
            if n_seeds > 1:
                result = run_dataset_multi_seed(
                    ds_name,
                    n_steps=n_steps,
                    base_seed=args.seed,
                    n_seeds=n_seeds,
                    max_real_steps=max_real_steps,
                )
            else:
                result = run_single_dataset(
                    ds_name, n_steps=n_steps, seed=args.seed,
                    max_real_steps=max_real_steps,
                )
            results.append(result)
        except Exception:
            logger.exception(f"Failed on dataset {ds_name}, skipping.")

    if not results:
        logger.error("No datasets completed successfully.")
        sys.exit(1)

    # Correlation analysis (uses mean improvements, which are already
    # averaged across seeds when n_seeds > 1)
    correlation = analyze_correlation(results)

    # Output
    print_summary_table(results, correlation)
    save_results(results, correlation, output_dir)

    logger.info("Diagnostic comparison complete.")


if __name__ == "__main__":
    main()
