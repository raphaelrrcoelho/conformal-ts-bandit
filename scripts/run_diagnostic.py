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
from typing import Dict, List, Any, Optional

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
    "gefcom_solar",      # GEFCom2014 solar track
    "gefcom_wind",       # GEFCom2014 wind track
]


# ---------------------------------------------------------------------------
# Data class for per-dataset results
# ---------------------------------------------------------------------------

@dataclass
class DatasetResult:
    """Results from running CTS on a single dataset."""

    dataset_name: str
    nonstationarity_index: float = 0.0
    cts_mean_score: float = 0.0
    fixed_mean_score: float = 0.0
    ensemble_mean_score: float = 0.0
    random_mean_score: float = 0.0
    cts_coverage: float = 0.0
    improvement_over_fixed_pct: float = 0.0
    improvement_over_ensemble_pct: float = 0.0
    n_steps: int = 0
    elapsed_seconds: float = 0.0
    nonstationarity_details: Dict[str, float] = field(default_factory=dict)


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

    At each step we compute the interval score for EVERY spec (using the
    generator's true quantiles).  Each method selects a spec via its own
    strategy and receives that spec's score.  CTS learns online via a
    Linear Thompson Sampling bandit; fixed always picks spec 0; random
    picks uniformly; ensemble averages across specs.

    Returns dict with per-step scores for each method, the per-step
    optimal specification array, and the full (T, K) scores matrix for
    non-stationarity diagnostics.
    """
    from conformal_ts.data.synthetic import SyntheticDataGenerator
    from conformal_ts.models.linear_ts import LinearThompsonSampling
    from conformal_ts.evaluation.metrics import interval_score

    num_specs = 4
    feature_dim = 8

    generator = SyntheticDataGenerator(
        num_series=50,
        num_specifications=num_specs,
        feature_dim=feature_dim,
        num_regimes=3,
        regime_persistence=regime_persistence,
        observation_noise=0.1,
        spec_advantage=0.3,
        seed=seed,
    )

    # Bandit for CTS action selection
    bandit = LinearThompsonSampling(
        num_actions=num_specs,
        feature_dim=feature_dim,
        prior_precision=0.1,
        exploration_variance=5.0,
        seed=seed,
    )
    warmup = min(20, n_steps // 5)

    rng = np.random.default_rng(seed + 999)

    scores_matrix = np.zeros((n_steps, num_specs))
    cts_scores = []
    fixed_scores = []
    ensemble_scores = []
    random_scores = []
    optimal_specs = []

    spec_advantage = generator.spec_advantage  # 0.3

    for t in range(n_steps):
        ctx = generator._generate_context(0)
        regime = generator.regimes[generator.current_regime]

        # --- Common true target (spec-independent observation) ---
        history = generator._history[0]
        ar_contrib = regime.ar_coefficient * history[-1] if history else 0.0
        noise = generator.rng.normal(0, regime.volatility)
        true_target = regime.mean + ar_contrib + noise

        # --- Compute interval score for EVERY spec ---
        # Optimal spec: well-centered tight interval
        # Suboptimal spec: biased and wider interval
        for k in range(num_specs):
            if k == regime.optimal_spec:
                pred_center = true_target + generator.rng.normal(0, 0.01)
                pred_width = regime.volatility
            else:
                distance = abs(k - regime.optimal_spec)
                # Suboptimal specs have biased, wider predictions
                bias = distance * spec_advantage * regime.volatility
                pred_center = true_target + bias * generator.rng.choice([-1, 1])
                pred_width = regime.volatility * (1.0 + 0.3 * distance)

            lower_k = pred_center - 1.645 * pred_width
            upper_k = pred_center + 1.645 * pred_width
            scores_matrix[t, k] = interval_score(
                np.array([lower_k]), np.array([upper_k]), np.array([true_target])
            )[0]

        best_spec = int(np.argmin(scores_matrix[t]))
        optimal_specs.append(best_spec)

        # --- CTS: Thompson Sampling selection ---
        if t < warmup:
            cts_action = t % num_specs
        else:
            cts_action = bandit.select_action(ctx)
        # Reward = negative interval score (higher is better)
        bandit.update(cts_action, ctx, -scores_matrix[t, cts_action])
        cts_scores.append(scores_matrix[t, cts_action])

        # --- Fixed: always spec 0 ---
        fixed_scores.append(scores_matrix[t, 0])

        # --- Random: uniform random ---
        random_scores.append(scores_matrix[t, rng.integers(0, num_specs)])

        # --- Ensemble: average score across specs ---
        ensemble_scores.append(float(np.mean(scores_matrix[t])))

        # Advance generator
        generator._history[0].append(true_target)
        if len(generator._history[0]) > 100:
            generator._history[0] = generator._history[0][-100:]
        generator._transition_regime()
        generator.t += 1

    return {
        "cts_scores": np.array(cts_scores),
        "fixed_scores": np.array(fixed_scores),
        "ensemble_scores": np.array(ensemble_scores),
        "random_scores": np.array(random_scores),
        "optimal_specs": np.array(optimal_specs),
        "scores_matrix": scores_matrix,
        "num_specs": num_specs,
    }


def _run_gefcom_experiment(
    track: str,
    n_steps: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Run CTS + baselines on GEFCom2014 (synthetic fallback).

    For each timestep, computes interval scores for ALL specs (each spec
    has its own forecast horizon, so targets differ).  Each method selects
    a spec and receives that spec's score.

    Returns the same structure as _run_synthetic_experiment.
    """
    from conformal_ts.data.gefcom2014 import GEFCom2014Loader, GEFComConfig
    from conformal_ts.models.linear_ts import LinearThompsonSampling
    from conformal_ts.evaluation.metrics import interval_score

    np.random.seed(seed)

    config = GEFComConfig(track=track)
    loader = GEFCom2014Loader(config)

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

    # Bandit for CTS
    bandit = LinearThompsonSampling(
        num_actions=num_specs,
        feature_dim=feature_dim,
        prior_precision=0.1,
        exploration_variance=5.0,
        seed=seed,
    )
    warmup = min(20, n_steps // 5)

    rng = np.random.default_rng(seed + 777)
    max_horizon = max(s["forecast_horizon"] for s in specs)

    # Start after enough history is available
    start_idx = max(s["lookback_hours"] for s in specs) + 10
    end_idx = min(start_idx + n_steps, len(power_df) - max_horizon - 1)
    actual_steps = end_idx - start_idx

    scores_matrix = np.zeros((actual_steps, num_specs))
    cts_scores = []
    fixed_scores = []
    ensemble_scores = []
    random_scores = []
    optimal_specs_list = []

    for step, i in enumerate(range(start_idx, end_idx)):
        ts = power_df.index[i]
        ctx = loader.compute_features(power_df[zone_col], ts, 1, "full", 168)

        # --- Compute interval score for EVERY spec ---
        # Each spec has its own forecast horizon → different target
        for sp_idx, sp in enumerate(specs):
            h = sp["forecast_horizon"]
            lookback = sp["lookback_hours"]
            t_idx = min(i + h, len(power_df) - 1)
            target_val = float(power_df.iloc[t_idx][zone_col])

            # Use a simple prediction interval based on recent volatility
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

        best_spec = int(np.argmin(scores_matrix[step]))
        optimal_specs_list.append(best_spec)

        # --- CTS: Thompson Sampling ---
        if step < warmup:
            cts_action = step % num_specs
        else:
            cts_action = bandit.select_action(ctx)
        bandit.update(cts_action, ctx, -scores_matrix[step, cts_action])
        cts_scores.append(scores_matrix[step, cts_action])

        # --- Fixed: always spec 0 ---
        fixed_scores.append(scores_matrix[step, 0])

        # --- Random ---
        random_scores.append(scores_matrix[step, rng.integers(0, num_specs)])

        # --- Ensemble: average ---
        ensemble_scores.append(float(np.mean(scores_matrix[step])))

    return {
        "cts_scores": np.array(cts_scores),
        "fixed_scores": np.array(fixed_scores),
        "ensemble_scores": np.array(ensemble_scores),
        "random_scores": np.array(random_scores),
        "optimal_specs": np.array(optimal_specs_list),
        "scores_matrix": scores_matrix,
        "num_specs": num_specs,
    }


# ---------------------------------------------------------------------------
# Non-stationarity computation
# ---------------------------------------------------------------------------

def compute_nonstationarity(
    optimal_specs: np.ndarray,
    num_specs: int,
    scores_matrix: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute composite non-stationarity index.

    Uses the full (T, K) scores_matrix when available (much richer signal
    than the optimal-spec sequence alone).  Falls back to a self-contained
    implementation from the optimal-spec sequence if the diagnostics
    module is not importable.
    """
    try:
        from conformal_ts.diagnostics import compute_nonstationarity_index, NonStationarityReport

        if scores_matrix is not None:
            sm = scores_matrix
        else:
            # Build a proxy (T, K) matrix from optimal_specs
            n = len(optimal_specs)
            sm = np.ones((n, num_specs))
            for t in range(n):
                sm[t, optimal_specs[t]] = 0.0  # lower is better

        result = compute_nonstationarity_index(sm)
        if isinstance(result, NonStationarityReport):
            return {
                "composite_index": result.composite_index,
                "switch_frequency": getattr(result, "switch_frequency", 0.0),
                "selection_entropy": getattr(result, "selection_entropy", 0.0),
                "performance_spread": getattr(result, "performance_spread", 0.0),
            }
        elif isinstance(result, dict):
            return result
        else:
            return {"composite_index": float(result)}

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

def run_single_dataset(
    dataset_name: str,
    n_steps: int,
    seed: int,
) -> DatasetResult:
    """Run CTS experiment on one dataset and compute diagnostics."""
    t0 = time.perf_counter()
    logger.info(f"Running dataset: {dataset_name}")

    # Dispatch
    if dataset_name == "synthetic_high":
        raw = _run_synthetic_experiment(regime_persistence=0.85, n_steps=n_steps, seed=seed)
    elif dataset_name == "synthetic_medium":
        raw = _run_synthetic_experiment(regime_persistence=0.95, n_steps=n_steps, seed=seed)
    elif dataset_name == "synthetic_low":
        raw = _run_synthetic_experiment(regime_persistence=0.995, n_steps=n_steps, seed=seed)
    elif dataset_name == "gefcom_solar":
        raw = _run_gefcom_experiment(track="solar", n_steps=n_steps, seed=seed)
    elif dataset_name == "gefcom_wind":
        raw = _run_gefcom_experiment(track="wind", n_steps=n_steps, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    elapsed = time.perf_counter() - t0

    # Compute non-stationarity
    ns = compute_nonstationarity(
        raw["optimal_specs"], raw["num_specs"],
        scores_matrix=raw.get("scores_matrix"),
    )

    # Mean scores
    cts_mean = float(np.mean(raw["cts_scores"])) if len(raw["cts_scores"]) > 0 else 0.0
    fixed_mean = float(np.mean(raw["fixed_scores"])) if len(raw["fixed_scores"]) > 0 else 0.0
    ens_mean = float(np.mean(raw["ensemble_scores"])) if len(raw["ensemble_scores"]) > 0 else 0.0
    rand_mean = float(np.mean(raw["random_scores"])) if len(raw["random_scores"]) > 0 else 0.0

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
        cts_coverage=0.0,  # would need separate tracking
        improvement_over_fixed_pct=imp_fixed,
        improvement_over_ensemble_pct=imp_ens,
        n_steps=len(raw["cts_scores"]),
        elapsed_seconds=elapsed,
        nonstationarity_details=ns,
    )

    logger.info(
        f"  {dataset_name}: NS_index={ns['composite_index']:.3f}  "
        f"CTS={cts_mean:.3f}  Fixed={fixed_mean:.3f}  "
        f"Imp={imp_fixed:+.1f}%  ({elapsed:.1f}s)"
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

def print_summary_table(results: List[DatasetResult], correlation: Dict[str, Any]):
    """Print a formatted summary table to stdout."""
    header = (
        f"{'Dataset':<20s} {'NS Index':>8s} {'CTS':>9s} {'Fixed':>9s} "
        f"{'Ensemble':>9s} {'Imp(Fixed)':>11s} {'Steps':>6s} {'Time':>6s}"
    )
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("  Diagnostic Summary: Non-Stationarity vs CTS Improvement")
    print("=" * len(header))
    print(header)
    print(sep)

    for r in sorted(results, key=lambda x: -x.nonstationarity_index):
        print(
            f"{r.dataset_name:<20s} "
            f"{r.nonstationarity_index:>8.3f} "
            f"{r.cts_mean_score:>9.3f} "
            f"{r.fixed_mean_score:>9.3f} "
            f"{r.ensemble_mean_score:>9.3f} "
            f"{r.improvement_over_fixed_pct:>+10.1f}% "
            f"{r.n_steps:>6d} "
            f"{r.elapsed_seconds:>5.1f}s"
        )

    print(sep)
    print()
    print("Correlation (NS Index vs CTS Improvement over Fixed):")
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
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    np.random.seed(args.seed)

    datasets = args.datasets or AVAILABLE_DATASETS
    n_steps = 100 if args.quick else 500
    output_dir = Path(args.output)

    logger.info("=" * 60)
    logger.info("  Diagnostic Comparison: Non-Stationarity vs CTS Benefit")
    logger.info("=" * 60)
    logger.info(f"  Datasets : {datasets}")
    logger.info(f"  Steps    : {n_steps}")
    logger.info(f"  Seed     : {args.seed}")
    logger.info(f"  Output   : {output_dir}")
    logger.info("=" * 60)

    results: List[DatasetResult] = []

    for ds_name in datasets:
        try:
            result = run_single_dataset(ds_name, n_steps=n_steps, seed=args.seed)
            results.append(result)
        except Exception:
            logger.exception(f"Failed on dataset {ds_name}, skipping.")

    if not results:
        logger.error("No datasets completed successfully.")
        sys.exit(1)

    # Correlation analysis
    correlation = analyze_correlation(results)

    # Output
    print_summary_table(results, correlation)
    save_results(results, correlation, output_dir)

    logger.info("Diagnostic comparison complete.")


if __name__ == "__main__":
    main()
