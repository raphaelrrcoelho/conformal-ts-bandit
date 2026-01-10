"""
Main Experiment Runner for Conformal Thompson Sampling.

This module provides:
- Complete experiment pipeline
- Baseline implementations
- Evaluation and comparison
- Logging and checkpointing
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from datetime import datetime
import logging

from ..config import ExperimentConfig, get_default_config
from ..models.cts_agent import ConformalThompsonSampling, CTSConfig
from ..models.cqr import ConformizedQuantileRegression
from ..evaluation.metrics import (
    interval_score, coverage_rate, mean_interval_width,
    compute_interval_metrics, diebold_mariano_test,
    bootstrap_confidence_interval, compare_methods
)
from ..data.synthetic import SyntheticDataGenerator, create_synthetic_experiment
from ..data.m5_loader import create_m5_simulation_data, M5SpecificationSelector


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentResults:
    """Container for experiment results."""
    
    experiment_name: str
    timestamp: str
    
    # Main metrics
    mean_interval_score: float
    coverage_rate: float
    mean_width: float
    
    # By method (for comparisons)
    method_scores: Dict[str, np.ndarray] = field(default_factory=dict)
    method_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Statistical tests
    dm_tests: Dict[str, Dict[str, float]] = field(default_factory=dict)
    bootstrap_cis: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    
    # Learning curves
    cumulative_regret: Optional[np.ndarray] = None
    rolling_coverage: Optional[np.ndarray] = None
    rolling_score: Optional[np.ndarray] = None
    
    # Action statistics
    action_counts: Optional[np.ndarray] = None
    action_rewards: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'mean_interval_score': self.mean_interval_score,
            'coverage_rate': self.coverage_rate,
            'mean_width': self.mean_width,
            'method_metrics': self.method_metrics,
            'dm_tests': self.dm_tests,
        }
    
    def save(self, path: str):
        """Save results to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class FixedSpecificationBaseline:
    """
    Baseline that always uses a fixed specification.
    """
    
    def __init__(
        self,
        specification: int,
        feature_dim: int,
        coverage_target: float = 0.90
    ):
        """
        Initialize fixed baseline.
        
        Args:
            specification: Fixed specification to use
            feature_dim: Feature dimension
            coverage_target: Target coverage
        """
        self.specification = specification
        self.cqr = ConformizedQuantileRegression(
            feature_dim=feature_dim,
            coverage_target=coverage_target
        )
    
    def select_action(self, context: np.ndarray) -> int:
        """Always return fixed specification."""
        return self.specification
    
    def predict_interval(self, context: np.ndarray) -> Tuple[float, float]:
        """Get prediction interval."""
        return self.cqr.predict_interval(context)
    
    def update(self, context: np.ndarray, outcome: float):
        """Update CQR model."""
        self.cqr.update(context, outcome)


class EqualWeightEnsemble:
    """
    Baseline that averages predictions from all specifications.
    """
    
    def __init__(
        self,
        num_specifications: int,
        feature_dim: int,
        coverage_target: float = 0.90
    ):
        """
        Initialize ensemble baseline.
        
        Args:
            num_specifications: Number of specifications
            feature_dim: Feature dimension
            coverage_target: Target coverage
        """
        self.num_specifications = num_specifications
        self.models = [
            ConformizedQuantileRegression(
                feature_dim=feature_dim,
                coverage_target=coverage_target
            )
            for _ in range(num_specifications)
        ]
    
    def predict_interval(self, context: np.ndarray) -> Tuple[float, float]:
        """Average intervals from all models."""
        lowers = []
        uppers = []
        
        for model in self.models:
            lower, upper = model.predict_interval(context)
            lowers.append(lower)
            uppers.append(upper)
        
        return np.mean(lowers), np.mean(uppers)
    
    def update(
        self,
        specification: int,
        context: np.ndarray,
        outcome: float
    ):
        """Update the model for the given specification."""
        self.models[specification].update(context, outcome)


class OracleBaseline:
    """
    Oracle baseline that always knows the optimal specification.
    
    Used for computing regret upper bound.
    """
    
    def __init__(
        self,
        num_specifications: int,
        feature_dim: int,
        coverage_target: float = 0.90
    ):
        self.num_specifications = num_specifications
        self.models = [
            ConformizedQuantileRegression(
                feature_dim=feature_dim,
                coverage_target=coverage_target
            )
            for _ in range(num_specifications)
        ]
    
    def get_best_specification(
        self,
        context: np.ndarray,
        outcome: float
    ) -> int:
        """
        Find specification with best interval score (hindsight).
        
        NOTE: This cheats by using the true outcome.
        """
        best_spec = 0
        best_score = float('inf')
        
        for i, model in enumerate(self.models):
            lower, upper = model.predict_interval(context)
            score = interval_score(
                np.array([lower]),
                np.array([upper]),
                np.array([outcome])
            )[0]
            
            if score < best_score:
                best_score = score
                best_spec = i
        
        return best_spec, best_score
    
    def update(
        self,
        specification: int,
        context: np.ndarray,
        outcome: float
    ):
        """Update the model for the given specification."""
        self.models[specification].update(context, outcome)


def run_synthetic_experiment(
    config: Optional[ExperimentConfig] = None,
    num_series: int = 50,
    train_steps: int = 500,
    test_steps: int = 200,
    seed: int = 42
) -> ExperimentResults:
    """
    Run experiment on synthetic data.
    
    Args:
        config: Experiment configuration
        num_series: Number of time series
        train_steps: Training steps
        test_steps: Test steps
        seed: Random seed
    
    Returns:
        Experiment results
    """
    logger.info("Starting synthetic experiment")
    
    if config is None:
        config = get_default_config("synthetic")
    
    np.random.seed(seed)
    
    # Create data generator
    generator = SyntheticDataGenerator(
        num_series=num_series,
        num_specifications=config.spec_config.num_actions,
        feature_dim=config.ts_config.feature_dim,
        seed=seed
    )
    
    # Create CTS agent
    cts_config = CTSConfig(
        num_actions=config.spec_config.num_actions,
        feature_dim=config.ts_config.feature_dim,
        prior_precision=config.ts_config.prior_precision,
        exploration_variance=config.ts_config.exploration_variance,
        coverage_target=config.cqr_config.coverage_target,
        cqr_learning_rate=config.cqr_config.learning_rate,
        warmup_rounds=config.ts_config.warmup_rounds,
        seed=seed
    )
    cts_agent = ConformalThompsonSampling(cts_config)
    
    # Create baselines
    best_fixed = FixedSpecificationBaseline(
        specification=0,
        feature_dim=config.ts_config.feature_dim
    )
    ensemble = EqualWeightEnsemble(
        num_specifications=config.spec_config.num_actions,
        feature_dim=config.ts_config.feature_dim
    )
    oracle = OracleBaseline(
        num_specifications=config.spec_config.num_actions,
        feature_dim=config.ts_config.feature_dim
    )
    
    # Training phase
    logger.info(f"Training for {train_steps} steps...")
    
    cts_scores = []
    fixed_scores = []
    ensemble_scores = []
    oracle_scores = []
    
    for t in range(train_steps):
        # Generate observation (sample one series)
        series_idx = t % num_series
        data = generator.step(0)  # Dummy spec for data generation
        
        context = data['contexts'][series_idx]
        outcome = data['returns'][series_idx]
        optimal_spec = data['optimal_spec']
        
        # CTS agent
        action, lower_cts, upper_cts = cts_agent.select_and_predict(context)
        cts_agent.update(action, context, outcome)
        score_cts = interval_score(
            np.array([lower_cts]),
            np.array([upper_cts]),
            np.array([outcome])
        )[0]
        cts_scores.append(score_cts)
        
        # Fixed baseline (use specification 0)
        lower_fixed, upper_fixed = best_fixed.predict_interval(context)
        best_fixed.update(context, outcome)
        score_fixed = interval_score(
            np.array([lower_fixed]),
            np.array([upper_fixed]),
            np.array([outcome])
        )[0]
        fixed_scores.append(score_fixed)
        
        # Ensemble
        lower_ens, upper_ens = ensemble.predict_interval(context)
        ensemble.update(optimal_spec, context, outcome)
        score_ens = interval_score(
            np.array([lower_ens]),
            np.array([upper_ens]),
            np.array([outcome])
        )[0]
        ensemble_scores.append(score_ens)
        
        # Oracle
        best_spec, score_oracle = oracle.get_best_specification(context, outcome)
        oracle.update(best_spec, context, outcome)
        oracle_scores.append(score_oracle)
        
        if (t + 1) % 100 == 0:
            logger.info(f"Step {t + 1}/{train_steps} - "
                       f"CTS score: {np.mean(cts_scores[-100:]):.4f}, "
                       f"Fixed: {np.mean(fixed_scores[-100:]):.4f}")
    
    # Test phase
    logger.info(f"Testing for {test_steps} steps...")
    
    test_cts_scores = []
    test_fixed_scores = []
    test_ensemble_scores = []
    test_oracle_scores = []
    
    test_cts_coverages = []
    test_fixed_coverages = []
    
    for t in range(test_steps):
        series_idx = t % num_series
        data = generator.step(0)
        
        context = data['contexts'][series_idx]
        outcome = data['returns'][series_idx]
        optimal_spec = data['optimal_spec']
        
        # CTS
        action, lower_cts, upper_cts = cts_agent.select_and_predict(context)
        cts_agent.update(action, context, outcome)
        test_cts_scores.append(interval_score(
            np.array([lower_cts]), np.array([upper_cts]), np.array([outcome])
        )[0])
        test_cts_coverages.append(lower_cts <= outcome <= upper_cts)
        
        # Fixed
        lower_fixed, upper_fixed = best_fixed.predict_interval(context)
        best_fixed.update(context, outcome)
        test_fixed_scores.append(interval_score(
            np.array([lower_fixed]), np.array([upper_fixed]), np.array([outcome])
        )[0])
        test_fixed_coverages.append(lower_fixed <= outcome <= upper_fixed)
        
        # Ensemble
        lower_ens, upper_ens = ensemble.predict_interval(context)
        ensemble.update(optimal_spec, context, outcome)
        test_ensemble_scores.append(interval_score(
            np.array([lower_ens]), np.array([upper_ens]), np.array([outcome])
        )[0])
        
        # Oracle
        best_spec, score_oracle = oracle.get_best_specification(context, outcome)
        oracle.update(best_spec, context, outcome)
        test_oracle_scores.append(score_oracle)
    
    # Compute results
    method_scores = {
        'CTS': np.array(test_cts_scores),
        'Fixed': np.array(test_fixed_scores),
        'Ensemble': np.array(test_ensemble_scores),
        'Oracle': np.array(test_oracle_scores),
    }
    
    method_metrics = {}
    for name, scores in method_scores.items():
        method_metrics[name] = {
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'median_score': float(np.median(scores)),
        }
    
    # Statistical tests
    logger.info("Computing statistical tests...")
    dm_tests = compare_methods(method_scores, baseline_name='Fixed')
    
    # Bootstrap CIs
    bootstrap_cis = {}
    for name, scores in method_scores.items():
        mean, lower, upper = bootstrap_confidence_interval(scores)
        bootstrap_cis[name] = (mean, lower, upper)
    
    # Agent statistics
    stats = cts_agent.get_statistics()
    
    results = ExperimentResults(
        experiment_name="synthetic_experiment",
        timestamp=datetime.now().isoformat(),
        mean_interval_score=float(np.mean(test_cts_scores)),
        coverage_rate=float(np.mean(test_cts_coverages)),
        mean_width=0.0,  # Would need to track
        method_scores=method_scores,
        method_metrics=method_metrics,
        dm_tests=dm_tests,
        bootstrap_cis=bootstrap_cis,
        action_counts=np.array(stats['action_pull_counts']),
    )
    
    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT RESULTS")
    logger.info("=" * 60)
    
    for name, metrics in method_metrics.items():
        logger.info(f"{name}:")
        logger.info(f"  Mean score: {metrics['mean_score']:.4f} ± {metrics['std_score']:.4f}")
    
    logger.info("\nComparison to Fixed baseline:")
    for name, test_result in dm_tests.items():
        sig = "***" if test_result['p_value'] < 0.01 else \
              "**" if test_result['p_value'] < 0.05 else \
              "*" if test_result['p_value'] < 0.10 else ""
        logger.info(f"  {name}: {test_result['pct_improvement']:+.2f}% improvement "
                   f"(p={test_result['p_value']:.4f}) {sig}")
    
    logger.info(f"\nCTS Coverage: {results.coverage_rate:.2%}")
    logger.info(f"Action distribution: {stats['action_pull_counts']}")
    
    return results


def run_m5_simulation_experiment(
    num_series: int = 100,
    num_days: int = 300,
    seed: int = 42
) -> ExperimentResults:
    """
    Run experiment on simulated M5-like data.
    
    Args:
        num_series: Number of time series
        num_days: Number of days
        seed: Random seed
    
    Returns:
        Experiment results
    """
    logger.info("Starting M5 simulation experiment")
    
    np.random.seed(seed)
    
    # Create simulated data
    data = create_m5_simulation_data(
        num_series=num_series,
        num_days=num_days,
        seed=seed
    )
    
    selector = M5SpecificationSelector(data)
    
    # Create agent
    feature_dim = 1 + 3 * len(selector.lookback_windows)  # Bias + (mean, std, nz_rate) per window
    
    cts_config = CTSConfig(
        num_actions=selector.num_specifications,
        feature_dim=feature_dim,
        warmup_rounds=50,
        seed=seed
    )
    agent = ConformalThompsonSampling(cts_config)
    
    # Training/testing
    train_end = data['train_end_day']
    val_end = data['val_end_day']
    
    scores = []
    coverages = []
    actions_taken = []
    
    logger.info(f"Training from day {selector.current_day} to {train_end}...")
    
    while selector.current_day < num_days - max(selector.forecast_horizons):
        for series_idx in range(min(10, num_series)):  # Sample series
            context = selector.get_context(series_idx)
            
            # Agent selects and predicts
            action, lower, upper = agent.select_and_predict(context)
            
            # Get true outcome
            target, horizon = selector.get_target(series_idx, action)
            
            # Update agent
            agent.update(action, context, target)
            
            # Track metrics
            score = interval_score(
                np.array([lower]),
                np.array([upper]),
                np.array([target])
            )[0]
            scores.append(score)
            coverages.append(lower <= target <= upper)
            actions_taken.append(action)
        
        selector.step()
        
        if selector.current_day % 50 == 0:
            logger.info(f"Day {selector.current_day}: "
                       f"score={np.mean(scores[-100:]):.4f}, "
                       f"coverage={np.mean(coverages[-100:]):.2%}")
    
    # Results
    results = ExperimentResults(
        experiment_name="m5_simulation",
        timestamp=datetime.now().isoformat(),
        mean_interval_score=float(np.mean(scores)),
        coverage_rate=float(np.mean(coverages)),
        mean_width=0.0,
        action_counts=np.bincount(actions_taken, minlength=selector.num_specifications),
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("M5 SIMULATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Mean interval score: {results.mean_interval_score:.4f}")
    logger.info(f"Coverage rate: {results.coverage_rate:.2%}")
    logger.info(f"Action distribution: {results.action_counts}")
    
    return results


def main():
    """Main entry point for experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Conformal Thompson Sampling experiments"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="synthetic",
        choices=["synthetic", "m5_sim"],
        help="Experiment type"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiment
    if args.experiment == "synthetic":
        results = run_synthetic_experiment(seed=args.seed)
    elif args.experiment == "m5_sim":
        results = run_m5_simulation_experiment(seed=args.seed)
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")
    
    # Save results
    results_path = output_dir / f"{args.experiment}_results.json"
    results.save(str(results_path))
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
