"""
Full-Scale Experiment Runner.

Runs Conformal Thompson Sampling experiments on real benchmark datasets
with proper baselines, evaluation, and result logging.

Usage:
    # M5 experiment
    python -m conformal_ts.experiments.run_full_experiment \
        --dataset m5 \
        --output ./results/m5 \
        --seed 42

    # GEFCom experiment  
    python -m conformal_ts.experiments.run_full_experiment \
        --dataset gefcom \
        --track solar \
        --output ./results/gefcom \
        --seed 42
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
import logging
import json
import pickle
from datetime import datetime
import time
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for full experiment."""
    
    # Dataset
    dataset: str = "m5"  # "m5" or "gefcom"
    gefcom_track: str = "solar"  # For GEFCom: "solar", "wind", "load", "price"
    
    # Data settings
    m5_aggregation_levels: List[int] = field(default_factory=lambda: [9, 10])
    m5_max_series: Optional[int] = 500  # Sample for faster iteration
    
    # Specification space
    lookback_windows: List[int] = field(default_factory=lambda: [7, 14, 28, 56])
    forecast_horizons: List[int] = field(default_factory=lambda: [7, 14, 28])
    
    # CTS agent settings
    prior_precision: float = 0.1
    exploration_variance: float = 5.0
    warmup_rounds: int = 100
    coverage_target: float = 0.90
    cqr_learning_rate: float = 0.02
    calibration_window: int = 250
    
    # Training settings
    batch_size: int = 100  # Series per day
    train_epochs: int = 1  # Passes through training data
    
    # Evaluation settings
    eval_batch_size: int = 200
    
    # Baselines
    use_lightgbm: bool = True
    use_aci: bool = True
    
    # Logging and checkpointing
    log_interval: int = 50  # Steps between logging
    checkpoint_interval: int = 200  # Steps between checkpoints
    
    # Output
    output_dir: str = "./results"
    save_predictions: bool = True
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration."""
        if self.dataset not in ["m5", "gefcom"]:
            raise ValueError(f"Unknown dataset: {self.dataset}")


class FullExperimentRunner:
    """
    Run complete experiments with proper evaluation.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seeds
        np.random.seed(config.seed)
        
        # Components (initialized in setup)
        self.dataset = None
        self.agent = None
        self.baselines = {}
        
        # Results tracking
        self.results = {
            'config': asdict(config),
            'training': {},
            'evaluation': {},
            'timing': {},
        }
    
    def setup(self):
        """Initialize dataset, agent, and baselines."""
        logger.info("Setting up experiment...")
        
        # Load dataset
        if self.config.dataset == "m5":
            self._setup_m5()
        else:
            self._setup_gefcom()
        
        # Create CTS agent
        self._setup_agent()
        
        # Create baselines
        self._setup_baselines()
        
        logger.info("Setup complete")
    
    def _setup_m5(self):
        """Load M5 dataset."""
        from conformal_ts.data.m5_real import M5RealDataLoader, M5Config
        
        config = M5Config(
            aggregation_levels=self.config.m5_aggregation_levels,
            max_series_per_level=self.config.m5_max_series,
            lookback_windows=self.config.lookback_windows,
            forecast_horizons=self.config.forecast_horizons,
        )
        
        loader = M5RealDataLoader(config)
        
        try:
            self.dataset = loader.prepare_dataset()
            logger.info(f"Loaded M5 dataset: {self.dataset['num_series']} series")
        except Exception as e:
            logger.warning(f"Could not load real M5 data: {e}")
            logger.info("Falling back to simulated data")
            
            from conformal_ts.data.m5_loader import create_m5_simulation_data
            self.dataset = create_m5_simulation_data(
                num_series=self.config.m5_max_series or 500,
                num_days=1000,
                seed=self.config.seed
            )
            
            # Add missing fields
            self.dataset['specifications'] = [
                (lb, fh)
                for lb in self.config.lookback_windows
                for fh in self.config.forecast_horizons
            ]
            self.dataset['num_specifications'] = len(self.dataset['specifications'])
            self.dataset['scale_factors'] = np.ones(self.dataset['num_series'])
    
    def _setup_gefcom(self):
        """Load GEFCom dataset."""
        from conformal_ts.data.gefcom2014 import GEFCom2014Loader, GEFComConfig
        
        config = GEFComConfig(
            track=self.config.gefcom_track,
            lookback_hours=self.config.lookback_windows,
            forecast_horizons=self.config.forecast_horizons,
        )
        
        loader = GEFCom2014Loader(config)
        self.dataset = loader.prepare_dataset()
        logger.info(f"Loaded GEFCom {config.track}: {self.dataset['num_zones']} zones")
    
    def _setup_agent(self):
        """Create CTS agent."""
        from conformal_ts.models.cts_agent import ConformalThompsonSampling, CTSConfig
        
        # Compute feature dimension
        feature_dim = self._compute_feature_dim()
        
        cts_config = CTSConfig(
            num_actions=self.dataset['num_specifications'],
            feature_dim=feature_dim,
            prior_precision=self.config.prior_precision,
            exploration_variance=self.config.exploration_variance,
            warmup_rounds=self.config.warmup_rounds,
            coverage_target=self.config.coverage_target,
            cqr_learning_rate=self.config.cqr_learning_rate,
            calibration_window=self.config.calibration_window,
            seed=self.config.seed,
        )
        
        self.agent = ConformalThompsonSampling(cts_config)
        logger.info(f"Created CTS agent with {feature_dim} features, "
                   f"{self.dataset['num_specifications']} specifications")
    
    def _setup_baselines(self):
        """Create baseline methods."""
        from conformal_ts.baselines.baselines import (
            create_baselines, LightGBMQuantileBaseline
        )
        
        feature_dim = self._compute_feature_dim()
        
        self.baselines = create_baselines(
            num_specifications=self.dataset['num_specifications'],
            feature_dim=feature_dim,
            use_lightgbm=self.config.use_lightgbm,
            seed=self.config.seed
        )
        
        logger.info(f"Created baselines: {list(self.baselines.keys())}")
    
    def _compute_feature_dim(self) -> int:
        """Compute feature dimension based on config."""
        # Bias + (mean, std, nz_rate, max, trend) per lookback window
        # + calendar features (dow, month, event, snap)
        n_windows = len(self.config.lookback_windows)
        return 1 + 5 * n_windows + 4
    
    def _compute_features(
        self,
        series_idx: int,
        day_idx: int
    ) -> np.ndarray:
        """Compute context features for a series at given day."""
        sales_matrix = self.dataset['sales_matrix']
        
        features = [1.0]  # Bias
        
        for window in self.config.lookback_windows:
            start = max(0, day_idx - window)
            history = sales_matrix[series_idx, start:day_idx]
            
            if len(history) > 0:
                features.extend([
                    np.mean(history),
                    np.std(history) + 1e-6,
                    np.mean(history > 0),
                    np.max(history),
                ])
                # Trend
                if len(history) >= 14:
                    recent = np.mean(history[-7:])
                    previous = np.mean(history[-14:-7])
                    trend = (recent - previous) / (previous + 1e-6)
                    features.append(np.clip(trend, -2, 2))
                else:
                    features.append(0.0)
            else:
                features.extend([0.0, 1.0, 0.5, 0.0, 0.0])
        
        # Calendar features (simplified)
        features.extend([
            (day_idx % 7) / 6.0,  # Day of week proxy
            ((day_idx // 30) % 12) / 11.0,  # Month proxy
            0.0,  # Event placeholder
            0.0,  # SNAP placeholder
        ])
        
        return np.array(features, dtype=np.float32)
    
    def run_training(self) -> Dict[str, Any]:
        """Run training loop."""
        from conformal_ts.evaluation.competition_metrics import interval_score
        
        logger.info("Starting training...")
        start_time = time.time()
        
        sales_matrix = self.dataset['sales_matrix']
        num_series = self.dataset['num_series']
        specifications = self.dataset['specifications']
        
        train_end = self.dataset.get('train_end_day', int(sales_matrix.shape[1] * 0.7))
        start_day = max(self.config.lookback_windows) + 1
        max_horizon = max(self.config.forecast_horizons)
        
        # Tracking
        all_scores = []
        all_coverages = []
        all_actions = []
        
        # Collect data for batch-training baselines (like LightGBM)
        collected_contexts = []
        collected_targets = []
        
        # Pending updates (deferred for horizon)
        pending_updates = {}
        
        total_steps = 0
        
        for epoch in range(self.config.train_epochs):
            logger.info(f"Training epoch {epoch + 1}/{self.config.train_epochs}")
            
            for day in range(start_day, train_end - max_horizon):
                # Process pending updates from previous days
                if day in pending_updates:
                    for update in pending_updates.pop(day):
                        target = self._get_target(
                            update['series_idx'],
                            update['day'],
                            update['horizon']
                        )
                        
                        # Compute score
                        score = interval_score(
                            np.array([update['lower']]),
                            np.array([update['upper']]),
                            np.array([target]),
                            alpha=0.10
                        )[0]
                        
                        covered = update['lower'] <= target <= update['upper']
                        
                        all_scores.append(score)
                        all_coverages.append(covered)
                        
                        # Collect for batch training
                        collected_contexts.append(update['context'])
                        collected_targets.append(target)
                        
                        # Update agent
                        self.agent.update(
                            update['action'],
                            update['context'],
                            target
                        )
                        
                        # Update baselines
                        for baseline in self.baselines.values():
                            baseline.update(update['context'], target)
                
                # Sample batch of series
                batch_indices = np.random.choice(
                    num_series,
                    size=min(self.config.batch_size, num_series),
                    replace=False
                )
                
                for series_idx in batch_indices:
                    # Compute context
                    context = self._compute_features(series_idx, day)
                    
                    # Agent selects action
                    action, lower, upper = self.agent.select_and_predict(context)
                    
                    # Get horizon for this specification
                    _, horizon = specifications[action]
                    
                    # Schedule deferred update
                    outcome_day = day + horizon
                    if outcome_day not in pending_updates:
                        pending_updates[outcome_day] = []
                    
                    pending_updates[outcome_day].append({
                        'series_idx': series_idx,
                        'day': day,
                        'action': action,
                        'context': context,
                        'lower': lower,
                        'upper': upper,
                        'horizon': horizon,
                    })
                    
                    all_actions.append(action)
                    total_steps += 1
                
                # Logging
                if day % self.config.log_interval == 0 and all_scores:
                    recent_scores = all_scores[-1000:]
                    recent_cov = all_coverages[-1000:]
                    logger.info(
                        f"Day {day}/{train_end}: "
                        f"score={np.mean(recent_scores):.2f}, "
                        f"coverage={np.mean(recent_cov):.2%}, "
                        f"steps={total_steps}"
                    )
        
        # Flush remaining pending updates
        for future_day in sorted(pending_updates.keys()):
            for update in pending_updates[future_day]:
                target = self._get_target(
                    update['series_idx'],
                    update['day'],
                    update['horizon']
                )
                collected_contexts.append(update['context'])
                collected_targets.append(target)
                all_scores.append(interval_score(
                    np.array([update['lower']]),
                    np.array([update['upper']]),
                    np.array([target])
                )[0])
                all_coverages.append(update['lower'] <= target <= update['upper'])
        
        # Fit batch-training baselines (like LightGBM)
        if collected_contexts:
            contexts_array = np.array(collected_contexts)
            targets_array = np.array(collected_targets)
            
            logger.info(f"Fitting batch baselines on {len(targets_array)} samples...")
            
            for name, baseline in self.baselines.items():
                try:
                    # Check if baseline needs batch fitting
                    if hasattr(baseline, '_fitted') and not baseline._fitted:
                        baseline.fit(contexts_array, targets_array)
                        logger.info(f"  Fitted {name}")
                except Exception as e:
                    logger.warning(f"  Failed to fit {name}: {e}")
        
        training_time = time.time() - start_time
        
        # Compute action distribution
        action_counts = np.bincount(all_actions, minlength=len(specifications))
        
        results = {
            'total_steps': total_steps,
            'training_time': training_time,
            'final_mean_score': float(np.mean(all_scores[-1000:])) if all_scores else 0,
            'final_coverage': float(np.mean(all_coverages[-1000:])) if all_coverages else 0,
            'action_distribution': action_counts.tolist(),
        }
        
        logger.info(f"Training complete in {training_time:.1f}s")
        logger.info(f"Final score: {results['final_mean_score']:.2f}")
        logger.info(f"Final coverage: {results['final_coverage']:.2%}")
        
        self.results['training'] = results
        return results
    
    def _get_target(self, series_idx: int, day: int, horizon: int) -> float:
        """Get target value (sum of sales over horizon)."""
        sales_matrix = self.dataset['sales_matrix']
        target_end = min(day + horizon, sales_matrix.shape[1])
        return float(np.sum(sales_matrix[series_idx, day:target_end]))
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run evaluation on test set."""
        from conformal_ts.evaluation.competition_metrics import (
            interval_score, coverage_rate, diebold_mariano_test,
            bootstrap_confidence_interval, m5_wspl_from_intervals
        )
        
        logger.info("Running evaluation...")
        start_time = time.time()
        
        sales_matrix = self.dataset['sales_matrix']
        num_series = self.dataset['num_series']
        specifications = self.dataset['specifications']
        
        val_end = self.dataset.get('val_end_day', int(sales_matrix.shape[1] * 0.85))
        test_end = self.dataset.get('test_end_day', sales_matrix.shape[1])
        max_horizon = max(self.config.forecast_horizons)
        
        # Collect predictions
        results = {
            'cts': {'lowers': [], 'uppers': [], 'targets': [], 'actions': []},
        }
        for name in self.baselines:
            results[name] = {'lowers': [], 'uppers': [], 'targets': []}
        
        for day in range(val_end, test_end - max_horizon):
            # Sample series for evaluation
            batch_indices = np.random.choice(
                num_series,
                size=min(self.config.eval_batch_size, num_series),
                replace=False
            )
            
            for series_idx in batch_indices:
                context = self._compute_features(series_idx, day)
                
                # CTS prediction
                action, lower, upper = self.agent.select_and_predict(context)
                _, horizon = specifications[action]
                target = self._get_target(series_idx, day, horizon)
                
                results['cts']['lowers'].append(lower)
                results['cts']['uppers'].append(upper)
                results['cts']['targets'].append(target)
                results['cts']['actions'].append(action)
                
                # Baseline predictions
                for name, baseline in self.baselines.items():
                    try:
                        b_lower, b_upper = baseline.predict(context)
                        results[name]['lowers'].append(b_lower)
                        results[name]['uppers'].append(b_upper)
                        results[name]['targets'].append(target)
                    except RuntimeError as e:
                        # Baseline not fitted - skip
                        pass
        
        # Compute metrics for each method
        eval_results = {}
        
        for method, data in results.items():
            lowers = np.array(data['lowers'])
            uppers = np.array(data['uppers'])
            targets = np.array(data['targets'])
            
            # Skip methods with no predictions
            if len(targets) == 0:
                logger.warning(f"No predictions for {method}, skipping evaluation")
                continue
            
            scores = interval_score(lowers, uppers, targets, alpha=0.10)
            cov = coverage_rate(lowers, uppers, targets)
            widths = uppers - lowers
            
            # Bootstrap CI
            mean_score, ci_lo, ci_hi = bootstrap_confidence_interval(scores)
            
            eval_results[method] = {
                'mean_score': float(mean_score),
                'score_ci_lower': float(ci_lo),
                'score_ci_upper': float(ci_hi),
                'coverage': float(cov),
                'mean_width': float(np.mean(widths)),
                'n_samples': len(targets),
            }
            
            # WSPL for M5
            if 'scale_factors' in self.dataset and len(targets) > 0:
                scale = self.dataset['scale_factors']
                # Repeat scale factors for all predictions
                scale_repeated = np.tile(scale, len(targets) // len(scale) + 1)[:len(targets)]
                wspl = m5_wspl_from_intervals(lowers, uppers, targets, scale_repeated)
                eval_results[method]['wspl'] = float(wspl)
            
            if method == 'cts':
                actions = np.array(data['actions'], dtype=np.int64)
                eval_results[method]['action_distribution'] = (
                    np.bincount(actions, minlength=len(specifications)).tolist()
                )
        
        # Diebold-Mariano tests vs CTS
        cts_scores = interval_score(
            np.array(results['cts']['lowers']),
            np.array(results['cts']['uppers']),
            np.array(results['cts']['targets']),
            alpha=0.10
        )
        
        dm_tests = {}
        for name in self.baselines:
            # Skip baselines with no predictions
            if len(results[name]['targets']) == 0:
                continue
            baseline_scores = interval_score(
                np.array(results[name]['lowers']),
                np.array(results[name]['uppers']),
                np.array(results[name]['targets']),
                alpha=0.10
            )
            dm_tests[name] = diebold_mariano_test(cts_scores, baseline_scores)
        
        eval_results['dm_tests'] = dm_tests
        
        eval_time = time.time() - start_time
        eval_results['eval_time'] = eval_time
        
        # Log results
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        
        for method, metrics in eval_results.items():
            if isinstance(metrics, dict) and 'mean_score' in metrics:
                logger.info(f"\n{method}:")
                logger.info(f"  Score: {metrics['mean_score']:.2f} "
                           f"[{metrics['score_ci_lower']:.2f}, {metrics['score_ci_upper']:.2f}]")
                logger.info(f"  Coverage: {metrics['coverage']:.2%}")
                logger.info(f"  Width: {metrics['mean_width']:.2f}")
        
        logger.info("\nDiebold-Mariano Tests (CTS vs baselines):")
        for name, test in dm_tests.items():
            sig = "***" if test['significant_0.01'] else \
                  "**" if test['significant_0.05'] else \
                  "*" if test['significant_0.10'] else ""
            logger.info(f"  vs {name}: {test['pct_improvement']:+.2f}% "
                       f"(p={test['p_value']:.4f}) {sig}")
        
        self.results['evaluation'] = eval_results
        return eval_results
    
    def save_results(self):
        """Save all results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON summary
        results_path = self.output_dir / f"results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Results saved to {results_path}")
        
        # Save agent state
        agent_path = self.output_dir / f"agent_{timestamp}.json"
        self.agent.save(str(agent_path))
        
        # Save latest symlink
        latest_path = self.output_dir / "results_latest.json"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(results_path.name)
    
    def run(self) -> Dict[str, Any]:
        """Run complete experiment."""
        logger.info("=" * 60)
        logger.info("CONFORMAL THOMPSON SAMPLING EXPERIMENT")
        logger.info("=" * 60)
        logger.info(f"Dataset: {self.config.dataset}")
        logger.info(f"Output: {self.output_dir}")
        
        # Setup
        self.setup()
        
        # Train
        self.run_training()
        
        # Evaluate
        self.run_evaluation()
        
        # Save
        self.save_results()
        
        logger.info("Experiment complete!")
        return self.results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Conformal Thompson Sampling experiments"
    )
    
    parser.add_argument(
        "--dataset", type=str, default="m5",
        choices=["m5", "gefcom"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--track", type=str, default="solar",
        choices=["solar", "wind", "load", "price"],
        help="GEFCom track (if using gefcom dataset)"
    )
    parser.add_argument(
        "--output", type=str, default="./results",
        help="Output directory"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--max-series", type=int, default=500,
        help="Maximum series to use (for faster iteration)"
    )
    parser.add_argument(
        "--no-lightgbm", action="store_true",
        help="Disable LightGBM baseline"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick run with reduced data for testing"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = ExperimentConfig(
        dataset=args.dataset,
        gefcom_track=args.track,
        output_dir=args.output,
        seed=args.seed,
        m5_max_series=args.max_series if not args.quick else 100,
        use_lightgbm=not args.no_lightgbm,
    )
    
    if args.quick:
        config.train_epochs = 1
        config.batch_size = 50
        config.log_interval = 20
    
    # Run
    runner = FullExperimentRunner(config)
    results = runner.run()
    
    return results


if __name__ == "__main__":
    main()