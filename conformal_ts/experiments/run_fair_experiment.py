"""
Fair Experiment Runner - Patches for run_full_experiment.py

This module provides the patches needed to use fair baselines in
your existing experiment infrastructure.

USAGE:
    In your run_full_experiment.py, replace the baseline setup with:
    
    ```python
    from conformal_ts.baselines.baselines_fair import (
        create_baselines,
        setup_shared_infrastructure,
        SharedQuantileModels,
        SharedConformalCalibrator
    )
    
    # In _setup_baselines():
    self.baselines = create_baselines(
        num_specifications=self.dataset['num_specifications'],
        feature_dim=feature_dim,
        use_lightgbm=self.config.use_lightgbm,
        use_fair_baselines=True,  # <-- CRITICAL
        seed=self.config.seed
    )
    
    # After collecting training data, before evaluation:
    setup_shared_infrastructure(
        baselines=self.baselines,
        num_specifications=self.dataset['num_specifications'],
        contexts=contexts_array,
        targets=targets_array,
        calibration_window=self.config.calibration_window,
        coverage_target=self.config.coverage_target
    )
    ```

Or, use the FairExperimentRunner class below which wraps everything.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)


def patch_full_experiment_runner(runner: 'FullExperimentRunner'):
    """
    Patch an existing FullExperimentRunner to use fair baselines.
    
    Call this after creating the runner but before run():
    
        runner = FullExperimentRunner(config)
        patch_full_experiment_runner(runner)
        runner.run()
    """
    from conformal_ts.baselines.baselines_fair import (
        create_baselines,
        setup_shared_infrastructure,
    )
    
    # Store original methods
    original_setup_baselines = runner._setup_baselines
    original_run_training = runner.run_training
    
    def patched_setup_baselines(self):
        """Create fair baselines instead of legacy ones."""
        feature_dim = self._compute_feature_dim()
        
        self.baselines = create_baselines(
            num_specifications=self.dataset['num_specifications'],
            feature_dim=feature_dim,
            use_lightgbm=self.config.use_lightgbm,
            use_fair_baselines=True,
            seed=self.config.seed
        )
        
        logger.info(f"Created FAIR baselines: {list(self.baselines.keys())}")
    
    def patched_run_training(self):
        """Run training and setup shared infrastructure."""
        # Run original training to collect data
        results = original_run_training()
        
        # If we collected training data, setup shared infrastructure
        if hasattr(self, '_collected_contexts') and len(self._collected_contexts) > 0:
            contexts = np.array(self._collected_contexts)
            targets = np.array(self._collected_targets)
            
            logger.info(f"\nSetting up shared infrastructure on {len(targets)} samples...")
            
            setup_shared_infrastructure(
                baselines=self.baselines,
                num_specifications=self.dataset['num_specifications'],
                contexts=contexts,
                targets=targets,
                calibration_window=self.config.calibration_window,
                coverage_target=self.config.coverage_target,
                seed=self.config.seed
            )
        
        return results
    
    # Apply patches
    runner._setup_baselines = lambda: patched_setup_baselines(runner)
    runner.run_training = lambda: patched_run_training(runner)
    
    logger.info("Patched FullExperimentRunner to use fair baselines")


@dataclass
class FairExperimentConfig:
    """Configuration for fair comparison experiment."""
    
    # Dataset
    dataset: str = "m5"
    gefcom_track: str = "solar"
    
    # Data settings
    m5_aggregation_levels: List[int] = field(default_factory=lambda: [9, 10])
    m5_max_series: Optional[int] = 500
    
    # Specification space
    lookback_windows: List[int] = field(default_factory=lambda: [7, 14, 28, 56])
    forecast_horizons: List[int] = field(default_factory=lambda: [7, 14, 28])
    
    # CTS agent settings
    prior_precision: float = 1.0
    exploration_variance: float = 1.0
    warmup_rounds: int = 100
    coverage_target: float = 0.90
    cqr_learning_rate: float = 0.02
    calibration_window: int = 250
    
    # Training
    batch_size: int = 100
    train_epochs: int = 1
    
    # Evaluation
    eval_batch_size: int = 200
    
    # Baselines
    use_lightgbm: bool = True
    
    # Logging
    log_interval: int = 50
    
    # Output
    output_dir: str = "./results"
    
    # Reproducibility
    seed: int = 42


class FairComparisonExperiment:
    """
    Fair comparison experiment runner.
    
    This is a simplified experiment runner that ensures methodologically
    correct comparisons. All methods use:
    - Same LightGBM base models (trained once)
    - Same conformal calibration infrastructure
    - Only differ in specification selection
    
    Usage:
        config = FairExperimentConfig(dataset="m5")
        exp = FairComparisonExperiment(config)
        results = exp.run()
    """
    
    def __init__(self, config: FairExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        np.random.seed(config.seed)
        
        self.dataset = None
        self.agent = None
        self.baselines = {}
        self.shared_models = None
        self.shared_calibrator = None
        
        self.results = {
            'config': asdict(config),
            'training': {},
            'evaluation': {},
        }
    
    def setup(self):
        """Load data and initialize components."""
        logger.info("Setting up fair comparison experiment...")
        
        # Load dataset
        self._load_dataset()
        
        # Create CTS agent
        self._create_agent()
        
        # Create baselines (will be connected to shared infra later)
        self._create_baselines()
        
        logger.info("Setup complete")
    
    def _load_dataset(self):
        """Load dataset."""
        if self.config.dataset == "m5":
            self._load_m5()
        else:
            self._load_gefcom()
    
    def _load_m5(self):
        """Load M5 or simulated data."""
        try:
            from conformal_ts.data.m5_real import M5RealDataLoader, M5Config
            
            config = M5Config(
                aggregation_levels=self.config.m5_aggregation_levels,
                max_series_per_level=self.config.m5_max_series,
                lookback_windows=self.config.lookback_windows,
                forecast_horizons=self.config.forecast_horizons,
            )
            
            loader = M5RealDataLoader(config)
            self.dataset = loader.prepare_dataset()
            logger.info(f"Loaded real M5: {self.dataset['num_series']} series")
            
        except Exception as e:
            logger.warning(f"Could not load M5: {e}, using simulation")
            self._create_simulated_data()
    
    def _load_gefcom(self):
        """Load GEFCom data."""
        try:
            from conformal_ts.data.gefcom2014 import GEFCom2014Loader, GEFComConfig
            
            config = GEFComConfig(
                track=self.config.gefcom_track,
                lookback_hours=self.config.lookback_windows,
                forecast_horizons=self.config.forecast_horizons,
            )
            
            loader = GEFCom2014Loader(config)
            self.dataset = loader.prepare_dataset()
            
        except Exception as e:
            logger.warning(f"Could not load GEFCom: {e}, using simulation")
            self._create_simulated_data()
    
    def _create_simulated_data(self):
        """Create simulated M5-like data."""
        num_series = self.config.m5_max_series or 500
        num_days = 400
        
        # Specifications
        specs = [
            (lb, h) for lb in self.config.lookback_windows
            for h in self.config.forecast_horizons
        ]
        
        # Generate data
        np.random.seed(self.config.seed)
        
        sales_matrix = np.zeros((num_series, num_days))
        
        for i in range(num_series):
            base = np.random.uniform(5, 50)
            trend = np.random.uniform(-0.01, 0.02)
            seasonality = np.random.uniform(0.1, 0.3)
            zero_prob = np.random.uniform(0.1, 0.5)
            
            for d in range(num_days):
                mean = base * (1 + trend * d / 100)
                mean *= (1 + seasonality * np.sin(2 * np.pi * d / 7))
                
                if np.random.random() < zero_prob:
                    sales_matrix[i, d] = 0
                else:
                    sales_matrix[i, d] = max(0, np.random.poisson(mean))
        
        self.dataset = {
            'num_series': num_series,
            'num_days': num_days,
            'sales_matrix': sales_matrix,
            'specifications': specs,
            'num_specifications': len(specs),
            'scale_factors': np.ones(num_series),
            'train_end': int(num_days * 0.7),
            'val_end': int(num_days * 0.85),
        }
        
        logger.info(f"Created simulated data: {num_series} series, {num_days} days")
    
    def _compute_feature_dim(self) -> int:
        """Compute feature dimension."""
        n_windows = len(self.config.lookback_windows)
        return 1 + 5 * n_windows + 4  # bias + features per window + calendar
    
    def _create_agent(self):
        """Create CTS agent."""
        from conformal_ts.models.cts_agent import ConformalThompsonSampling, CTSConfig
        
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
        logger.info(f"Created CTS agent: {feature_dim} features, {self.dataset['num_specifications']} specs")
    
    def _create_baselines(self):
        """Create fair baselines."""
        from conformal_ts.baselines.baselines_fair import create_baselines
        
        feature_dim = self._compute_feature_dim()
        
        self.baselines = create_baselines(
            num_specifications=self.dataset['num_specifications'],
            feature_dim=feature_dim,
            use_lightgbm=self.config.use_lightgbm,
            use_fair_baselines=True,
            seed=self.config.seed
        )
        
        logger.info(f"Created fair baselines: {list(self.baselines.keys())}")
    
    def _compute_features(self, series_idx: int, day: int) -> np.ndarray:
        """Compute context features for a series at given day."""
        sales = self.dataset['sales_matrix']
        lookbacks = self.config.lookback_windows
        
        features = [1.0]  # Bias
        
        for lb in lookbacks:
            if day >= lb:
                window = sales[series_idx, day-lb:day]
            else:
                window = sales[series_idx, :day] if day > 0 else np.array([0])
            
            features.append(np.mean(window))
            features.append(np.std(window) + 1e-6)
            features.append(np.mean(window > 0))
            features.append(np.max(window) if len(window) > 0 else 0)
            features.append(window[-1] - np.mean(window) if len(window) > 1 else 0)
        
        # Calendar features
        features.append(day % 7 / 7)
        features.append((day // 7) % 4 / 4)
        features.append(np.sin(2 * np.pi * day / 365))
        features.append(np.cos(2 * np.pi * day / 365))
        
        return np.array(features)
    
    def run_training(self) -> Dict[str, Any]:
        """Run training phase and setup shared infrastructure."""
        from conformal_ts.baselines.baselines_fair import setup_shared_infrastructure
        from conformal_ts.evaluation.metrics import interval_score
        
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING PHASE")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Collect training data
        contexts_list = []
        targets_list = []
        actions_list = []
        scores_list = []
        
        train_end = self.dataset.get('train_end', int(self.dataset['num_days'] * 0.7))
        max_lookback = max(self.config.lookback_windows)
        max_horizon = max(self.config.forecast_horizons)
        
        num_series = self.dataset['num_series']
        sales = self.dataset['sales_matrix']
        specs = self.dataset['specifications']
        
        total_steps = 0
        
        for day in range(max_lookback, train_end - max_horizon):
            batch_indices = np.random.choice(
                num_series,
                size=min(self.config.batch_size, num_series),
                replace=False
            )
            
            for series_idx in batch_indices:
                context = self._compute_features(series_idx, day)
                
                # CTS selects action and predicts
                action, lower, upper = self.agent.select_and_predict(context)
                
                # Get target
                _, horizon = specs[action]
                target = float(np.sum(sales[series_idx, day:day+horizon]))
                
                # Store for shared infrastructure
                contexts_list.append(context)
                targets_list.append(target)
                actions_list.append(action)
                
                # Compute score
                score = interval_score(
                    np.array([lower]), np.array([upper]), np.array([target])
                )[0]
                scores_list.append(score)
                
                # Update agent
                self.agent.update(action, context, target)
                
                total_steps += 1
            
            if (day - max_lookback) % self.config.log_interval == 0:
                recent_score = np.mean(scores_list[-1000:]) if scores_list else 0
                logger.info(f"Day {day}: steps={total_steps}, recent_score={recent_score:.2f}")
        
        training_time = time.time() - start_time
        
        # Setup shared infrastructure with collected data
        logger.info(f"\nSetting up shared infrastructure on {len(targets_list)} samples...")
        
        contexts_array = np.array(contexts_list)
        targets_array = np.array(targets_list)
        
        self.shared_models, self.shared_calibrator = setup_shared_infrastructure(
            baselines=self.baselines,
            num_specifications=self.dataset['num_specifications'],
            contexts=contexts_array,
            targets=targets_array,
            calibration_window=self.config.calibration_window,
            coverage_target=self.config.coverage_target,
            seed=self.config.seed
        )
        
        # Fit LightGBM separately
        if 'lightgbm' in self.baselines:
            self.baselines['lightgbm'].fit(contexts_array, targets_array)
        
        results = {
            'total_steps': total_steps,
            'training_time': training_time,
            'final_score': float(np.mean(scores_list[-1000:])) if scores_list else 0,
        }
        
        self.results['training'] = results
        logger.info(f"Training complete: {training_time:.1f}s, final_score={results['final_score']:.2f}")
        
        return results
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Evaluate all methods on test set."""
        from conformal_ts.evaluation.metrics import interval_score, diebold_mariano_test
        
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION PHASE (fair comparison)")
        logger.info("=" * 60)
        
        val_end = self.dataset.get('val_end', int(self.dataset['num_days'] * 0.85))
        num_days = self.dataset['num_days']
        max_lookback = max(self.config.lookback_windows)
        max_horizon = max(self.config.forecast_horizons)
        
        num_series = self.dataset['num_series']
        sales = self.dataset['sales_matrix']
        specs = self.dataset['specifications']
        
        # Collect predictions
        results = {name: {'lowers': [], 'uppers': [], 'targets': []}
                   for name in ['cts'] + list(self.baselines.keys())}
        
        for day in range(val_end, num_days - max_horizon):
            batch_indices = np.random.choice(
                num_series,
                size=min(self.config.eval_batch_size, num_series),
                replace=False
            )
            
            for series_idx in batch_indices:
                context = self._compute_features(series_idx, day)
                
                # Get target (use fixed horizon for fair comparison)
                horizon = specs[0][1]  # Use first spec's horizon
                target = float(np.sum(sales[series_idx, day:day+horizon]))
                
                # CTS prediction
                action, lower, upper = self.agent.select_and_predict(context)
                results['cts']['lowers'].append(lower)
                results['cts']['uppers'].append(upper)
                results['cts']['targets'].append(target)
                
                # Baseline predictions
                for name, baseline in self.baselines.items():
                    if name == 'oracle':
                        lower, upper, _ = baseline.oracle_predict_with_target(context, target)
                    else:
                        lower, upper = baseline.predict(context)
                    
                    results[name]['lowers'].append(lower)
                    results[name]['uppers'].append(upper)
                    results[name]['targets'].append(target)
        
        # Compute metrics
        eval_results = {}
        
        for method, data in results.items():
            if len(data['targets']) == 0:
                continue
            
            lowers = np.array(data['lowers'])
            uppers = np.array(data['uppers'])
            targets = np.array(data['targets'])
            
            scores = interval_score(lowers, uppers, targets, alpha=0.10)
            coverage = np.mean((targets >= lowers) & (targets <= uppers))
            width = np.mean(uppers - lowers)
            
            eval_results[method] = {
                'mean_score': float(np.mean(scores)),
                'coverage': float(coverage),
                'mean_width': float(width),
                'scores': scores,
            }
        
        # DM tests
        cts_scores = eval_results['cts']['scores']
        dm_tests = {}
        
        for name in eval_results:
            if name == 'cts':
                continue
            
            dm_tests[name] = diebold_mariano_test(
                cts_scores,
                eval_results[name]['scores']
            )
        
        eval_results['dm_tests'] = dm_tests
        
        # Log results
        logger.info("\n" + "-" * 60)
        logger.info("RESULTS (all baselines use SAME models)")
        logger.info("-" * 60)
        
        for method, metrics in sorted(
            [(m, v) for m, v in eval_results.items() if isinstance(v, dict) and 'mean_score' in v],
            key=lambda x: x[1]['mean_score']
        ):
            logger.info(
                f"{method:12s}  score={metrics['mean_score']:8.2f}  "
                f"coverage={metrics['coverage']:5.1%}  "
                f"width={metrics['mean_width']:8.2f}"
            )
        
        logger.info("\nDM Tests (CTS vs baselines):")
        for name, test in dm_tests.items():
            sig = "**" if test.get('significant_0.01') else "*" if test.get('significant_0.05') else ""
            pct = test.get('pct_improvement', 0)
            p = test.get('p_value', 1)
            logger.info(f"  vs {name:12s}: {pct:+.1f}% (p={p:.4f}) {sig}")
        
        self.results['evaluation'] = {
            k: v if not isinstance(v, dict) or 'scores' not in v else {
                kk: vv for kk, vv in v.items() if kk != 'scores'
            }
            for k, v in eval_results.items()
        }
        
        return eval_results
    
    def save_results(self):
        """Save results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_path = self.output_dir / f"fair_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
    
    def run(self) -> Dict[str, Any]:
        """Run complete fair comparison experiment."""
        logger.info("=" * 60)
        logger.info("FAIR COMPARISON EXPERIMENT")
        logger.info("All baselines use SAME models & calibration")
        logger.info("=" * 60)
        
        self.setup()
        self.run_training()
        self.run_evaluation()
        self.save_results()
        
        logger.info("\nExperiment complete!")
        return self.results


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="m5", choices=["m5", "gefcom"])
    parser.add_argument("--output", default="./results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-series", type=int, default=500)
    parser.add_argument("--quick", action="store_true")
    
    args = parser.parse_args()
    
    config = FairExperimentConfig(
        dataset=args.dataset,
        output_dir=args.output,
        seed=args.seed,
        m5_max_series=args.max_series if not args.quick else 100,
    )
    
    if args.quick:
        config.batch_size = 50
        config.log_interval = 20
    
    exp = FairComparisonExperiment(config)
    exp.run()
