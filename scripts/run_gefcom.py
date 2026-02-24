#!/usr/bin/env python3
"""
GEFCom2014 Experiment Entry Point.

Runs Conformal Thompson Sampling on the GEFCom2014 energy forecasting
benchmark (solar or wind track).  Supports both the standard experiment
runner and the fair-comparison mode where all baselines share the same
LightGBM models and conformal calibration infrastructure.

Usage:
    # Solar track, default settings
    python scripts/run_gefcom.py --track solar

    # Wind track, fair comparison mode
    python scripts/run_gefcom.py --track wind --fair

    # Quick run with limited steps
    python scripts/run_gefcom.py --track solar --n-steps 200 --output ./results/quick

    # Set random seed
    python scripts/run_gefcom.py --track solar --seed 123
"""

import argparse
import logging
import os
import sys
from pathlib import Path

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
logger = logging.getLogger("run_gefcom")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Conformal Thompson Sampling on GEFCom2014 data."
    )
    parser.add_argument(
        "--track",
        type=str,
        default="solar",
        choices=["solar", "wind"],
        help="GEFCom2014 track to use (default: solar)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/gefcom",
        help="Output directory for results (default: ./results/gefcom)",
    )
    parser.add_argument(
        "--fair",
        action="store_true",
        help="Use fair comparison mode (shared LightGBM + conformal infrastructure)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Number of evaluation steps (default: None = all available)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def run_fair_experiment(args):
    """Run the fair-comparison experiment on GEFCom2014 data."""
    from conformal_ts.experiments.run_fair_experiment import (
        FairComparisonExperiment,
        FairExperimentConfig,
    )
    from conformal_ts.config import GEFCom2014Config

    gefcom_defaults = GEFCom2014Config(track=args.track)

    config = FairExperimentConfig(
        dataset="gefcom",
        gefcom_track=args.track,
        lookback_windows=gefcom_defaults.lookback_windows,
        forecast_horizons=list(gefcom_defaults.forecast_horizons)
        if hasattr(gefcom_defaults.forecast_horizons, "__iter__")
        else [24, 48, 72],
        output_dir=args.output,
        seed=args.seed,
    )

    if args.n_steps is not None:
        # Limit the batch/log interval to keep the quick run short
        config.batch_size = min(config.batch_size, 50)
        config.log_interval = max(1, args.n_steps // 10)

    experiment = FairComparisonExperiment(config)

    # If n_steps is given, monkey-patch the dataset after setup
    # to truncate training/evaluation ranges.
    experiment.setup()

    if args.n_steps is not None:
        _truncate_dataset(experiment, args.n_steps)

    experiment.run_training()
    experiment.run_evaluation()
    experiment.save_results()

    logger.info("Fair GEFCom experiment complete.")
    return experiment.results


def run_standard_experiment(args):
    """Run the standard (non-fair) experiment on GEFCom2014 data."""
    from conformal_ts.experiments.run_full_experiment import (
        FullExperimentRunner,
        ExperimentConfig,
    )
    from conformal_ts.config import GEFCom2014Config

    gefcom_defaults = GEFCom2014Config(track=args.track)

    config = ExperimentConfig(
        dataset="gefcom",
        gefcom_track=args.track,
        lookback_windows=gefcom_defaults.lookback_windows,
        forecast_horizons=list(gefcom_defaults.forecast_horizons)
        if hasattr(gefcom_defaults.forecast_horizons, "__iter__")
        else [24, 48, 72],
        output_dir=args.output,
        seed=args.seed,
    )

    if args.n_steps is not None:
        config.batch_size = min(config.batch_size, 50)
        config.log_interval = max(1, args.n_steps // 10)

    runner = FullExperimentRunner(config)
    runner.setup()

    if args.n_steps is not None:
        _truncate_runner_dataset(runner, args.n_steps)

    results = runner.run_training()
    runner.run_evaluation()
    runner.save_results()

    logger.info("Standard GEFCom experiment complete.")
    return runner.results


# ---------------------------------------------------------------------------
# Helpers to limit the number of evaluation steps
# ---------------------------------------------------------------------------

def _truncate_dataset(experiment, n_steps):
    """Truncate the FairComparisonExperiment dataset to at most n_steps days."""
    ds = experiment.dataset

    # The fair experiment uses keys 'train_end', 'val_end', 'num_days'
    if "num_days" in ds:
        max_lb = max(experiment.config.lookback_windows)
        max_h = max(experiment.config.forecast_horizons)
        # Keep at least max_lb + n_steps + max_h days
        required = max_lb + n_steps + max_h + 10
        if ds["num_days"] > required:
            ds["num_days"] = required
            ds["sales_matrix"] = ds["sales_matrix"][:, :required]
            ds["train_end"] = min(ds.get("train_end", int(required * 0.7)), int(required * 0.7))
            ds["val_end"] = min(ds.get("val_end", int(required * 0.85)), int(required * 0.85))

    logger.info(f"Truncated dataset to ~{n_steps} evaluation steps.")


def _truncate_runner_dataset(runner, n_steps):
    """Truncate the FullExperimentRunner dataset."""
    ds = runner.dataset
    if ds is None:
        return

    if "sales_matrix" in ds:
        total_days = ds["sales_matrix"].shape[1]
        max_lb = max(runner.config.lookback_windows)
        max_h = max(runner.config.forecast_horizons)
        required = max_lb + n_steps + max_h + 10
        if total_days > required:
            ds["sales_matrix"] = ds["sales_matrix"][:, :required]

    logger.info(f"Truncated runner dataset to ~{n_steps} evaluation steps.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    np.random.seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("GEFCom2014 Experiment")
    logger.info("=" * 60)
    logger.info(f"  Track  : {args.track}")
    logger.info(f"  Fair   : {args.fair}")
    logger.info(f"  Steps  : {args.n_steps or 'all'}")
    logger.info(f"  Seed   : {args.seed}")
    logger.info(f"  Output : {args.output}")
    logger.info("=" * 60)

    try:
        if args.fair:
            results = run_fair_experiment(args)
        else:
            results = run_standard_experiment(args)
    except Exception:
        logger.exception("Experiment failed")
        sys.exit(1)

    logger.info("Done. Results written to %s", args.output)


if __name__ == "__main__":
    main()
