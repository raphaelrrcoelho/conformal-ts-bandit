#!/usr/bin/env python
"""
Data Download and Setup Script.

Downloads M5 and GEFCom2014 datasets and verifies the setup.

Usage:
    python setup_data.py --dataset m5
    python setup_data.py --dataset gefcom
    python setup_data.py --all
"""

import argparse
import logging
from pathlib import Path
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials are configured."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        logger.error("Kaggle credentials not found!")
        logger.info("To set up Kaggle API:")
        logger.info("1. Go to kaggle.com/account")
        logger.info("2. Click 'Create New API Token'")
        logger.info("3. Save kaggle.json to ~/.kaggle/")
        logger.info("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    return True


def download_m5(data_dir: Path) -> bool:
    """Download M5 competition data from Kaggle."""
    logger.info("Downloading M5 dataset from Kaggle...")
    
    if not check_kaggle_credentials():
        logger.info("\nAlternative: Download manually from:")
        logger.info("https://www.kaggle.com/c/m5-forecasting-uncertainty/data")
        return False
    
    try:
        import kaggle
        
        data_dir.mkdir(parents=True, exist_ok=True)
        
        kaggle.api.competition_download_files(
            'm5-forecasting-uncertainty',
            path=str(data_dir),
            quiet=False
        )
        
        # Extract
        import zipfile
        zip_path = data_dir / "m5-forecasting-uncertainty.zip"
        if zip_path.exists():
            logger.info("Extracting files...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(data_dir)
            zip_path.unlink()
        
        logger.info(f"M5 data downloaded to {data_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def download_gefcom(data_dir: Path) -> bool:
    """
    Provide instructions for GEFCom2014 data.
    
    GEFCom data requires manual download due to licensing.
    """
    logger.info("GEFCom2014 Data Setup")
    logger.info("=" * 50)
    logger.info("\nThe GEFCom2014 dataset requires manual download:")
    logger.info("\n1. Solar Track:")
    logger.info("   - Source: CrowdAnalytix competition (may require account)")
    logger.info("   - Alternative: GitHub repositories with preprocessed data")
    logger.info("\n2. Wind Track:")
    logger.info("   - Source: CrowdAnalytix competition")
    logger.info("\n3. Preprocessed versions available at:")
    logger.info("   - https://github.com/camroach87/gefcom2014-solar-data")
    logger.info("   - https://github.com/energy-forecasting/gefcom-2014-data")
    logger.info("\nExpected file structure:")
    logger.info(f"  {data_dir}/solar/solar_power.csv")
    logger.info(f"  {data_dir}/solar/solar_weather.csv")
    logger.info(f"  {data_dir}/wind/wind_power.csv")
    logger.info(f"  {data_dir}/wind/wind_weather.csv")
    
    # Create directory structure
    (data_dir / "solar").mkdir(parents=True, exist_ok=True)
    (data_dir / "wind").mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nCreated directory structure at {data_dir}")
    logger.info("Please download the data and place files in these directories.")
    
    return True


def verify_m5_data(data_dir: Path) -> bool:
    """Verify M5 data files exist and are valid."""
    required_files = [
        "sales_train_evaluation.csv",
        "calendar.csv",
        "sell_prices.csv",
    ]
    
    missing = []
    for f in required_files:
        if not (data_dir / f).exists():
            # Try alternative name
            alt = f.replace("evaluation", "validation")
            if not (data_dir / alt).exists():
                missing.append(f)
    
    if missing:
        logger.error(f"Missing M5 files: {missing}")
        return False
    
    # Check file sizes
    import pandas as pd
    
    try:
        sales_path = data_dir / "sales_train_evaluation.csv"
        if not sales_path.exists():
            sales_path = data_dir / "sales_train_validation.csv"
        
        sales_df = pd.read_csv(sales_path, nrows=10)
        logger.info(f"Sales data columns: {len(sales_df.columns)}")
        
        calendar_df = pd.read_csv(data_dir / "calendar.csv", nrows=10)
        logger.info(f"Calendar data columns: {len(calendar_df.columns)}")
        
        prices_df = pd.read_csv(data_dir / "sell_prices.csv", nrows=10)
        logger.info(f"Prices data columns: {len(prices_df.columns)}")
        
        logger.info("M5 data verification passed!")
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def run_quick_test() -> bool:
    """Run a quick test to verify the setup."""
    logger.info("\nRunning quick sanity check...")
    
    try:
        # Test imports
        from conformal_ts.models.cts_agent import ConformalThompsonSampling, CTSConfig
        from conformal_ts.baselines.baselines import create_baselines
        from conformal_ts.evaluation.competition_metrics import interval_score
        
        import numpy as np
        
        # Create minimal agent
        config = CTSConfig(
            num_actions=4,
            feature_dim=10,
            seed=42
        )
        agent = ConformalThompsonSampling(config)
        
        # Test prediction
        context = np.random.randn(10)
        action, lower, upper = agent.select_and_predict(context)
        
        # Test update
        target = np.random.randn() * 10
        reward = agent.update(action, context, target)
        
        # Test baselines
        baselines = create_baselines(
            num_specifications=4,
            feature_dim=10,
            use_lightgbm=False  # Skip for quick test
        )
        
        # Test metrics
        scores = interval_score(
            np.array([lower]),
            np.array([upper]),
            np.array([target])
        )
        
        logger.info("All imports and basic functionality working!")
        logger.info(f"  Agent action: {action}")
        logger.info(f"  Interval: [{lower:.2f}, {upper:.2f}]")
        logger.info(f"  Score: {scores[0]:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Setup data for experiments")
    parser.add_argument(
        "--dataset", type=str, choices=["m5", "gefcom", "all"],
        default="all", help="Which dataset to download"
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data",
        help="Base directory for data"
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only verify existing data, don't download"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run quick test after setup"
    )
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    success = True
    
    if args.dataset in ["m5", "all"]:
        m5_dir = data_dir / "m5"
        
        if args.verify_only:
            success = verify_m5_data(m5_dir) and success
        else:
            if not verify_m5_data(m5_dir):
                success = download_m5(m5_dir) and success
            else:
                logger.info("M5 data already exists and is valid")
    
    if args.dataset in ["gefcom", "all"]:
        gefcom_dir = data_dir / "gefcom2014"
        download_gefcom(gefcom_dir)
    
    if args.test:
        success = run_quick_test() and success
    
    if success:
        logger.info("\n" + "=" * 50)
        logger.info("Setup complete!")
        logger.info("=" * 50)
        logger.info("\nTo run experiments:")
        logger.info("  # Quick test on simulated data:")
        logger.info("  python -m conformal_ts.experiments.run_full_experiment --quick")
        logger.info("")
        logger.info("  # Full M5 experiment:")
        logger.info("  python -m conformal_ts.experiments.run_full_experiment --dataset m5")
        logger.info("")
        logger.info("  # Full GEFCom experiment:")
        logger.info("  python -m conformal_ts.experiments.run_full_experiment --dataset gefcom")
    else:
        logger.error("\nSetup incomplete. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
