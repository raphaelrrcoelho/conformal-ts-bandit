"""
Real M5 Competition Dataset Loader.

Downloads and processes the actual M5 Kaggle competition data:
- 30,490 product-store time series (bottom level)
- 1,941 days of sales (2011-01-29 to 2016-06-19)
- Rich covariates: prices, calendar events, SNAP

The M5 Uncertainty track evaluated probabilistic forecasts using
Weighted Scaled Pinball Loss (WSPL) across 9 quantiles.

Usage:
    loader = M5RealDataLoader("./data/m5")
    loader.download()  # Requires Kaggle API credentials
    dataset = loader.prepare_dataset()
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
import warnings
import pickle
import json
from datetime import datetime

logger = logging.getLogger(__name__)


# M5 Uncertainty track quantile levels
M5_QUANTILES = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]

# Aggregation level hierarchy
AGGREGATION_LEVELS = {
    1: ['total'],                           # Total (1 series)
    2: ['state_id'],                        # State (3 series)
    3: ['store_id'],                        # Store (10 series)
    4: ['cat_id'],                          # Category (3 series)
    5: ['dept_id'],                         # Department (7 series)
    6: ['state_id', 'cat_id'],              # State-Category (9 series)
    7: ['state_id', 'dept_id'],             # State-Department (21 series)
    8: ['store_id', 'cat_id'],              # Store-Category (30 series)
    9: ['store_id', 'dept_id'],             # Store-Department (70 series)
    10: ['item_id'],                        # Item (3,049 series)
    11: ['item_id', 'state_id'],            # Item-State (9,147 series)
    12: ['item_id', 'store_id'],            # Item-Store (30,490 series)
}


@dataclass
class M5Config:
    """Configuration for M5 data processing."""
    
    # Data paths
    data_dir: str = "./data/m5"
    
    # Which aggregation levels to use (1-12, or list)
    # For paper: recommend levels 8-10 for tractability
    aggregation_levels: List[int] = field(default_factory=lambda: [8, 9, 10])
    
    # Sampling for faster iteration (None = use all)
    max_series_per_level: Optional[int] = None
    
    # Time splits (day indices, 1-indexed like M5)
    # Training: d_1 to d_1913 (but we use 1-1885 for train)
    # The last 28 days were the test period in the competition
    train_end_day: int = 1857      # ~End of May 2016
    val_end_day: int = 1885        # ~Mid June 2016
    test_end_day: int = 1913       # End of data
    
    # Specification space
    lookback_windows: List[int] = field(default_factory=lambda: [7, 14, 28, 56])
    forecast_horizons: List[int] = field(default_factory=lambda: [7, 14, 28])
    
    # Feature engineering
    use_calendar_features: bool = True
    use_price_features: bool = True
    use_snap_features: bool = True
    use_lag_features: bool = True
    use_rolling_features: bool = True
    
    # Scaling
    scale_targets: bool = True
    
    # Cache
    use_cache: bool = True
    cache_dir: str = "./cache/m5"


class M5RealDataLoader:
    """
    Load and process real M5 competition data.
    
    Handles the full pipeline:
    1. Download from Kaggle (requires API credentials)
    2. Load and merge sales, calendar, prices
    3. Aggregate to desired hierarchy levels
    4. Generate features for bandit context
    5. Create train/val/test splits
    """
    
    def __init__(self, config: Optional[M5Config] = None):
        self.config = config or M5Config()
        self.data_dir = Path(self.config.data_dir)
        self.cache_dir = Path(self.config.cache_dir)
        
        # Raw data
        self._sales_df: Optional[pd.DataFrame] = None
        self._calendar_df: Optional[pd.DataFrame] = None
        self._prices_df: Optional[pd.DataFrame] = None
        
        # Processed data
        self._processed: Dict[int, pd.DataFrame] = {}  # By aggregation level
        
    def download(self, force: bool = False) -> bool:
        """
        Download M5 data from Kaggle.
        
        Requires:
            - kaggle package installed: pip install kaggle
            - API credentials in ~/.kaggle/kaggle.json
        
        Args:
            force: Re-download even if files exist
            
        Returns:
            True if successful
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        required_files = [
            "sales_train_evaluation.csv",
            "calendar.csv", 
            "sell_prices.csv"
        ]
        
        # Check if already downloaded
        if not force and all((self.data_dir / f).exists() for f in required_files):
            logger.info("M5 data already downloaded")
            return True
        
        try:
            import kaggle
            
            logger.info("Downloading M5 data from Kaggle...")
            kaggle.api.competition_download_files(
                'm5-forecasting-uncertainty',
                path=str(self.data_dir),
                quiet=False
            )
            
            # Extract if zipped
            import zipfile
            zip_path = self.data_dir / "m5-forecasting-uncertainty.zip"
            if zip_path.exists():
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(self.data_dir)
                zip_path.unlink()
            
            logger.info("M5 data downloaded successfully")
            return True
            
        except ImportError:
            logger.error("kaggle package not installed. Run: pip install kaggle")
            return False
        except Exception as e:
            logger.error(f"Failed to download M5 data: {e}")
            logger.info("Please download manually from: "
                       "https://www.kaggle.com/c/m5-forecasting-uncertainty/data")
            return False
    
    def load_raw_data(self) -> bool:
        """Load raw CSV files into DataFrames."""
        try:
            logger.info("Loading raw M5 data...")
            
            # Sales data
            sales_path = self.data_dir / "sales_train_evaluation.csv"
            if not sales_path.exists():
                # Try alternative name
                sales_path = self.data_dir / "sales_train_validation.csv"
            
            self._sales_df = pd.read_csv(sales_path)
            logger.info(f"Loaded sales: {self._sales_df.shape}")
            
            # Calendar
            self._calendar_df = pd.read_csv(self.data_dir / "calendar.csv")
            logger.info(f"Loaded calendar: {self._calendar_df.shape}")
            
            # Prices
            self._prices_df = pd.read_csv(self.data_dir / "sell_prices.csv")
            logger.info(f"Loaded prices: {self._prices_df.shape}")
            
            return True
            
        except FileNotFoundError as e:
            logger.error(f"M5 data files not found: {e}")
            logger.info("Run loader.download() first or download manually")
            return False
    
    def _get_day_columns(self) -> List[str]:
        """Get list of day columns (d_1, d_2, ..., d_1913)."""
        return [c for c in self._sales_df.columns if c.startswith('d_')]
    
    def aggregate_to_level(self, level: int) -> pd.DataFrame:
        """
        Aggregate sales to specified hierarchy level.
        
        Args:
            level: Aggregation level (1-12)
            
        Returns:
            Aggregated DataFrame with series_id and day columns
        """
        if self._sales_df is None:
            self.load_raw_data()
        
        day_cols = self._get_day_columns()
        
        if level == 1:
            # Total: sum all series
            agg = self._sales_df[day_cols].sum().to_frame().T
            agg['series_id'] = 'Total'
        else:
            group_cols = AGGREGATION_LEVELS[level]
            agg = self._sales_df.groupby(group_cols)[day_cols].sum().reset_index()
            agg['series_id'] = agg[group_cols].astype(str).agg('_'.join, axis=1)
        
        # Reorder columns
        cols = ['series_id'] + day_cols
        agg = agg[cols]
        
        # Optional sampling
        if self.config.max_series_per_level is not None:
            if len(agg) > self.config.max_series_per_level:
                agg = agg.sample(n=self.config.max_series_per_level, random_state=42)
        
        return agg
    
    def compute_scaling_factors(
        self, 
        sales_df: pd.DataFrame,
        train_end_day: int
    ) -> pd.DataFrame:
        """
        Compute scaling factors for WSPL (average absolute differences).
        
        The M5 uncertainty metric scales by MAD of training data.
        """
        day_cols = [c for c in sales_df.columns if c.startswith('d_')]
        train_cols = [c for c in day_cols if int(c.split('_')[1]) <= train_end_day]
        
        # Get training sales matrix
        train_sales = sales_df[train_cols].values
        
        # Compute mean absolute differences (for scaling)
        # MAD = mean(|y_t - y_{t-1}|) over training period
        diffs = np.abs(np.diff(train_sales, axis=1))
        mad = np.mean(diffs, axis=1)
        mad = np.maximum(mad, 1.0)  # Avoid division by zero
        
        scale_df = pd.DataFrame({
            'series_id': sales_df['series_id'],
            'scale': mad
        })
        
        return scale_df
    
    def build_calendar_features(self) -> pd.DataFrame:
        """Build calendar feature matrix."""
        cal = self._calendar_df.copy()
        
        # Basic features
        cal['day_of_week'] = pd.to_datetime(cal['date']).dt.dayofweek
        cal['day_of_month'] = pd.to_datetime(cal['date']).dt.day
        cal['month'] = pd.to_datetime(cal['date']).dt.month
        cal['year'] = pd.to_datetime(cal['date']).dt.year
        cal['week_of_year'] = pd.to_datetime(cal['date']).dt.isocalendar().week
        
        # Event indicators
        cal['has_event_1'] = cal['event_name_1'].notna().astype(int)
        cal['has_event_2'] = cal['event_name_2'].notna().astype(int)
        
        # Event type encoding
        event_types = ['Cultural', 'National', 'Religious', 'Sporting']
        for et in event_types:
            cal[f'event_{et.lower()}'] = (
                (cal['event_type_1'] == et) | (cal['event_type_2'] == et)
            ).astype(int)
        
        # SNAP indicators (average across states)
        cal['snap_avg'] = cal[['snap_CA', 'snap_TX', 'snap_WI']].mean(axis=1)
        
        return cal
    
    def compute_context_features(
        self,
        sales_matrix: np.ndarray,
        day_idx: int,
        calendar_features: pd.DataFrame,
        lookback_windows: List[int]
    ) -> np.ndarray:
        """
        Compute context features for bandit at given day.
        
        Features include:
        - Sales statistics over multiple lookback windows
        - Calendar features (day of week, events, etc.)
        - Trend indicators
        
        Args:
            sales_matrix: (num_series, num_days) array
            day_idx: Current day index (0-indexed)
            calendar_features: Calendar DataFrame
            lookback_windows: List of lookback periods
            
        Returns:
            (num_series, feature_dim) array
        """
        num_series = sales_matrix.shape[0]
        features_list = []
        
        # Bias term
        features_list.append(np.ones((num_series, 1)))
        
        # Sales features for each lookback window
        for window in lookback_windows:
            start_idx = max(0, day_idx - window)
            window_sales = sales_matrix[:, start_idx:day_idx]
            
            if window_sales.shape[1] > 0:
                # Mean
                features_list.append(
                    np.mean(window_sales, axis=1, keepdims=True)
                )
                # Std
                features_list.append(
                    np.std(window_sales, axis=1, keepdims=True) + 1e-6
                )
                # Non-zero rate (important for intermittent demand)
                features_list.append(
                    np.mean(window_sales > 0, axis=1, keepdims=True)
                )
                # Max
                features_list.append(
                    np.max(window_sales, axis=1, keepdims=True)
                )
                # Trend (last week vs previous week)
                if window >= 14 and window_sales.shape[1] >= 14:
                    recent = np.mean(window_sales[:, -7:], axis=1, keepdims=True)
                    previous = np.mean(window_sales[:, -14:-7], axis=1, keepdims=True)
                    trend = (recent - previous) / (previous + 1e-6)
                    features_list.append(trend)
                else:
                    features_list.append(np.zeros((num_series, 1)))
            else:
                # Not enough history - use zeros
                features_list.append(np.zeros((num_series, 5)))
        
        # Calendar features (same for all series at this day)
        if self.config.use_calendar_features and day_idx < len(calendar_features):
            cal_row = calendar_features.iloc[day_idx]
            
            # Day of week (one-hot would be better, but keep simple)
            features_list.append(
                np.full((num_series, 1), cal_row['day_of_week'] / 6.0)
            )
            # Month
            features_list.append(
                np.full((num_series, 1), cal_row['month'] / 12.0)
            )
            # Events
            features_list.append(
                np.full((num_series, 1), cal_row['has_event_1'])
            )
            # SNAP
            features_list.append(
                np.full((num_series, 1), cal_row['snap_avg'])
            )
        
        return np.concatenate(features_list, axis=1).astype(np.float32)
    
    def prepare_dataset(
        self,
        aggregation_levels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Prepare full dataset for experiments.
        
        Args:
            aggregation_levels: Which levels to include (default from config)
            
        Returns:
            Dataset dictionary with all necessary components
        """
        # Check cache
        cache_path = self.cache_dir / "m5_dataset.pkl"
        if self.config.use_cache and cache_path.exists():
            logger.info(f"Loading cached dataset from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Load raw data
        if self._sales_df is None:
            if not self.load_raw_data():
                raise RuntimeError("Failed to load M5 data")
        
        levels = aggregation_levels or self.config.aggregation_levels
        
        # Aggregate each level
        logger.info(f"Aggregating sales to levels: {levels}")
        all_sales = []
        level_indices = []
        
        for level in levels:
            agg = self.aggregate_to_level(level)
            agg['agg_level'] = level
            all_sales.append(agg)
            level_indices.extend([level] * len(agg))
        
        combined = pd.concat(all_sales, ignore_index=True)
        
        # Get sales matrix
        day_cols = self._get_day_columns()
        series_ids = combined['series_id'].values
        sales_matrix = combined[day_cols].values.astype(np.float32)
        
        logger.info(f"Combined dataset: {sales_matrix.shape[0]} series, "
                   f"{sales_matrix.shape[1]} days")
        
        # Compute scaling factors
        scale_df = self.compute_scaling_factors(combined, self.config.train_end_day)
        
        # Build calendar features
        calendar_features = self.build_calendar_features()
        
        # Build specification space
        specifications = [
            (lb, fh)
            for lb in self.config.lookback_windows
            for fh in self.config.forecast_horizons
        ]
        
        dataset = {
            # Core data
            'series_ids': series_ids,
            'sales_matrix': sales_matrix,
            'scale_factors': scale_df['scale'].values,
            'aggregation_levels': np.array(level_indices),
            
            # Dimensions
            'num_series': len(series_ids),
            'num_days': sales_matrix.shape[1],
            
            # Time splits
            'train_end_day': self.config.train_end_day,
            'val_end_day': self.config.val_end_day,
            'test_end_day': self.config.test_end_day,
            
            # Specifications
            'specifications': specifications,
            'num_specifications': len(specifications),
            'lookback_windows': self.config.lookback_windows,
            'forecast_horizons': self.config.forecast_horizons,
            
            # Calendar
            'calendar_features': calendar_features,
            
            # Quantiles for M5 evaluation
            'quantiles': M5_QUANTILES,
            
            # Metadata
            'config': self.config.__dict__,
            'created_at': datetime.now().isoformat(),
        }
        
        # Cache
        if self.config.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(dataset, f)
            logger.info(f"Cached dataset to {cache_path}")
        
        return dataset


class M5ExperimentRunner:
    """
    Run Conformal Thompson Sampling experiments on M5 data.
    
    Handles the online learning loop:
    1. At each timestep, observe context (features)
    2. Agent selects specification
    3. Make forecast with selected specification
    4. Observe outcome after horizon days
    5. Update agent with reward (interval score)
    """
    
    def __init__(
        self,
        dataset: Dict[str, Any],
        agent,  # ConformalThompsonSampling
        baselines: Optional[Dict[str, Any]] = None,
        batch_size: int = 100,  # Series per batch
        checkpoint_interval: int = 50,  # Days between checkpoints
        checkpoint_dir: str = "./checkpoints/m5"
    ):
        self.dataset = dataset
        self.agent = agent
        self.baselines = baselines or {}
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # State
        self.current_day = max(dataset['lookback_windows']) + 1
        self.results_history = []
        
        # Pending updates (for deferred horizon outcomes)
        self.pending_updates: Dict[int, List[Dict]] = {}  # day -> list of updates
    
    def get_context_batch(
        self,
        day_idx: int,
        series_indices: np.ndarray
    ) -> np.ndarray:
        """Get context features for a batch of series at given day."""
        sales_matrix = self.dataset['sales_matrix'][series_indices]
        
        # Compute features
        features = []
        for i, series_idx in enumerate(series_indices):
            series_sales = self.dataset['sales_matrix'][series_idx:series_idx+1]
            feat = self._compute_series_features(series_sales, day_idx)
            features.append(feat)
        
        return np.vstack(features)
    
    def _compute_series_features(
        self,
        series_sales: np.ndarray,
        day_idx: int
    ) -> np.ndarray:
        """Compute features for a single series."""
        features = [1.0]  # Bias
        
        for window in self.dataset['lookback_windows']:
            start = max(0, day_idx - window)
            history = series_sales[0, start:day_idx]
            
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
        
        # Calendar features
        cal = self.dataset['calendar_features']
        if day_idx < len(cal):
            row = cal.iloc[day_idx]
            features.extend([
                row['day_of_week'] / 6.0,
                row['month'] / 12.0,
                float(row['has_event_1']),
                float(row['snap_avg']),
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def get_target(
        self,
        series_idx: int,
        day_idx: int,
        horizon: int
    ) -> float:
        """Get target value (sum of sales over horizon)."""
        sales = self.dataset['sales_matrix']
        target_end = min(day_idx + horizon, sales.shape[1])
        return float(np.sum(sales[series_idx, day_idx:target_end]))
    
    def run_training(
        self,
        start_day: Optional[int] = None,
        end_day: Optional[int] = None,
        log_interval: int = 10
    ) -> Dict[str, Any]:
        """
        Run the training loop.
        
        Args:
            start_day: Starting day (default: after warmup)
            end_day: Ending day (default: train_end_day)
            log_interval: Days between logging
            
        Returns:
            Training results dictionary
        """
        sales_matrix = self.dataset['sales_matrix']
        num_series = self.dataset['num_series']
        specifications = self.dataset['specifications']
        
        start_day = start_day or max(self.dataset['lookback_windows']) + 1
        end_day = end_day or self.dataset['train_end_day']
        max_horizon = max(self.dataset['forecast_horizons'])
        
        logger.info(f"Training from day {start_day} to {end_day}")
        logger.info(f"Series: {num_series}, Specifications: {len(specifications)}")
        
        # Tracking
        all_scores = []
        all_coverages = []
        all_actions = []
        
        for day in range(start_day, end_day - max_horizon):
            # Process pending updates from previous days
            self._process_pending_updates(day)
            
            # Sample batch of series
            batch_indices = np.random.choice(
                num_series, 
                size=min(self.batch_size, num_series),
                replace=False
            )
            
            for series_idx in batch_indices:
                # Get context
                context = self._compute_series_features(
                    sales_matrix[series_idx:series_idx+1],
                    day
                )
                
                # Agent selects action and predicts interval
                action, lower, upper = self.agent.select_and_predict(context)
                
                # Get specification details
                lookback, horizon = specifications[action]
                
                # Schedule deferred update
                outcome_day = day + horizon
                if outcome_day not in self.pending_updates:
                    self.pending_updates[outcome_day] = []
                
                self.pending_updates[outcome_day].append({
                    'series_idx': series_idx,
                    'day': day,
                    'action': action,
                    'context': context,
                    'lower': lower,
                    'upper': upper,
                    'horizon': horizon,
                })
                
                all_actions.append(action)
            
            # Logging
            if day % log_interval == 0:
                recent_scores = all_scores[-1000:] if all_scores else [0]
                recent_cov = all_coverages[-1000:] if all_coverages else [0]
                logger.info(
                    f"Day {day}/{end_day}: "
                    f"score={np.mean(recent_scores):.2f}, "
                    f"coverage={np.mean(recent_cov):.2%}, "
                    f"pending={sum(len(v) for v in self.pending_updates.values())}"
                )
            
            # Checkpoint
            if day % self.checkpoint_interval == 0:
                self._save_checkpoint(day)
        
        # Process remaining pending updates
        for future_day in sorted(self.pending_updates.keys()):
            self._process_pending_updates(future_day, force=True)
        
        return {
            'scores': all_scores,
            'coverages': all_coverages,
            'actions': all_actions,
            'agent_stats': self.agent.get_statistics(),
        }
    
    def _process_pending_updates(self, current_day: int, force: bool = False):
        """Process updates that have matured."""
        days_to_process = [d for d in self.pending_updates.keys() 
                         if d <= current_day or force]
        
        for day in days_to_process:
            updates = self.pending_updates.pop(day, [])
            
            for update in updates:
                # Get actual outcome
                target = self.get_target(
                    update['series_idx'],
                    update['day'],
                    update['horizon']
                )
                
                # Update agent
                self.agent.update(
                    update['action'],
                    update['context'],
                    target
                )
    
    def _save_checkpoint(self, day: int):
        """Save checkpoint."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'day': day,
            'agent_state': self.agent.get_statistics(),
            'pending_updates': len(self.pending_updates),
        }
        
        path = self.checkpoint_dir / f"checkpoint_day_{day}.json"
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
    
    def run_evaluation(
        self,
        start_day: Optional[int] = None,
        end_day: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run evaluation on validation/test set.
        
        Args:
            start_day: Start of evaluation period
            end_day: End of evaluation period
            
        Returns:
            Evaluation results with metrics
        """
        from .metrics import (
            interval_score, coverage_rate, 
            weighted_scaled_pinball_loss
        )
        
        start_day = start_day or self.dataset['val_end_day']
        end_day = end_day or self.dataset['test_end_day']
        max_horizon = max(self.dataset['forecast_horizons'])
        
        sales_matrix = self.dataset['sales_matrix']
        num_series = self.dataset['num_series']
        specifications = self.dataset['specifications']
        scale_factors = self.dataset['scale_factors']
        
        logger.info(f"Evaluating from day {start_day} to {end_day}")
        
        results = {
            'cts': {'scores': [], 'coverages': [], 'widths': [], 'actions': []},
        }
        
        # Add baseline tracking
        for name in self.baselines:
            results[name] = {'scores': [], 'coverages': [], 'widths': []}
        
        for day in range(start_day, end_day - max_horizon):
            for series_idx in range(min(self.batch_size, num_series)):
                context = self._compute_series_features(
                    sales_matrix[series_idx:series_idx+1],
                    day
                )
                
                # CTS prediction
                action, lower, upper = self.agent.select_and_predict(context)
                _, horizon = specifications[action]
                target = self.get_target(series_idx, day, horizon)
                
                score = interval_score(
                    np.array([lower]), 
                    np.array([upper]), 
                    np.array([target]),
                    alpha=0.10
                )[0]
                covered = lower <= target <= upper
                
                results['cts']['scores'].append(score)
                results['cts']['coverages'].append(covered)
                results['cts']['widths'].append(upper - lower)
                results['cts']['actions'].append(action)
                
                # Baselines
                for name, baseline in self.baselines.items():
                    b_lower, b_upper = baseline.predict(context)
                    b_score = interval_score(
                        np.array([b_lower]),
                        np.array([b_upper]),
                        np.array([target]),
                        alpha=0.10
                    )[0]
                    b_covered = b_lower <= target <= b_upper
                    
                    results[name]['scores'].append(b_score)
                    results[name]['coverages'].append(b_covered)
                    results[name]['widths'].append(b_upper - b_lower)
        
        # Compute summary statistics
        summary = {}
        for method, data in results.items():
            summary[method] = {
                'mean_score': float(np.mean(data['scores'])),
                'std_score': float(np.std(data['scores'])),
                'coverage': float(np.mean(data['coverages'])),
                'mean_width': float(np.mean(data['widths'])),
            }
            
            if method == 'cts':
                summary[method]['action_distribution'] = (
                    np.bincount(data['actions'], minlength=len(specifications)).tolist()
                )
        
        return {
            'detailed': results,
            'summary': summary,
        }


if __name__ == "__main__":
    # Test the loader
    logging.basicConfig(level=logging.INFO)
    
    config = M5Config(
        data_dir="./data/m5",
        aggregation_levels=[9, 10],  # Store-Dept and Item levels
        max_series_per_level=100,    # Sample for testing
    )
    
    loader = M5RealDataLoader(config)
    
    # Try to load data
    if loader.load_raw_data():
        dataset = loader.prepare_dataset()
        print(f"\nDataset prepared:")
        print(f"  Series: {dataset['num_series']}")
        print(f"  Days: {dataset['num_days']}")
        print(f"  Specifications: {dataset['num_specifications']}")
        print(f"  Train end: day {dataset['train_end_day']}")
    else:
        print("\nM5 data not found. Creating simulated data for testing...")
        from conformal_ts.data.m5_loader import create_m5_simulation_data
        dataset = create_m5_simulation_data(num_series=100, num_days=500)
        print(f"Simulated: {dataset['num_series']} series, {dataset['num_days']} days")
