"""
M5 Competition Dataset Loader.

The M5 competition dataset contains Walmart sales data with:
- 42,840 hierarchical time series
- 12-level hierarchy (item/dept/cat × store/state)
- 1,913 days of data (Jan 2011 - June 2016)
- Rich covariates (prices, calendar, SNAP)

Download from: https://www.kaggle.com/c/m5-forecasting-accuracy

This loader implements:
- Flexible aggregation levels
- Feature engineering for specification selection
- Train/val/test splits respecting temporal order
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Union
from dataclasses import dataclass
import warnings


@dataclass
class M5DataConfig:
    """Configuration for M5 data loading."""
    
    # Paths (relative to data_dir)
    sales_file: str = "sales_train_evaluation.csv"
    calendar_file: str = "calendar.csv"
    prices_file: str = "sell_prices.csv"
    
    # Aggregation level
    # Options: 'item', 'dept', 'cat', 'store', 'state', 'total', 
    #          'item_store', 'dept_store', 'cat_store', etc.
    aggregation_level: str = "store"
    
    # Sample size (None = use all)
    sample_size: Optional[int] = None
    
    # Time periods (in days)
    train_end_day: int = 1800  # Day number where training ends
    val_end_day: int = 1885    # Day number where validation ends
    # Test goes from val_end_day to end of data
    
    # Feature engineering
    use_price_features: bool = True
    use_calendar_features: bool = True
    use_snap_features: bool = True
    
    # Lookback windows for features (in days)
    lookback_windows: List[int] = None
    
    def __post_init__(self):
        if self.lookback_windows is None:
            self.lookback_windows = [7, 14, 28, 56]


class M5DataLoader:
    """
    Load and preprocess M5 competition data.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        config: Optional[M5DataConfig] = None
    ):
        """
        Initialize M5 data loader.
        
        Args:
            data_dir: Directory containing M5 data files
            config: Data loading configuration
        """
        self.data_dir = Path(data_dir)
        self.config = config or M5DataConfig()
        
        self._sales_df: Optional[pd.DataFrame] = None
        self._calendar_df: Optional[pd.DataFrame] = None
        self._prices_df: Optional[pd.DataFrame] = None
        
        self._aggregated_sales: Optional[pd.DataFrame] = None
        self._features: Optional[pd.DataFrame] = None
    
    def load_raw_data(self) -> bool:
        """
        Load raw M5 data files.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load sales data
            sales_path = self.data_dir / self.config.sales_file
            if sales_path.exists():
                self._sales_df = pd.read_csv(sales_path)
            else:
                warnings.warn(f"Sales file not found: {sales_path}")
                return False
            
            # Load calendar
            calendar_path = self.data_dir / self.config.calendar_file
            if calendar_path.exists():
                self._calendar_df = pd.read_csv(calendar_path)
            else:
                warnings.warn(f"Calendar file not found: {calendar_path}")
            
            # Load prices
            prices_path = self.data_dir / self.config.prices_file
            if prices_path.exists():
                self._prices_df = pd.read_csv(prices_path)
            else:
                warnings.warn(f"Prices file not found: {prices_path}")
            
            return True
            
        except Exception as e:
            warnings.warn(f"Error loading M5 data: {e}")
            return False
    
    def aggregate_sales(self) -> pd.DataFrame:
        """
        Aggregate sales to specified level.
        
        Returns:
            Aggregated sales DataFrame (series × days)
        """
        if self._sales_df is None:
            raise ValueError("Must call load_raw_data() first")
        
        # Get day columns
        day_cols = [c for c in self._sales_df.columns if c.startswith('d_')]
        
        # Define aggregation keys based on level
        level = self.config.aggregation_level
        
        if level == 'total':
            # Sum all series
            agg = self._sales_df[day_cols].sum().to_frame().T
            agg['series_id'] = 'total'
            
        elif level == 'state':
            agg = self._sales_df.groupby('state_id')[day_cols].sum()
            agg['series_id'] = agg.index
            
        elif level == 'store':
            agg = self._sales_df.groupby('store_id')[day_cols].sum()
            agg['series_id'] = agg.index
            
        elif level == 'cat':
            agg = self._sales_df.groupby('cat_id')[day_cols].sum()
            agg['series_id'] = agg.index
            
        elif level == 'dept':
            agg = self._sales_df.groupby('dept_id')[day_cols].sum()
            agg['series_id'] = agg.index
            
        elif level == 'item':
            # Keep item level but aggregate across stores
            agg = self._sales_df.groupby('item_id')[day_cols].sum()
            agg['series_id'] = agg.index
            
        elif level == 'store_cat':
            agg = self._sales_df.groupby(['store_id', 'cat_id'])[day_cols].sum()
            agg['series_id'] = [f"{s}_{c}" for s, c in agg.index]
            
        elif level == 'store_dept':
            agg = self._sales_df.groupby(['store_id', 'dept_id'])[day_cols].sum()
            agg['series_id'] = [f"{s}_{d}" for s, d in agg.index]
            
        elif level == 'item_store':
            # Keep original item-store granularity
            agg = self._sales_df.copy()
            agg['series_id'] = agg['id']
            
        else:
            raise ValueError(f"Unknown aggregation level: {level}")
        
        agg = agg.reset_index(drop=True)
        
        # Sample if requested
        if self.config.sample_size is not None:
            if len(agg) > self.config.sample_size:
                agg = agg.sample(n=self.config.sample_size, random_state=42)
        
        self._aggregated_sales = agg
        return agg
    
    def compute_features(
        self,
        day: int,
        series_sales: np.ndarray
    ) -> np.ndarray:
        """
        Compute context features for a given day.
        
        Args:
            day: Day number (1-indexed as in d_1, d_2, etc.)
            series_sales: Historical sales up to day-1 (shape: num_series × days)
        
        Returns:
            Feature matrix (num_series, feature_dim)
        """
        num_series = series_sales.shape[0]
        features_list = []
        
        # Bias term
        bias = np.ones((num_series, 1))
        features_list.append(bias)
        
        # Rolling statistics for each lookback window
        for window in self.config.lookback_windows:
            if series_sales.shape[1] >= window:
                recent = series_sales[:, -window:]
                
                # Mean
                mean = np.mean(recent, axis=1, keepdims=True)
                features_list.append(mean)
                
                # Std (with small epsilon for stability)
                std = np.std(recent, axis=1, keepdims=True) + 1e-6
                features_list.append(std)
                
                # Trend (simple linear fit slope)
                x = np.arange(window)
                x_centered = x - x.mean()
                slopes = np.sum(recent * x_centered, axis=1) / np.sum(x_centered**2)
                features_list.append(slopes.reshape(-1, 1))
            else:
                # Not enough history - use zeros
                features_list.append(np.zeros((num_series, 3)))
        
        # Calendar features (if available)
        if self.config.use_calendar_features and self._calendar_df is not None:
            day_str = f'd_{day}'
            cal_row = self._calendar_df[self._calendar_df['d'] == day_str]
            
            if len(cal_row) > 0:
                cal_row = cal_row.iloc[0]
                
                # Day of week (one-hot would be better, but keep simple)
                dow = cal_row.get('wday', 0)
                features_list.append(
                    np.full((num_series, 1), dow / 7.0)
                )
                
                # Month
                month = cal_row.get('month', 0)
                features_list.append(
                    np.full((num_series, 1), month / 12.0)
                )
                
                # Event indicators
                event1 = 1.0 if pd.notna(cal_row.get('event_name_1', None)) else 0.0
                event2 = 1.0 if pd.notna(cal_row.get('event_name_2', None)) else 0.0
                features_list.append(
                    np.full((num_series, 1), event1)
                )
                features_list.append(
                    np.full((num_series, 1), event2)
                )
        
        # SNAP features (if available)
        if self.config.use_snap_features and self._calendar_df is not None:
            day_str = f'd_{day}'
            cal_row = self._calendar_df[self._calendar_df['d'] == day_str]
            
            if len(cal_row) > 0:
                cal_row = cal_row.iloc[0]
                
                # Average SNAP across states
                snap_cols = ['snap_CA', 'snap_TX', 'snap_WI']
                snap_vals = [cal_row.get(c, 0) for c in snap_cols if c in cal_row]
                snap_avg = np.mean(snap_vals) if snap_vals else 0.0
                features_list.append(
                    np.full((num_series, 1), snap_avg)
                )
        
        # Concatenate all features
        features = np.concatenate(features_list, axis=1)
        
        return features
    
    def prepare_dataset(
        self,
        forecast_horizon: int = 28
    ) -> Dict[str, Any]:
        """
        Prepare full dataset for training.
        
        Args:
            forecast_horizon: Forecast horizon in days
        
        Returns:
            Dataset dictionary with train/val/test splits
        """
        if self._aggregated_sales is None:
            self.aggregate_sales()
        
        # Get day columns and sales matrix
        day_cols = [c for c in self._aggregated_sales.columns if c.startswith('d_')]
        day_nums = [int(c.split('_')[1]) for c in day_cols]
        
        series_ids = self._aggregated_sales['series_id'].values
        sales_matrix = self._aggregated_sales[day_cols].values.astype(np.float32)
        
        # Determine split points
        train_end = min(self.config.train_end_day, max(day_nums) - forecast_horizon)
        val_end = min(self.config.val_end_day, max(day_nums) - forecast_horizon)
        
        # Create dataset
        dataset = {
            'series_ids': series_ids,
            'day_columns': day_cols,
            'sales_matrix': sales_matrix,
            'num_series': len(series_ids),
            'num_days': len(day_cols),
            'train_end_day': train_end,
            'val_end_day': val_end,
            'forecast_horizon': forecast_horizon,
            'lookback_windows': self.config.lookback_windows,
        }
        
        return dataset
    
    def generate_training_examples(
        self,
        dataset: Dict[str, Any],
        lookback: int,
        horizon: int,
        split: str = 'train'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate training examples for a specific specification.
        
        Args:
            dataset: Prepared dataset dictionary
            lookback: Lookback window (specification parameter)
            horizon: Forecast horizon (specification parameter)
            split: 'train', 'val', or 'test'
        
        Returns:
            (contexts, targets, target_indices) arrays
        """
        sales = dataset['sales_matrix']
        
        # Determine day range for split
        if split == 'train':
            start_day = max(self.config.lookback_windows) + 1
            end_day = dataset['train_end_day']
        elif split == 'val':
            start_day = dataset['train_end_day'] + 1
            end_day = dataset['val_end_day']
        else:  # test
            start_day = dataset['val_end_day'] + 1
            end_day = dataset['num_days'] - horizon
        
        contexts = []
        targets = []
        indices = []
        
        for day in range(start_day, end_day + 1):
            # Get historical sales up to day-1
            history = sales[:, :day-1]
            
            # Compute features
            ctx = self.compute_features(day, history)
            contexts.append(ctx)
            
            # Target: sum of sales over forecast horizon
            target_start = day - 1  # 0-indexed
            target_end = min(target_start + horizon, sales.shape[1])
            target = np.sum(sales[:, target_start:target_end], axis=1)
            targets.append(target)
            
            indices.append(day)
        
        return (
            np.array(contexts),   # (num_days, num_series, feature_dim)
            np.array(targets),    # (num_days, num_series)
            np.array(indices)     # (num_days,)
        )


def create_m5_simulation_data(
    num_series: int = 100,
    num_days: int = 500,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Create simulated M5-like data for testing when real data unavailable.
    
    Simulates intermittent demand patterns typical of retail data.
    
    Args:
        num_series: Number of time series
        num_days: Number of days
        seed: Random seed
    
    Returns:
        Simulated dataset dictionary
    """
    rng = np.random.default_rng(seed)
    
    # Generate base demand patterns
    # Mix of different demand profiles
    profiles = []
    
    for i in range(num_series):
        # Random base level
        base_level = rng.exponential(50)
        
        # Trend
        trend = rng.uniform(-0.01, 0.02)
        
        # Seasonality (weekly)
        weekly_pattern = rng.dirichlet(np.ones(7) * 2) * 7
        
        # Generate series
        series = np.zeros(num_days)
        for t in range(num_days):
            # Base + trend
            level = base_level * (1 + trend * t)
            
            # Weekly seasonality
            dow = t % 7
            level *= weekly_pattern[dow]
            
            # Random events (promotions, etc.)
            if rng.random() < 0.05:
                level *= rng.uniform(1.5, 3.0)
            
            # Intermittent demand (zeros)
            if rng.random() < 0.1:
                level = 0
            
            # Add noise
            if level > 0:
                series[t] = max(0, rng.poisson(level))
            else:
                series[t] = 0
        
        profiles.append(series)
    
    sales_matrix = np.array(profiles, dtype=np.float32)
    
    # Create calendar-like features
    calendar = {
        'day': np.arange(1, num_days + 1),
        'dow': np.arange(num_days) % 7,
        'month': (np.arange(num_days) // 30) % 12 + 1,
        'event': rng.random(num_days) < 0.1,
    }
    
    return {
        'series_ids': [f'series_{i}' for i in range(num_series)],
        'sales_matrix': sales_matrix,
        'calendar': calendar,
        'num_series': num_series,
        'num_days': num_days,
        'train_end_day': int(num_days * 0.7),
        'val_end_day': int(num_days * 0.85),
        'lookback_windows': [7, 14, 28, 56],
    }


class M5SpecificationSelector:
    """
    Wrapper for using M5 data with Conformal Thompson Sampling.
    
    Specifications represent different (lookback, horizon) combinations.
    """
    
    def __init__(
        self,
        dataset: Dict[str, Any],
        lookback_windows: List[int] = None,
        forecast_horizons: List[int] = None
    ):
        """
        Initialize specification selector.
        
        Args:
            dataset: Prepared M5 dataset
            lookback_windows: Lookback windows for specifications
            forecast_horizons: Forecast horizons for specifications
        """
        self.dataset = dataset
        self.lookback_windows = lookback_windows or [7, 14, 28, 56]
        self.forecast_horizons = forecast_horizons or [7, 14, 28]
        
        # Build specification space
        self.specifications = [
            (lb, fh) 
            for lb in self.lookback_windows 
            for fh in self.forecast_horizons
        ]
        self.num_specifications = len(self.specifications)
        
        # Current position
        self.current_day = max(self.lookback_windows) + 1
        self.sales = dataset['sales_matrix']
    
    def get_context(self, series_idx: int) -> np.ndarray:
        """Get context features for current state."""
        history = self.sales[series_idx, :self.current_day-1]
        
        features = [1.0]  # Bias
        
        for window in self.lookback_windows:
            if len(history) >= window:
                recent = history[-window:]
                features.extend([
                    np.mean(recent),
                    np.std(recent) + 1e-6,
                    np.sum(recent > 0) / window,  # Non-zero rate
                ])
            else:
                features.extend([0.0, 1.0, 0.5])
        
        return np.array(features)
    
    def get_target(
        self,
        series_idx: int,
        specification: int
    ) -> Tuple[float, int]:
        """
        Get target value for specification.
        
        Args:
            series_idx: Series index
            specification: Specification index
        
        Returns:
            (target_value, horizon)
        """
        lookback, horizon = self.specifications[specification]
        
        target_start = self.current_day - 1  # 0-indexed
        target_end = min(target_start + horizon, self.sales.shape[1])
        target = np.sum(self.sales[series_idx, target_start:target_end])
        
        return float(target), horizon
    
    def step(self):
        """Advance to next day."""
        self.current_day += 1
    
    def reset(self, day: Optional[int] = None):
        """Reset to starting position."""
        if day is not None:
            self.current_day = day
        else:
            self.current_day = max(self.lookback_windows) + 1


if __name__ == "__main__":
    # Test with simulated data
    print("Testing M5 data utilities with simulated data...")
    
    # Create simulated dataset
    sim_data = create_m5_simulation_data(
        num_series=50,
        num_days=300,
        seed=42
    )
    
    print(f"Simulated dataset:")
    print(f"  Series: {sim_data['num_series']}")
    print(f"  Days: {sim_data['num_days']}")
    print(f"  Sales shape: {sim_data['sales_matrix'].shape}")
    print(f"  Mean daily sales: {sim_data['sales_matrix'].mean():.2f}")
    
    # Test specification selector
    selector = M5SpecificationSelector(sim_data)
    print(f"\nSpecification selector:")
    print(f"  Num specifications: {selector.num_specifications}")
    print(f"  Specifications: {selector.specifications[:4]}...")
    
    # Get context for first series
    ctx = selector.get_context(0)
    print(f"\nContext for series 0:")
    print(f"  Shape: {ctx.shape}")
    print(f"  Values: {ctx[:5]}...")
    
    # Get target for first specification
    target, horizon = selector.get_target(0, 0)
    print(f"\nTarget for spec 0:")
    print(f"  Value: {target:.2f}")
    print(f"  Horizon: {horizon}")
