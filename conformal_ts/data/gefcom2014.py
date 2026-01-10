"""
GEFCom2014 Energy Forecasting Dataset Loader.

The Global Energy Forecasting Competition 2014 includes 4 tracks:
- Load forecasting (electricity demand)
- Price forecasting (electricity prices)  
- Wind power forecasting
- Solar power forecasting

Each track requires probabilistic forecasts evaluated via pinball loss
across 99 quantiles (1%, 2%, ..., 99%).

This loader focuses on the Solar and Wind tracks which have the
most interesting specification selection problems (feature subsets,
lookback windows, spatial aggregation).

References:
    Hong et al. (2016) "Probabilistic energy forecasting: Global Energy
    Forecasting Competition 2014 and beyond", IJF

Data sources:
    - Competition data: https://www.crowdanalytix.com/contests/gefcom2014
    - Preprocessed: Various GitHub repositories
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
import pickle
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# GEFCom uses 99 quantiles
GEFCOM_QUANTILES = [q/100 for q in range(1, 100)]


@dataclass  
class GEFComConfig:
    """Configuration for GEFCom2014 data loading."""
    
    data_dir: str = "./conformal_ts/data/gefcom2014"
    
    # Track: 'solar', 'wind', 'load', 'price'
    track: str = "solar"
    
    # For solar/wind: which zones to include (None = all)
    zones: Optional[List[int]] = None
    
    # Time splits
    train_end: str = "2013-06-30"
    val_end: str = "2013-12-31"
    # Test is Jan-Jun 2014 (competition period)
    
    # Specification space
    lookback_hours: List[int] = field(default_factory=lambda: [24, 48, 168, 336])
    forecast_horizons: List[int] = field(default_factory=lambda: [24, 48, 72])
    
    # Feature selection options (for specification space)
    feature_subsets: List[str] = field(default_factory=lambda: [
        'minimal',      # Just lagged power
        'weather',      # Add weather vars  
        'full',         # All features
    ])
    
    # Frequency
    frequency: str = "H"  # Hourly
    
    # Cache
    use_cache: bool = True
    cache_dir: str = "./cache/gefcom"


class GEFCom2014Loader:
    """
    Load and process GEFCom2014 competition data.
    
    Handles:
    1. Loading raw data (assumes downloaded to data_dir)
    2. Feature engineering for different tracks
    3. Creating specification space for bandit
    4. Train/val/test splits
    """
    
    def __init__(self, config: Optional[GEFComConfig] = None):
        self.config = config or GEFComConfig()
        self.data_dir = Path(self.config.data_dir)
        self.cache_dir = Path(self.config.cache_dir)
        
        self._raw_data: Optional[pd.DataFrame] = None
        self._weather_data: Optional[pd.DataFrame] = None
        
    def load_solar_data(self) -> pd.DataFrame:
        """
        Load GEFCom2014 solar track data.
        
        Expected format:
        - solar_power.csv: timestamp, zone1, zone2, zone3 (power in MW)
        - solar_weather.csv: timestamp, zone, var1, var2, ... (weather vars)
        """
        power_path = self.data_dir / "solar" / "solar_power.csv"
        weather_path = self.data_dir / "solar" / "solar_weather.csv"
        
        if not power_path.exists():
            # Try alternative locations
            power_path = self.data_dir / "solar_power.csv"
            weather_path = self.data_dir / "solar_weather.csv"
        
        if not power_path.exists():
            raise FileNotFoundError(
                f"Solar data not found at {power_path}. "
                "Please download GEFCom2014 data."
            )
        
        # Load power data
        power_df = pd.read_csv(power_path, parse_dates=['timestamp'])
        power_df = power_df.set_index('timestamp')
        
        # Load weather data if available
        if weather_path.exists():
            weather_df = pd.read_csv(weather_path, parse_dates=['timestamp'])
            self._weather_data = weather_df
        
        self._raw_data = power_df
        return power_df
    
    def load_wind_data(self) -> pd.DataFrame:
        """
        Load GEFCom2014 wind track data.
        
        Expected format:
        - wind_power.csv: timestamp, zone1, ..., zone10 (10 wind farms)
        - wind_weather.csv: timestamp, zone, u10, v10, u100, v100, ...
        """
        power_path = self.data_dir / "wind" / "wind_power.csv"
        weather_path = self.data_dir / "wind" / "wind_weather.csv"
        
        if not power_path.exists():
            power_path = self.data_dir / "wind_power.csv"
            weather_path = self.data_dir / "wind_weather.csv"
        
        if not power_path.exists():
            raise FileNotFoundError(
                f"Wind data not found at {power_path}. "
                "Please download GEFCom2014 data."
            )
        
        power_df = pd.read_csv(power_path, parse_dates=['timestamp'])
        power_df = power_df.set_index('timestamp')
        
        if weather_path.exists():
            weather_df = pd.read_csv(weather_path, parse_dates=['timestamp'])
            self._weather_data = weather_df
        
        self._raw_data = power_df
        return power_df
    
    def create_synthetic_gefcom_data(
        self,
        num_zones: int = 3,
        num_days: int = 365 * 2,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Create synthetic GEFCom-like data for testing.
        
        Simulates solar/wind patterns with:
        - Daily patterns (for solar)
        - Weather correlation
        - Seasonal variation
        """
        rng = np.random.default_rng(seed)
        
        start_date = pd.Timestamp("2012-01-01")
        timestamps = pd.date_range(start_date, periods=num_days*24, freq='H')
        
        data = {'timestamp': timestamps}
        
        for zone in range(1, num_zones + 1):
            # Base capacity
            capacity = rng.uniform(50, 200)
            
            power = np.zeros(len(timestamps))
            
            for i, ts in enumerate(timestamps):
                hour = ts.hour
                day_of_year = ts.dayofyear
                
                if self.config.track == 'solar':
                    # Solar: only during daylight
                    # Sunrise ~ 6am, sunset ~ 6pm, peak at noon
                    if 6 <= hour <= 18:
                        # Bell curve for daily pattern
                        hour_factor = np.exp(-((hour - 12) ** 2) / 18)
                        
                        # Seasonal variation (more sun in summer)
                        seasonal = 0.5 + 0.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
                        
                        # Weather noise (clouds)
                        weather = rng.beta(2, 1)  # Skewed toward clear
                        
                        power[i] = capacity * hour_factor * seasonal * weather
                    else:
                        power[i] = 0
                else:
                    # Wind: more variable, no daily pattern
                    # AR(1) process for persistence
                    if i > 0:
                        persistence = 0.95
                        power[i] = (persistence * power[i-1] + 
                                   (1 - persistence) * capacity * rng.beta(2, 3))
                    else:
                        power[i] = capacity * rng.beta(2, 3)
                    
                    # Weather events (storms)
                    if rng.random() < 0.01:
                        power[i] *= rng.uniform(0.1, 0.5)  # Curtailment
                
                # Add noise
                power[i] = max(0, power[i] + rng.normal(0, capacity * 0.05))
            
            data[f'zone{zone}'] = power
        
        df = pd.DataFrame(data)
        df = df.set_index('timestamp')
        
        # Create synthetic weather
        weather_data = []
        for zone in range(1, num_zones + 1):
            for ts in timestamps:
                weather_data.append({
                    'timestamp': ts,
                    'zone': zone,
                    'temperature': 15 + 10 * np.sin(2 * np.pi * ts.dayofyear / 365) + rng.normal(0, 3),
                    'cloud_cover': rng.beta(2, 3),
                    'wind_speed': rng.exponential(5),
                    'humidity': rng.beta(3, 2),
                })
        
        self._weather_data = pd.DataFrame(weather_data)
        self._raw_data = df
        
        return df
    
    def compute_features(
        self,
        power_series: pd.Series,
        timestamp: pd.Timestamp,
        zone: int,
        feature_subset: str,
        lookback_hours: int
    ) -> np.ndarray:
        """
        Compute features for a given timestamp and zone.
        
        Args:
            power_series: Historical power values
            timestamp: Current timestamp
            zone: Zone index
            feature_subset: 'minimal', 'weather', or 'full'
            lookback_hours: How far back to look
            
        Returns:
            Feature vector
        """
        features = [1.0]  # Bias
        
        # Get historical data
        history_end = timestamp - timedelta(hours=1)
        history_start = history_end - timedelta(hours=lookback_hours)
        
        history = power_series[history_start:history_end]
        
        # Lagged power features (always included)
        if len(history) > 0:
            features.extend([
                history.iloc[-1] if len(history) >= 1 else 0,      # Last hour
                history.iloc[-24] if len(history) >= 24 else 0,    # Same hour yesterday
                history.iloc[-168] if len(history) >= 168 else 0,  # Same hour last week
                np.mean(history),
                np.std(history) + 1e-6,
                np.max(history),
            ])
        else:
            features.extend([0, 0, 0, 0, 1, 0])
        
        # Time features
        features.extend([
            timestamp.hour / 23.0,
            timestamp.dayofweek / 6.0,
            np.sin(2 * np.pi * timestamp.hour / 24),
            np.cos(2 * np.pi * timestamp.hour / 24),
            np.sin(2 * np.pi * timestamp.dayofyear / 365),
            np.cos(2 * np.pi * timestamp.dayofyear / 365),
        ])
        
        # Weather features (if requested and available)
        if feature_subset in ['weather', 'full'] and self._weather_data is not None:
            weather = self._weather_data[
                (self._weather_data['timestamp'] == timestamp) &
                (self._weather_data['zone'] == zone)
            ]
            
            if len(weather) > 0:
                row = weather.iloc[0]
                features.extend([
                    row.get('temperature', 15) / 40.0,
                    row.get('cloud_cover', 0.5),
                    row.get('wind_speed', 5) / 20.0,
                    row.get('humidity', 0.5),
                ])
            else:
                features.extend([0.5, 0.5, 0.25, 0.5])
        
        # Full features: add rolling statistics
        if feature_subset == 'full' and len(history) >= 24:
            features.extend([
                np.mean(history[-24:]),   # Last 24h mean
                np.std(history[-24:]),    # Last 24h std
                np.percentile(history, 10),
                np.percentile(history, 90),
            ])
        elif feature_subset == 'full':
            features.extend([0, 1, 0, 0])
        
        return np.array(features, dtype=np.float32)
    
    def build_specification_space(self) -> List[Dict[str, Any]]:
        """
        Build the specification space for the bandit.
        
        Each specification is a combination of:
        - lookback_hours: How much history to use
        - forecast_horizon: How far ahead to predict
        - feature_subset: Which features to include
        """
        specifications = []
        
        for lookback in self.config.lookback_hours:
            for horizon in self.config.forecast_horizons:
                for features in self.config.feature_subsets:
                    specifications.append({
                        'lookback_hours': lookback,
                        'forecast_horizon': horizon,
                        'feature_subset': features,
                        'spec_id': len(specifications),
                    })
        
        return specifications
    
    def prepare_dataset(self) -> Dict[str, Any]:
        """
        Prepare full dataset for experiments.
        
        Returns:
            Dataset dictionary with all components
        """
        # Check cache
        cache_path = self.cache_dir / f"gefcom_{self.config.track}_dataset.pkl"
        if self.config.use_cache and cache_path.exists():
            logger.info(f"Loading cached dataset from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Load data
        try:
            if self.config.track == 'solar':
                power_df = self.load_solar_data()
            elif self.config.track == 'wind':
                power_df = self.load_wind_data()
            else:
                raise ValueError(f"Unknown track: {self.config.track}")
        except FileNotFoundError:
            logger.warning("GEFCom data not found, creating synthetic data")
            power_df = self.create_synthetic_gefcom_data()
        
        # Get zones
        zone_cols = [c for c in power_df.columns if c.startswith('zone')]
        zones = self.config.zones or list(range(1, len(zone_cols) + 1))
        
        # Time splits
        train_end = pd.Timestamp(self.config.train_end)
        val_end = pd.Timestamp(self.config.val_end)
        
        train_mask = power_df.index <= train_end
        val_mask = (power_df.index > train_end) & (power_df.index <= val_end)
        test_mask = power_df.index > val_end
        
        # Build specification space
        specifications = self.build_specification_space()
        
        # Compute feature dimension
        sample_features = self.compute_features(
            power_df[zone_cols[0]],
            power_df.index[1000],
            1,
            'full',
            168
        )
        feature_dim = len(sample_features)
        
        dataset = {
            # Core data
            'power_df': power_df,
            'zone_columns': zone_cols,
            'zones': zones,
            
            # Splits
            'train_end': train_end,
            'val_end': val_end,
            'train_indices': power_df.index[train_mask].tolist(),
            'val_indices': power_df.index[val_mask].tolist(),
            'test_indices': power_df.index[test_mask].tolist(),
            
            # Specifications
            'specifications': specifications,
            'num_specifications': len(specifications),
            
            # Dimensions
            'num_zones': len(zones),
            'num_timestamps': len(power_df),
            'feature_dim': feature_dim,
            
            # Competition settings
            'quantiles': GEFCOM_QUANTILES,
            
            # Metadata
            'track': self.config.track,
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


class GEFComExperimentRunner:
    """
    Run Conformal Thompson Sampling experiments on GEFCom2014 data.
    """
    
    def __init__(
        self,
        dataset: Dict[str, Any],
        agent,
        baselines: Optional[Dict[str, Any]] = None,
        checkpoint_dir: str = "./checkpoints/gefcom"
    ):
        self.dataset = dataset
        self.agent = agent
        self.baselines = baselines or {}
        self.checkpoint_dir = Path(checkpoint_dir)
        
        self.loader = GEFCom2014Loader(
            GEFComConfig(track=dataset['track'])
        )
        self.loader._raw_data = dataset['power_df']
        
        # Pending updates for deferred outcomes
        self.pending_updates = {}
    
    def get_context(
        self,
        zone: int,
        timestamp: pd.Timestamp,
        spec: Dict[str, Any]
    ) -> np.ndarray:
        """Get context features for given zone, time, and specification."""
        zone_col = f'zone{zone}'
        power_series = self.dataset['power_df'][zone_col]
        
        return self.loader.compute_features(
            power_series,
            timestamp,
            zone,
            spec['feature_subset'],
            spec['lookback_hours']
        )
    
    def get_target(
        self,
        zone: int,
        timestamp: pd.Timestamp,
        horizon: int
    ) -> float:
        """Get target value (power at timestamp + horizon)."""
        zone_col = f'zone{zone}'
        target_time = timestamp + timedelta(hours=horizon)
        
        power_df = self.dataset['power_df']
        if target_time in power_df.index:
            return float(power_df.loc[target_time, zone_col])
        else:
            return np.nan
    
    def run_training(
        self,
        log_interval: int = 100
    ) -> Dict[str, Any]:
        """Run training loop."""
        train_indices = self.dataset['train_indices']
        zones = self.dataset['zones']
        specifications = self.dataset['specifications']
        
        logger.info(f"Training on {len(train_indices)} timestamps, {len(zones)} zones")
        
        all_scores = []
        all_coverages = []
        all_actions = []
        
        max_horizon = max(s['forecast_horizon'] for s in specifications)
        
        for i, timestamp in enumerate(train_indices[:-max_horizon]):
            # Process pending updates
            self._process_pending_updates(timestamp)
            
            for zone in zones:
                # Get base context (will be modified by spec)
                base_spec = specifications[0]
                context = self.get_context(zone, timestamp, base_spec)
                
                # Agent selects specification
                action, lower, upper = self.agent.select_and_predict(context)
                spec = specifications[action]
                
                # Schedule deferred update
                outcome_time = timestamp + timedelta(hours=spec['forecast_horizon'])
                if outcome_time not in self.pending_updates:
                    self.pending_updates[outcome_time] = []
                
                self.pending_updates[outcome_time].append({
                    'zone': zone,
                    'timestamp': timestamp,
                    'action': action,
                    'context': context,
                    'lower': lower,
                    'upper': upper,
                    'spec': spec,
                })
                
                all_actions.append(action)
            
            if i % log_interval == 0:
                recent = all_scores[-1000:] if all_scores else [0]
                logger.info(f"Step {i}/{len(train_indices)}: "
                           f"mean_score={np.mean(recent):.2f}")
        
        # Flush pending updates
        for ts in sorted(self.pending_updates.keys()):
            self._process_pending_updates(ts, force=True)
        
        return {
            'scores': all_scores,
            'coverages': all_coverages,
            'actions': all_actions,
        }
    
    def _process_pending_updates(
        self, 
        current_time: pd.Timestamp,
        force: bool = False
    ):
        """Process matured updates."""
        from conformal_ts.evaluation.metrics import interval_score
        
        times_to_process = [t for t in self.pending_updates.keys()
                          if t <= current_time or force]
        
        for ts in times_to_process:
            updates = self.pending_updates.pop(ts, [])
            
            for update in updates:
                target = self.get_target(
                    update['zone'],
                    update['timestamp'],
                    update['spec']['forecast_horizon']
                )
                
                if not np.isnan(target):
                    self.agent.update(
                        update['action'],
                        update['context'],
                        target
                    )
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run evaluation on test set."""
        from conformal_ts.evaluation.metrics import (
            interval_score, pinball_loss
        )
        
        test_indices = self.dataset['test_indices']
        zones = self.dataset['zones']
        specifications = self.dataset['specifications']
        
        max_horizon = max(s['forecast_horizon'] for s in specifications)
        
        results = {
            'cts': {'scores': [], 'coverages': [], 'pinball': []},
        }
        for name in self.baselines:
            results[name] = {'scores': [], 'coverages': [], 'pinball': []}
        
        for timestamp in test_indices[:-max_horizon]:
            for zone in zones:
                # CTS
                context = self.get_context(zone, timestamp, specifications[0])
                action, lower, upper = self.agent.select_and_predict(context)
                spec = specifications[action]
                
                target = self.get_target(zone, timestamp, spec['forecast_horizon'])
                
                if not np.isnan(target):
                    score = interval_score(
                        np.array([lower]),
                        np.array([upper]),
                        np.array([target]),
                        alpha=0.10
                    )[0]
                    
                    results['cts']['scores'].append(score)
                    results['cts']['coverages'].append(lower <= target <= upper)
                    
                    # Pinball at median
                    median_pred = (lower + upper) / 2
                    results['cts']['pinball'].append(
                        pinball_loss(np.array([median_pred]), np.array([target]), 0.5)[0]
                    )
        
        # Summary
        summary = {}
        for method, data in results.items():
            if data['scores']:
                summary[method] = {
                    'mean_score': float(np.mean(data['scores'])),
                    'coverage': float(np.mean(data['coverages'])),
                    'mean_pinball': float(np.mean(data['pinball'])),
                }
        
        return {'detailed': results, 'summary': summary}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with synthetic data
    config = GEFComConfig(
        track='solar',
        data_dir='./data/gefcom2014'
    )
    
    loader = GEFCom2014Loader(config)
    
    # Create synthetic data for testing
    logger.info("Creating synthetic GEFCom-like data...")
    power_df = loader.create_synthetic_gefcom_data(num_zones=3, num_days=365)
    
    # Prepare dataset
    dataset = loader.prepare_dataset()
    
    print(f"\nDataset prepared:")
    print(f"  Track: {dataset['track']}")
    print(f"  Zones: {dataset['num_zones']}")
    print(f"  Timestamps: {dataset['num_timestamps']}")
    print(f"  Specifications: {dataset['num_specifications']}")
    print(f"  Feature dim: {dataset['feature_dim']}")
    print(f"  Train samples: {len(dataset['train_indices'])}")
    print(f"  Test samples: {len(dataset['test_indices'])}")
