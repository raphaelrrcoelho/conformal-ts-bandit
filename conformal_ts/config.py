"""
Configuration for Conformal Thompson Sampling.

This module defines all hyperparameters and settings for:
- Linear Thompson Sampling
- Conformalized Quantile Regression
- Dataset-specific configurations
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import json
from pathlib import Path


class DatasetType(Enum):
    """Supported dataset types."""
    M5 = "m5"
    GEFCOM2014_SOLAR = "gefcom2014_solar"
    GEFCOM2014_WIND = "gefcom2014_wind"
    GEFCOM2014_LOAD = "gefcom2014_load"
    SYNTHETIC = "synthetic"


@dataclass
class ThompsonSamplingConfig:
    """Configuration for Linear Thompson Sampling."""
    
    # Prior regularization (lambda in the dissertation)
    prior_precision: float = 0.1
    
    # Variance inflation for exploration (sigma^2_explore)
    exploration_variance: float = 5.0
    
    # Feature dimension (will be set based on dataset)
    feature_dim: int = 10
    
    # Numerical stability
    condition_number_threshold: float = 1e10
    regularization_epsilon: float = 1e-6
    recompute_interval: int = 100  # Recompute covariance every N updates
    
    # Warm-up period before Thompson Sampling kicks in
    warmup_rounds: int = 50


@dataclass
class CQRConfig:
    """Configuration for Conformalized Quantile Regression."""
    
    # Target coverage level (1 - alpha)
    coverage_target: float = 0.90
    
    # Quantile levels for prediction intervals
    lower_quantile: float = 0.05
    upper_quantile: float = 0.95
    
    # Online learning rate for quantile regression
    learning_rate: float = 0.02
    
    # L2 regularization coefficient
    l2_regularization: float = 1e-4
    
    # Calibration window size (rolling buffer)
    calibration_window: int = 250
    
    # Warm-up bounds for intervals
    warmup_min_observations: int = 20
    warmup_interval_min: float = 0.002  # 0.2%
    warmup_interval_max: float = 0.05   # 5%


@dataclass
class IntervalScoreConfig:
    """Configuration for interval score reward computation."""
    
    # Coverage level for interval score penalty
    alpha: float = 0.10  # For 90% intervals
    
    # Optional reward scaling
    reward_scale: float = 1.0
    
    # Whether to clip extreme rewards
    clip_rewards: bool = True
    clip_min: float = -10.0
    clip_max: float = 0.0  # Rewards are negative interval scores


@dataclass
class SpecificationConfig:
    """Configuration for specification options (action space)."""
    
    # Lookback windows (in time steps)
    lookback_windows: List[int] = field(default_factory=lambda: [10, 21, 42, 63])
    
    # Forecast horizons (in time steps)
    forecast_horizons: List[int] = field(default_factory=lambda: [10, 21, 42, 63])
    
    # Whether to use all combinations or subset
    use_all_combinations: bool = True
    
    # Optional: specify exact (lookback, horizon) pairs
    custom_specifications: Optional[List[tuple]] = None
    
    @property
    def num_actions(self) -> int:
        """Number of specification options."""
        if self.custom_specifications:
            return len(self.custom_specifications)
        return len(self.lookback_windows) * len(self.forecast_horizons)
    
    @property
    def action_space(self) -> List[tuple]:
        """Get all (lookback, horizon) combinations."""
        if self.custom_specifications:
            return self.custom_specifications
        return [
            (w, h) 
            for w in self.lookback_windows 
            for h in self.forecast_horizons
        ]


@dataclass
class M5Config:
    """M5 dataset-specific configuration."""
    
    # Data paths
    data_dir: str = "./data/m5"
    
    # Aggregation level for experiments
    # Options: 'item', 'dept', 'cat', 'store', 'state', 'total'
    aggregation_level: str = "store"
    
    # Sample size for development (None = use all)
    sample_size: Optional[int] = 1000
    
    # Train/validation/test split
    train_days: int = 1000  # First N days for training
    val_days: int = 100     # Next N days for validation
    test_days: int = 28     # Final N days for test (M5 competition horizon)
    
    # Features to use
    use_price_features: bool = True
    use_calendar_features: bool = True
    use_snap_features: bool = True
    
    # Lookback windows adapted for daily retail data
    lookback_windows: List[int] = field(default_factory=lambda: [7, 14, 28, 56])
    
    # Forecast horizons
    forecast_horizons: List[int] = field(default_factory=lambda: [7, 14, 28])


@dataclass
class GEFCom2014Config:
    """GEFCom2014 dataset-specific configuration."""
    
    # Data paths
    data_dir: str = "./data/gefcom2014"
    
    # Track selection
    track: str = "solar"  # Options: 'solar', 'wind', 'load', 'price'
    
    # Zone/station selection (depends on track)
    zones: Optional[List[int]] = None  # None = use all
    
    # Train/validation/test split
    train_months: int = 24
    val_months: int = 6
    test_months: int = 6
    
    # Hourly aggregation for specifications
    lookback_windows: List[int] = field(default_factory=lambda: [24, 48, 168, 336])  # 1d, 2d, 1w, 2w in hours
    forecast_horizons: List[int] = field(default_factory=lambda: [24, 48, 168])  # 1d, 2d, 1w in hours
    
    # Weather features to use
    weather_features: List[str] = field(default_factory=lambda: [
        'Total_Tpv', 'VAR134', 'VAR157', 'VAR164', 'VAR165', 'VAR166',
        'VAR167', 'VAR169', 'VAR175', 'VAR178', 'VAR228'
    ])


@dataclass 
class SyntheticConfig:
    """Configuration for synthetic data generation (for testing)."""
    
    num_series: int = 100
    series_length: int = 1000
    
    # Regime switching parameters
    num_regimes: int = 3
    regime_persistence: float = 0.95
    
    # Noise levels
    observation_noise: float = 0.1
    regime_effect_size: float = 0.5
    
    # Optimal specification varies by regime
    regime_optimal_specs: List[int] = field(default_factory=lambda: [0, 1, 2])


@dataclass
class ExperimentConfig:
    """Overall experiment configuration."""
    
    # Experiment identification
    experiment_name: str = "conformal_ts"
    seed: int = 42
    
    # Dataset selection
    dataset_type: DatasetType = DatasetType.M5
    
    # Component configs
    ts_config: ThompsonSamplingConfig = field(default_factory=ThompsonSamplingConfig)
    cqr_config: CQRConfig = field(default_factory=CQRConfig)
    interval_score_config: IntervalScoreConfig = field(default_factory=IntervalScoreConfig)
    spec_config: SpecificationConfig = field(default_factory=SpecificationConfig)
    
    # Dataset-specific configs
    m5_config: M5Config = field(default_factory=M5Config)
    gefcom_config: GEFCom2014Config = field(default_factory=GEFCom2014Config)
    synthetic_config: SyntheticConfig = field(default_factory=SyntheticConfig)
    
    # Logging and checkpointing
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 100
    checkpoint_interval: int = 1000
    
    # Evaluation settings
    eval_interval: int = 500
    num_eval_episodes: int = 100
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict (handling enums and nested dataclasses)
        config_dict = self._to_dict()
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls._from_dict(config_dict)
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif hasattr(value, '__dataclass_fields__'):
                result[key] = {k: v for k, v in value.__dict__.items()}
            else:
                result[key] = value
        return result
    
    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        # Handle dataset_type enum
        if 'dataset_type' in config_dict:
            config_dict['dataset_type'] = DatasetType(config_dict['dataset_type'])
        
        # Handle nested configs
        if 'ts_config' in config_dict:
            config_dict['ts_config'] = ThompsonSamplingConfig(**config_dict['ts_config'])
        if 'cqr_config' in config_dict:
            config_dict['cqr_config'] = CQRConfig(**config_dict['cqr_config'])
        if 'interval_score_config' in config_dict:
            config_dict['interval_score_config'] = IntervalScoreConfig(**config_dict['interval_score_config'])
        if 'spec_config' in config_dict:
            config_dict['spec_config'] = SpecificationConfig(**config_dict['spec_config'])
        if 'm5_config' in config_dict:
            config_dict['m5_config'] = M5Config(**config_dict['m5_config'])
        if 'gefcom_config' in config_dict:
            config_dict['gefcom_config'] = GEFCom2014Config(**config_dict['gefcom_config'])
        if 'synthetic_config' in config_dict:
            config_dict['synthetic_config'] = SyntheticConfig(**config_dict['synthetic_config'])
        
        return cls(**config_dict)


def get_default_config(dataset: str = "m5") -> ExperimentConfig:
    """Get default configuration for a dataset."""
    config = ExperimentConfig()
    
    if dataset.lower() == "m5":
        config.dataset_type = DatasetType.M5
        config.spec_config.lookback_windows = config.m5_config.lookback_windows
        config.spec_config.forecast_horizons = config.m5_config.forecast_horizons
        config.ts_config.feature_dim = 12  # M5 features
        
    elif dataset.lower().startswith("gefcom"):
        track = dataset.split("_")[-1] if "_" in dataset else "solar"
        config.dataset_type = DatasetType(f"gefcom2014_{track}")
        config.gefcom_config.track = track
        config.spec_config.lookback_windows = config.gefcom_config.lookback_windows
        config.spec_config.forecast_horizons = config.gefcom_config.forecast_horizons
        config.ts_config.feature_dim = 15  # GEFCom features
        
    elif dataset.lower() == "synthetic":
        config.dataset_type = DatasetType.SYNTHETIC
        config.ts_config.feature_dim = 8
    
    return config


if __name__ == "__main__":
    # Test configuration
    config = get_default_config("m5")
    print(f"Dataset: {config.dataset_type.value}")
    print(f"Num actions: {config.spec_config.num_actions}")
    print(f"Action space: {config.spec_config.action_space}")
    
    # Test save/load
    config.save("/tmp/test_config.json")
    loaded = ExperimentConfig.load("/tmp/test_config.json")
    print(f"Loaded dataset: {loaded.dataset_type.value}")
