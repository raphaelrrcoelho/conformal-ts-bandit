"""
Synthetic Data Generator for Testing.

Creates controlled environments for testing Conformal Thompson Sampling:
- Regime-switching processes where optimal specification changes
- Heteroscedastic noise
- Non-stationary dynamics
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Generator
from dataclasses import dataclass


@dataclass
class RegimeConfig:
    """Configuration for a single regime."""
    
    # Which specification is optimal in this regime
    optimal_spec: int
    
    # Base mean return
    mean: float
    
    # Volatility level
    volatility: float
    
    # AR(1) coefficient for momentum effect
    ar_coefficient: float
    
    # How much the optimal spec outperforms others
    spec_advantage: float


class SyntheticDataGenerator:
    """
    Generate synthetic data with regime-switching dynamics.
    
    The optimal specification changes with the regime, simulating
    the real-world scenario where different forecast parameters
    work better under different market conditions.
    """
    
    def __init__(
        self,
        num_series: int = 100,
        num_specifications: int = 4,
        feature_dim: int = 8,
        num_regimes: int = 3,
        regime_persistence: float = 0.95,
        observation_noise: float = 0.1,
        spec_advantage: float = 0.3,
        seed: Optional[int] = None
    ):
        """
        Initialize synthetic data generator.
        
        Args:
            num_series: Number of time series to simulate
            num_specifications: Number of specification options
            feature_dim: Dimension of context features
            num_regimes: Number of market regimes
            regime_persistence: P(stay in same regime)
            observation_noise: Base observation noise std
            spec_advantage: How much optimal spec outperforms
            seed: Random seed
        """
        self.num_series = num_series
        self.num_specifications = num_specifications
        self.feature_dim = feature_dim
        self.num_regimes = num_regimes
        self.regime_persistence = regime_persistence
        self.observation_noise = observation_noise
        self.spec_advantage = spec_advantage
        
        self.rng = np.random.default_rng(seed)
        
        # Generate regime configurations
        self.regimes = self._generate_regimes()
        
        # Initialize state
        self.current_regime = self.rng.integers(0, num_regimes)
        self.t = 0
        
        # History buffers for features
        self._history: Dict[int, List[float]] = {
            i: [] for i in range(num_series)
        }
    
    def _generate_regimes(self) -> List[RegimeConfig]:
        """Generate random regime configurations."""
        regimes = []
        
        for i in range(self.num_regimes):
            # Each regime has a different optimal specification
            optimal_spec = i % self.num_specifications
            
            # Regime-specific parameters
            mean = self.rng.normal(0, 0.02)
            volatility = 0.05 + self.rng.uniform(0, 0.1)
            ar_coef = self.rng.uniform(-0.3, 0.3)
            
            regime = RegimeConfig(
                optimal_spec=optimal_spec,
                mean=mean,
                volatility=volatility,
                ar_coefficient=ar_coef,
                spec_advantage=self.spec_advantage
            )
            regimes.append(regime)
        
        return regimes
    
    def _transition_regime(self):
        """Potentially transition to new regime."""
        if self.rng.random() > self.regime_persistence:
            # Transition to random different regime
            new_regime = self.rng.integers(0, self.num_regimes)
            while new_regime == self.current_regime and self.num_regimes > 1:
                new_regime = self.rng.integers(0, self.num_regimes)
            self.current_regime = new_regime
    
    def _generate_context(self, series_idx: int) -> np.ndarray:
        """
        Generate context features for a series.
        
        Features include:
        - Bias term
        - Regime indicators (noisy)
        - Historical volatility
        - Recent returns
        - Noise features
        """
        context = np.zeros(self.feature_dim)
        
        # Bias term
        context[0] = 1.0
        
        # Noisy regime indicators
        regime = self.regimes[self.current_regime]
        context[1] = regime.volatility + self.rng.normal(0, 0.01)
        context[2] = regime.ar_coefficient + self.rng.normal(0, 0.1)
        
        # Historical features from series
        history = self._history[series_idx]
        if len(history) >= 5:
            context[3] = np.mean(history[-5:])  # Short MA
            context[4] = np.std(history[-5:]) if len(history) >= 5 else 0.1
        if len(history) >= 20:
            context[5] = np.mean(history[-20:])  # Long MA
            context[6] = np.std(history[-20:])
        
        # Noise features
        context[7:] = self.rng.normal(0, 0.5, size=self.feature_dim - 7)
        
        return context
    
    def _generate_return(
        self,
        series_idx: int,
        specification: int
    ) -> Tuple[float, float, float]:
        """
        Generate return for a series given specification choice.
        
        Returns:
            (true_return, lower_quantile, upper_quantile)
        """
        regime = self.regimes[self.current_regime]
        history = self._history[series_idx]
        
        # Base return from regime mean + AR term
        if len(history) > 0:
            ar_contribution = regime.ar_coefficient * history[-1]
        else:
            ar_contribution = 0.0
        
        base_return = regime.mean + ar_contribution
        
        # Specification effect
        if specification == regime.optimal_spec:
            # Optimal spec gets bonus
            spec_effect = regime.spec_advantage
        else:
            # Suboptimal specs get penalty proportional to distance
            distance = abs(specification - regime.optimal_spec)
            spec_effect = -0.1 * distance
        
        # Generate return with noise
        noise = self.rng.normal(0, regime.volatility)
        true_return = base_return + spec_effect * regime.volatility + noise
        
        # True quantiles
        lower_q = true_return - 1.645 * regime.volatility
        upper_q = true_return + 1.645 * regime.volatility
        
        return true_return, lower_q, upper_q
    
    def step(
        self,
        specification: int
    ) -> Dict[str, Any]:
        """
        Generate one step of data for all series.
        
        Args:
            specification: Chosen specification
        
        Returns:
            Dictionary with contexts, returns, intervals, regime info
        """
        # Potentially transition regime
        self._transition_regime()
        
        contexts = []
        returns = []
        lower_bounds = []
        upper_bounds = []
        
        for i in range(self.num_series):
            # Generate context
            ctx = self._generate_context(i)
            contexts.append(ctx)
            
            # Generate return
            ret, lower, upper = self._generate_return(i, specification)
            returns.append(ret)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
            
            # Update history
            self._history[i].append(ret)
            if len(self._history[i]) > 100:
                self._history[i] = self._history[i][-100:]
        
        self.t += 1
        
        return {
            'contexts': np.array(contexts),
            'returns': np.array(returns),
            'lower_bounds': np.array(lower_bounds),
            'upper_bounds': np.array(upper_bounds),
            'regime': self.current_regime,
            'optimal_spec': self.regimes[self.current_regime].optimal_spec,
            't': self.t,
        }
    
    def get_oracle_specification(self) -> int:
        """Get the optimal specification for current regime."""
        return self.regimes[self.current_regime].optimal_spec
    
    def generate_dataset(
        self,
        num_steps: int,
        specifications: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete dataset.
        
        Args:
            num_steps: Number of time steps
            specifications: Optional list of specifications to use
                           (if None, cycles through all)
        
        Returns:
            Complete dataset dictionary
        """
        all_contexts = []
        all_returns = []
        all_lower = []
        all_upper = []
        all_regimes = []
        all_specs = []
        all_optimal = []
        
        for t in range(num_steps):
            # Choose specification
            if specifications is not None:
                spec = specifications[t % len(specifications)]
            else:
                spec = t % self.num_specifications
            
            # Generate step
            data = self.step(spec)
            
            all_contexts.append(data['contexts'])
            all_returns.append(data['returns'])
            all_lower.append(data['lower_bounds'])
            all_upper.append(data['upper_bounds'])
            all_regimes.append(data['regime'])
            all_specs.append(spec)
            all_optimal.append(data['optimal_spec'])
        
        return {
            'contexts': np.array(all_contexts),  # (T, N, D)
            'returns': np.array(all_returns),    # (T, N)
            'lower_bounds': np.array(all_lower), # (T, N)
            'upper_bounds': np.array(all_upper), # (T, N)
            'regimes': np.array(all_regimes),    # (T,)
            'specifications': np.array(all_specs),  # (T,)
            'optimal_specs': np.array(all_optimal), # (T,)
        }
    
    def reset(self):
        """Reset generator state."""
        self.current_regime = self.rng.integers(0, self.num_regimes)
        self.t = 0
        self._history = {i: [] for i in range(self.num_series)}


class StreamingDataGenerator:
    """
    Generator that yields data one observation at a time.
    
    Suitable for online learning scenarios.
    """
    
    def __init__(
        self,
        base_generator: SyntheticDataGenerator,
        batch_size: int = 1
    ):
        """
        Initialize streaming generator.
        
        Args:
            base_generator: Underlying data generator
            batch_size: Number of series to sample per step
        """
        self.generator = base_generator
        self.batch_size = batch_size
        self.series_indices = list(range(base_generator.num_series))
    
    def __iter__(self) -> Generator[Dict[str, Any], int, None]:
        """
        Iterate over observations.
        
        Yields observation dict, receives specification choice.
        """
        while True:
            # Sample series for this batch
            if self.batch_size >= len(self.series_indices):
                indices = self.series_indices
            else:
                indices = self.generator.rng.choice(
                    self.series_indices,
                    size=self.batch_size,
                    replace=False
                ).tolist()
            
            # Generate contexts
            contexts = np.array([
                self.generator._generate_context(i) for i in indices
            ])
            
            # Yield context and wait for action
            obs = {
                'contexts': contexts,
                'series_indices': indices,
                'regime': self.generator.current_regime,
                't': self.generator.t,
            }
            
            # Receive specification choice from caller
            specification = yield obs
            
            # Generate outcomes
            returns = []
            lower_bounds = []
            upper_bounds = []
            
            for i in indices:
                ret, lower, upper = self.generator._generate_return(i, specification)
                returns.append(ret)
                lower_bounds.append(lower)
                upper_bounds.append(upper)
                
                # Update history
                self.generator._history[i].append(ret)
                if len(self.generator._history[i]) > 100:
                    self.generator._history[i] = self.generator._history[i][-100:]
            
            # Potentially transition regime
            self.generator._transition_regime()
            self.generator.t += 1
            
            # Yield outcomes
            outcome = {
                'returns': np.array(returns),
                'lower_bounds': np.array(lower_bounds),
                'upper_bounds': np.array(upper_bounds),
                'optimal_spec': self.generator.get_oracle_specification(),
            }
            yield outcome


def create_synthetic_experiment(
    num_series: int = 100,
    train_steps: int = 500,
    test_steps: int = 200,
    num_specs: int = 4,
    seed: int = 42
) -> Tuple[Dict[str, Any], Dict[str, Any], SyntheticDataGenerator]:
    """
    Create train/test split for synthetic experiment.
    
    Args:
        num_series: Number of time series
        train_steps: Number of training steps
        test_steps: Number of test steps
        num_specs: Number of specifications
        seed: Random seed
    
    Returns:
        (train_data, test_data, generator)
    """
    generator = SyntheticDataGenerator(
        num_series=num_series,
        num_specifications=num_specs,
        seed=seed
    )
    
    # Generate training data (explore all specs)
    train_data = generator.generate_dataset(
        train_steps,
        specifications=None  # Cycle through specs
    )
    
    # Generate test data (continue from same state)
    test_data = generator.generate_dataset(
        test_steps,
        specifications=None
    )
    
    return train_data, test_data, generator


if __name__ == "__main__":
    # Test the generator
    np.random.seed(42)
    
    generator = SyntheticDataGenerator(
        num_series=50,
        num_specifications=4,
        num_regimes=3,
        regime_persistence=0.95,
        seed=42
    )
    
    # Generate dataset
    data = generator.generate_dataset(100)
    
    print("Synthetic Data Generator Test")
    print(f"Contexts shape: {data['contexts'].shape}")
    print(f"Returns shape: {data['returns'].shape}")
    print(f"Unique regimes: {np.unique(data['regimes'])}")
    print(f"Regime distribution: {np.bincount(data['regimes'])}")
    
    # Test streaming generator
    print("\n--- Streaming Generator Test ---")
    generator.reset()
    stream = StreamingDataGenerator(generator, batch_size=10)
    
    gen = iter(stream)
    
    # First step: get context
    obs = next(gen)
    print(f"Step 0 - Contexts shape: {obs['contexts'].shape}")
    
    # Send action and get outcome
    outcome = gen.send(0)
    print(f"Outcome returns shape: {outcome['returns'].shape}")
    print(f"Optimal spec: {outcome['optimal_spec']}")
    
    # Create experiment
    print("\n--- Experiment Creation Test ---")
    train, test, gen = create_synthetic_experiment(
        num_series=50,
        train_steps=200,
        test_steps=50,
        seed=42
    )
    
    print(f"Train contexts: {train['contexts'].shape}")
    print(f"Test contexts: {test['contexts'].shape}")
    print(f"Train regime changes: {np.sum(np.diff(train['regimes']) != 0)}")
