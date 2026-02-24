"""
Conformal Thompson Sampling Agent.

This module combines:
- Linear Thompson Sampling for specification selection
- Conformalized Quantile Regression for uncertainty quantification
- Interval scores as rewards

The agent adaptively selects among specification options based on
context, generating prediction intervals with coverage guarantees.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import json
from pathlib import Path

from ..models.linear_ts import LinearThompsonSampling
from ..models.cqr import ConformizedQuantileRegression, ActionSpecificCQR
from ..evaluation.metrics import interval_score_reward, OnlineMetrics


@dataclass
class CTSConfig:
    """Configuration for Conformal Thompson Sampling agent."""
    
    # Number of specifications (actions)
    num_actions: int = 16
    
    # Feature dimension
    feature_dim: int = 10
    
    # Thompson Sampling parameters
    prior_precision: float = 1.0
    exploration_variance: float = 1.0
    
    # CQR parameters
    coverage_target: float = 0.90
    cqr_learning_rate: float = 0.02
    cqr_l2_reg: float = 1e-4
    calibration_window: int = 250
    
    # Reward parameters
    alpha: float = 0.10  # For interval score
    reward_scale: float = 1.0
    clip_rewards: bool = True
    clip_min: float = -10.0
    clip_max: float = 0.0
    
    # Warm-up
    warmup_rounds: int = 50
    
    # Random seed
    seed: Optional[int] = None


class ConformalThompsonSampling:
    """
    Conformal Thompson Sampling agent for adaptive specification selection.
    
    The agent:
    1. Observes context (market state, series features)
    2. Uses Thompson Sampling to select a specification
    3. Uses CQR to generate a prediction interval
    4. Observes outcome and receives interval score reward
    5. Updates both TS posterior and CQR models
    """
    
    def __init__(self, config: CTSConfig):
        """
        Initialize CTS agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        
        # Thompson Sampling for action selection
        self.bandit = LinearThompsonSampling(
            num_actions=config.num_actions,
            feature_dim=config.feature_dim,
            prior_precision=config.prior_precision,
            exploration_variance=config.exploration_variance,
            seed=self.rng.integers(0, 2**31)
        )
        
        # CQR models for each action
        self.cqr_models = ActionSpecificCQR(
            num_actions=config.num_actions,
            feature_dim=config.feature_dim,
            coverage_target=config.coverage_target,
            learning_rate=config.cqr_learning_rate,
            l2_reg=config.cqr_l2_reg,
            calibration_window=config.calibration_window,
            seed=self.rng.integers(0, 2**31)
        )
        
        # Online metrics tracking
        self.metrics = OnlineMetrics(window_size=1000)
        
        # State tracking
        self.total_rounds = 0
        self.warmup_rounds = config.warmup_rounds
        
        # History for analysis
        self.action_history: List[int] = []
        self.reward_history: List[float] = []
        self.context_history: List[np.ndarray] = []
        
        # Pending updates (for delayed feedback)
        self._pending_updates: Dict[int, Dict[str, Any]] = {}
        self._update_counter = 0
    
    def select_action(
        self,
        context: np.ndarray,
        available_actions: Optional[List[int]] = None
    ) -> int:
        """
        Select specification based on context.
        
        During warmup, uses round-robin exploration.
        After warmup, uses Thompson Sampling.
        
        Args:
            context: Feature vector
            available_actions: List of available specification indices
        
        Returns:
            Selected specification index
        """
        context = np.asarray(context).flatten()
        
        if available_actions is None:
            available_actions = list(range(self.config.num_actions))
        
        # Warmup: explore uniformly
        if self.total_rounds < self.warmup_rounds:
            action = available_actions[
                self.total_rounds % len(available_actions)
            ]
        else:
            # Thompson Sampling selection
            action = self.bandit.select_action(context, available_actions)
        
        return action
    
    def predict_interval(
        self,
        action: int,
        context: np.ndarray
    ) -> Tuple[float, float]:
        """
        Generate prediction interval for selected action.
        
        Args:
            action: Selected specification
            context: Feature vector
        
        Returns:
            (lower_bound, upper_bound) prediction interval
        """
        return self.cqr_models.predict_interval(action, context)
    
    def select_and_predict(
        self,
        context: np.ndarray,
        available_actions: Optional[List[int]] = None
    ) -> Tuple[int, float, float]:
        """
        Select action and generate prediction interval in one call.
        
        Args:
            context: Feature vector
            available_actions: Available specification indices
        
        Returns:
            (action, lower_bound, upper_bound)
        """
        action = self.select_action(context, available_actions)
        lower, upper = self.predict_interval(action, context)
        return action, lower, upper
    
    def update(
        self,
        action: int,
        context: np.ndarray,
        outcome: float
    ) -> float:
        """
        Update agent with observed outcome.
        
        Computes interval score reward and updates:
        - Thompson Sampling posterior
        - CQR quantile models and calibration
        
        Args:
            action: Selected action
            context: Feature vector used for selection
            outcome: True outcome value
        
        Returns:
            Reward (negative interval score)
        """
        context = np.asarray(context).flatten()
        
        # Get prediction interval
        lower, upper = self.cqr_models.predict_interval(action, context)
        
        # Compute reward (negative interval score)
        reward = interval_score_reward(
            lower, upper, outcome,
            alpha=self.config.alpha,
            scale=self.config.reward_scale,
            clip_min=self.config.clip_min if self.config.clip_rewards else None,
            clip_max=self.config.clip_max if self.config.clip_rewards else None
        )
        
        # Update Thompson Sampling
        self.bandit.update(action, context, reward)
        
        # Update CQR
        self.cqr_models.update(action, context, outcome)
        
        # Update metrics
        self.metrics.update(lower, upper, outcome, reward)
        
        # Track history
        self.total_rounds += 1
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.context_history.append(context.copy())
        
        return reward
    
    def register_pending(
        self,
        action: int,
        context: np.ndarray,
        horizon: int
    ) -> int:
        """
        Register a pending update for delayed feedback.
        
        Use when outcome won't be available immediately.
        
        Args:
            action: Selected action
            context: Feature vector
            horizon: Number of steps until outcome available
        
        Returns:
            Update ID for later reference
        """
        update_id = self._update_counter
        self._update_counter += 1
        
        self._pending_updates[update_id] = {
            'action': action,
            'context': context.copy(),
            'horizon': horizon,
            'registered_at': self.total_rounds,
        }
        
        return update_id
    
    def complete_pending(
        self,
        update_id: int,
        outcome: float
    ) -> float:
        """
        Complete a pending update when outcome becomes available.
        
        Args:
            update_id: Update ID from register_pending
            outcome: True outcome value
        
        Returns:
            Reward
        """
        if update_id not in self._pending_updates:
            raise ValueError(f"Unknown update_id: {update_id}")
        
        pending = self._pending_updates.pop(update_id)
        return self.update(pending['action'], pending['context'], outcome)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current agent statistics."""
        bandit_stats = self.bandit.get_action_statistics()
        metric_stats = self.metrics.get_statistics()
        
        return {
            'total_rounds': self.total_rounds,
            'warmup_complete': self.total_rounds >= self.warmup_rounds,
            'pending_updates': len(self._pending_updates),
            
            # Bandit statistics
            'action_pull_counts': bandit_stats['pull_counts'],
            'action_mean_rewards': bandit_stats['mean_rewards'],
            
            # Performance metrics
            'recent_mean_score': metric_stats['mean_score'],
            'recent_coverage': metric_stats['coverage'],
            'recent_mean_width': metric_stats['mean_width'],
            'recent_mean_reward': metric_stats['mean_reward'],
        }
    
    def get_action_probabilities(
        self,
        context: np.ndarray,
        n_samples: int = 1000
    ) -> np.ndarray:
        """
        Estimate action selection probabilities via Monte Carlo.
        
        Args:
            context: Feature vector
            n_samples: Number of Monte Carlo samples
        
        Returns:
            Probability for each action
        """
        context = np.asarray(context).flatten()
        counts = np.zeros(self.config.num_actions)
        
        for _ in range(n_samples):
            action = self.bandit.select_action(context)
            counts[action] += 1
        
        return counts / n_samples
    
    def save(self, path: str):
        """Save agent state to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'config': self.config.__dict__,
            'bandit_state': self.bandit.save_state(),
            'cqr_state': self.cqr_models.get_state(),
            'total_rounds': self.total_rounds,
            'action_history': self.action_history,
            'reward_history': self.reward_history,
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    
    @classmethod
    def load(cls, path: str) -> 'ConformalThompsonSampling':
        """Load agent state from file."""
        with open(path, 'r') as f:
            state = json.load(f)
        
        config = CTSConfig(**state['config'])
        agent = cls(config)
        
        agent.bandit.load_state(state['bandit_state'])
        agent.cqr_models.load_state(state['cqr_state'])
        agent.total_rounds = state['total_rounds']
        agent.action_history = state['action_history']
        agent.reward_history = state['reward_history']
        
        return agent


class ConformalThompsonSamplingEnsemble:
    """
    Ensemble of CTS agents for index-weighted aggregation.
    
    Useful when predictions need to be aggregated across
    multiple series with different weights.
    """
    
    def __init__(
        self,
        config: CTSConfig,
        num_series: int,
        weights: Optional[np.ndarray] = None
    ):
        """
        Initialize CTS ensemble.
        
        Args:
            config: Agent configuration
            num_series: Number of series
            weights: Series weights (uniform if None)
        """
        self.config = config
        self.num_series = num_series
        self.weights = weights if weights is not None else np.ones(num_series) / num_series
        
        # Single shared agent (parameters shared across series)
        self.agent = ConformalThompsonSampling(config)
    
    def select_action(
        self,
        contexts: np.ndarray,
        available_actions: Optional[List[int]] = None
    ) -> int:
        """
        Select a single action for all series.
        
        Uses weighted average context.
        
        Args:
            contexts: Feature matrix (num_series, feature_dim)
            available_actions: Available specifications
        
        Returns:
            Selected specification
        """
        # Weighted average context
        weighted_context = np.average(contexts, weights=self.weights, axis=0)
        return self.agent.select_action(weighted_context, available_actions)
    
    def predict_intervals(
        self,
        action: int,
        contexts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals for all series.
        
        Args:
            action: Selected specification
            contexts: Feature matrix
        
        Returns:
            (lower_bounds, upper_bounds) arrays
        """
        lowers = []
        uppers = []
        
        for i in range(self.num_series):
            lower, upper = self.agent.predict_interval(action, contexts[i])
            lowers.append(lower)
            uppers.append(upper)
        
        return np.array(lowers), np.array(uppers)
    
    def update(
        self,
        action: int,
        contexts: np.ndarray,
        outcomes: np.ndarray
    ) -> float:
        """
        Update with outcomes for all series.
        
        Computes weighted average reward.
        
        Args:
            action: Selected action
            contexts: Feature matrix
            outcomes: Outcome values
        
        Returns:
            Weighted average reward
        """
        rewards = []
        
        for i in range(self.num_series):
            reward = self.agent.update(action, contexts[i], outcomes[i])
            rewards.append(reward)
        
        return float(np.average(rewards, weights=self.weights))
    
    def get_weighted_metrics(
        self,
        contexts: np.ndarray,
        outcomes: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute weighted metrics across all series.
        
        Args:
            contexts: Feature matrix
            outcomes: Outcome values
        
        Returns:
            Dictionary of weighted metrics
        """
        from ..evaluation.metrics import interval_score, coverage_rate
        
        # Get intervals for current best action
        weighted_ctx = np.average(contexts, weights=self.weights, axis=0)
        action = self.agent.select_action(weighted_ctx)
        lowers, uppers = self.predict_intervals(action, contexts)
        
        # Compute metrics
        scores = interval_score(lowers, uppers, outcomes, self.config.alpha)
        weighted_score = np.average(scores, weights=self.weights)
        
        covered = (outcomes >= lowers) & (outcomes <= uppers)
        weighted_coverage = np.average(covered, weights=self.weights)
        
        widths = uppers - lowers
        weighted_width = np.average(widths, weights=self.weights)
        
        return {
            'weighted_interval_score': float(weighted_score),
            'weighted_coverage': float(weighted_coverage),
            'weighted_width': float(weighted_width),
        }


def create_agent_from_config(config_path: str) -> ConformalThompsonSampling:
    """
    Create CTS agent from configuration file.
    
    Args:
        config_path: Path to JSON config file
    
    Returns:
        Configured CTS agent
    """
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = CTSConfig(**config_dict)
    return ConformalThompsonSampling(config)


if __name__ == "__main__":
    # Test the agent
    np.random.seed(42)
    
    print("Testing Conformal Thompson Sampling Agent")
    print("=" * 50)
    
    # Create agent
    config = CTSConfig(
        num_actions=4,
        feature_dim=8,
        prior_precision=1.0,
        exploration_variance=1.0,
        coverage_target=0.90,
        warmup_rounds=5,
        seed=42
    )
    
    agent = ConformalThompsonSampling(config)
    
    # Simulate environment
    true_thetas = np.random.randn(config.num_actions, config.feature_dim)
    noise_scales = [0.5 + 0.2 * i for i in range(config.num_actions)]
    
    # Run simulation
    num_rounds = 200
    cumulative_reward = 0.0
    
    for t in range(num_rounds):
        # Generate context
        context = np.random.randn(config.feature_dim)
        context[0] = 1.0  # Bias
        
        # Select action and get prediction
        action, lower, upper = agent.select_and_predict(context)
        
        # Generate true outcome
        true_mean = context @ true_thetas[action]
        noise = np.random.randn() * noise_scales[action]
        outcome = true_mean + noise
        
        # Update agent
        reward = agent.update(action, context, outcome)
        cumulative_reward += reward
        
        if (t + 1) % 50 == 0:
            stats = agent.get_statistics()
            print(f"\nRound {t + 1}:")
            print(f"  Recent coverage: {stats['recent_coverage']:.2%}")
            print(f"  Recent mean score: {stats['recent_mean_score']:.4f}")
            print(f"  Action counts: {stats['action_pull_counts']}")
    
    print("\n" + "=" * 50)
    print("Final Statistics:")
    stats = agent.get_statistics()
    print(f"Total rounds: {stats['total_rounds']}")
    print(f"Coverage: {stats['recent_coverage']:.2%}")
    print(f"Mean interval score: {stats['recent_mean_score']:.4f}")
    print(f"Cumulative reward: {cumulative_reward:.2f}")
    
    # Test save/load
    print("\nTesting save/load...")
    agent.save("/tmp/test_agent.json")
    loaded_agent = ConformalThompsonSampling.load("/tmp/test_agent.json")
    print(f"Loaded agent rounds: {loaded_agent.total_rounds}")
