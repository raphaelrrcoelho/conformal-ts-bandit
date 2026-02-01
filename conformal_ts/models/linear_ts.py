"""
Linear Thompson Sampling with Sherman-Morrison Updates.

This module implements the disjoint linear bandit model with:
- Bayesian posterior updates for each action
- Sherman-Morrison formula for O(d^2) covariance updates
- Exploration variance inflation
- Numerical stability safeguards

Reference:
    Agrawal & Goyal (2013) - Thompson Sampling for Contextual Bandits
    Russo & Van Roy (2018) - A Tutorial on Thompson Sampling
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import warnings


@dataclass
class ArmState:
    """State for a single arm in the bandit."""
    
    # Precision matrix A = lambda*I + sum(phi*phi^T)
    precision_matrix: np.ndarray
    
    # Covariance matrix V = A^{-1}
    covariance_matrix: np.ndarray
    
    # Accumulated feature-reward products b = sum(phi * r)
    b_vector: np.ndarray
    
    # Posterior mean theta_hat = V @ b
    theta_hat: np.ndarray
    
    # Number of times this arm was pulled
    pull_count: int = 0
    
    # Cumulative reward for this arm
    cumulative_reward: float = 0.0
    
    # For numerical stability tracking
    updates_since_recompute: int = 0


class LinearThompsonSampling:
    """
    Linear Thompson Sampling with disjoint models per arm.
    
    Each arm (specification) has its own linear model:
        r = phi^T @ theta_a + epsilon
    
    The algorithm maintains Bayesian posteriors and uses posterior
    sampling for action selection, naturally balancing exploration
    and exploitation.
    """
    
    def __init__(
        self,
        num_actions: int,
        feature_dim: int,
        prior_precision: float = 0.1,
        exploration_variance: float = 5.0,
        condition_number_threshold: float = 1e10,
        regularization_epsilon: float = 1e-6,
        recompute_interval: int = 100,
        seed: Optional[int] = None
    ):
        """
        Initialize Linear Thompson Sampling.
        
        Args:
            num_actions: Number of arms/specifications
            feature_dim: Dimension of feature vectors
            prior_precision: Lambda for prior (N(0, lambda^{-1}*I))
            exploration_variance: Variance inflation factor for exploration
            condition_number_threshold: Max condition number before regularization
            regularization_epsilon: Small value for numerical stability
            recompute_interval: Recompute covariance from precision every N updates
            seed: Random seed for reproducibility
        """
        self.num_actions = num_actions
        self.feature_dim = feature_dim
        self.prior_precision = prior_precision
        self.exploration_variance = exploration_variance
        self.condition_number_threshold = condition_number_threshold
        self.regularization_epsilon = regularization_epsilon
        self.recompute_interval = recompute_interval
        
        # Random state
        self.rng = np.random.default_rng(seed)
        
        # Initialize arm states
        self.arms: List[ArmState] = []
        self._initialize_arms()
        
        # Tracking
        self.total_rounds = 0
        self.action_history: List[int] = []
        self.reward_history: List[float] = []
    
    def _initialize_arms(self):
        """Initialize arm states with prior."""
        for _ in range(self.num_actions):
            # Prior: theta ~ N(0, lambda^{-1} * I)
            # Precision matrix starts as lambda * I
            precision = self.prior_precision * np.eye(self.feature_dim)
            covariance = (1.0 / self.prior_precision) * np.eye(self.feature_dim)
            
            arm = ArmState(
                precision_matrix=precision.copy(),
                covariance_matrix=covariance.copy(),
                b_vector=np.zeros(self.feature_dim),
                theta_hat=np.zeros(self.feature_dim),
                pull_count=0,
                cumulative_reward=0.0,
                updates_since_recompute=0
            )
            self.arms.append(arm)
    
    def select_action(
        self,
        context: np.ndarray,
        available_actions: Optional[List[int]] = None
    ) -> int:
        """
        Select action using Thompson Sampling.
        
        Samples from posterior for each arm and selects the action
        with highest expected reward under sampled parameters.
        
        Args:
            context: Feature vector phi (shape: feature_dim,)
            available_actions: List of available action indices (None = all)
        
        Returns:
            Selected action index
        """
        context = np.asarray(context).flatten()
        assert len(context) == self.feature_dim, \
            f"Context dim {len(context)} != feature_dim {self.feature_dim}"
        
        if available_actions is None:
            available_actions = list(range(self.num_actions))
        
        if len(available_actions) == 0:
            raise ValueError("No available actions")
        
        if len(available_actions) == 1:
            return available_actions[0]
        
        # Sample from posterior for each available arm
        sampled_rewards = []
        for action in available_actions:
            arm = self.arms[action]
            
            # Sample theta ~ N(theta_hat, sigma^2_explore * V)
            sampled_theta = self._sample_posterior(arm)
            
            # Expected reward under sampled parameters
            expected_reward = context @ sampled_theta
            sampled_rewards.append(expected_reward)
        
        # Select action with highest sampled expected reward
        best_idx = np.argmax(sampled_rewards)
        selected_action = available_actions[best_idx]
        
        return selected_action
    
    def _sample_posterior(self, arm: ArmState) -> np.ndarray:
        """
        Sample from the posterior distribution of arm parameters.
        
        Uses variance inflation for exploration:
            theta ~ N(theta_hat, sigma^2_explore * V)
        
        Args:
            arm: Arm state containing posterior parameters
        
        Returns:
            Sampled parameter vector
        """
        # Ensure covariance is positive definite
        try:
            # Cholesky decomposition for stable sampling
            inflated_cov = self.exploration_variance * arm.covariance_matrix
            L = np.linalg.cholesky(inflated_cov)
            z = self.rng.standard_normal(self.feature_dim)
            sampled_theta = arm.theta_hat + L @ z
        except np.linalg.LinAlgError:
            # Fallback: add regularization and use eigendecomposition
            # warnings.warn("Cholesky failed, using eigendecomposition")
            inflated_cov = self.exploration_variance * arm.covariance_matrix
            inflated_cov += self.regularization_epsilon * np.eye(self.feature_dim)
            
            eigvals, eigvecs = np.linalg.eigh(inflated_cov)
            eigvals = np.maximum(eigvals, self.regularization_epsilon)
            sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            z = self.rng.standard_normal(self.feature_dim)
            sampled_theta = arm.theta_hat + sqrt_cov @ z
        
        return sampled_theta
    
    def update(
        self,
        action: int,
        context: np.ndarray,
        reward: float
    ):
        """
        Update the posterior for the selected arm.
        
        Uses Sherman-Morrison formula for O(d^2) update:
            V_{t+1} = V_t - (V_t @ phi @ phi^T @ V_t) / (1 + phi^T @ V_t @ phi)
        
        Args:
            action: Selected action index
            context: Feature vector used for selection
            reward: Observed reward
        """
        context = np.asarray(context).flatten()
        arm = self.arms[action]
        
        # Update precision matrix: A += phi @ phi^T
        arm.precision_matrix += np.outer(context, context)
        
        # Sherman-Morrison update for covariance
        # V_{t+1} = V_t - (V_t @ phi @ phi^T @ V_t) / (1 + phi^T @ V_t @ phi)
        Vphi = arm.covariance_matrix @ context
        denom = 1.0 + context @ Vphi
        
        if denom > self.regularization_epsilon:
            arm.covariance_matrix -= np.outer(Vphi, Vphi) / denom
        else:
            # Numerical issue - recompute from precision
            self._recompute_covariance(arm)
        
        # Update b vector: b += phi * r
        arm.b_vector += context * reward
        
        # Update posterior mean: theta_hat = V @ b
        arm.theta_hat = arm.covariance_matrix @ arm.b_vector
        
        # Update tracking
        arm.pull_count += 1
        arm.cumulative_reward += reward
        arm.updates_since_recompute += 1
        
        # Periodic recomputation for numerical stability
        if arm.updates_since_recompute >= self.recompute_interval:
            self._check_and_recompute(arm)
        
        # Global tracking
        self.total_rounds += 1
        self.action_history.append(action)
        self.reward_history.append(reward)
    
    def _recompute_covariance(self, arm: ArmState):
        """Recompute covariance matrix from precision matrix."""
        try:
            arm.covariance_matrix = np.linalg.inv(arm.precision_matrix)
        except np.linalg.LinAlgError:
            # Add regularization
            reg_precision = arm.precision_matrix + \
                self.regularization_epsilon * np.eye(self.feature_dim)
            arm.covariance_matrix = np.linalg.inv(reg_precision)
        
        arm.updates_since_recompute = 0
    
    def _check_and_recompute(self, arm: ArmState):
        """Check condition number and recompute if necessary."""
        try:
            cond = np.linalg.cond(arm.covariance_matrix)
            if cond > self.condition_number_threshold:
                self._recompute_covariance(arm)
        except:
            self._recompute_covariance(arm)
    
    def get_action_statistics(self) -> Dict[str, Any]:
        """Get statistics about action selection."""
        stats = {
            'total_rounds': self.total_rounds,
            'pull_counts': [arm.pull_count for arm in self.arms],
            'cumulative_rewards': [arm.cumulative_reward for arm in self.arms],
            'mean_rewards': [
                arm.cumulative_reward / max(arm.pull_count, 1) 
                for arm in self.arms
            ],
            'theta_norms': [np.linalg.norm(arm.theta_hat) for arm in self.arms],
        }
        return stats
    
    def get_expected_rewards(self, context: np.ndarray) -> np.ndarray:
        """
        Get expected rewards for all actions given context.
        
        Uses posterior mean (no sampling).
        
        Args:
            context: Feature vector
        
        Returns:
            Expected reward for each action
        """
        context = np.asarray(context).flatten()
        expected = np.array([
            context @ arm.theta_hat for arm in self.arms
        ])
        return expected
    
    def get_posterior_uncertainty(self, context: np.ndarray) -> np.ndarray:
        """
        Get posterior standard deviation for each action.
        
        Args:
            context: Feature vector
        
        Returns:
            Posterior std for each action
        """
        context = np.asarray(context).flatten()
        uncertainties = np.array([
            np.sqrt(context @ arm.covariance_matrix @ context)
            for arm in self.arms
        ])
        return uncertainties
    
    def save_state(self) -> Dict[str, Any]:
        """Save bandit state for checkpointing."""
        state = {
            'num_actions': self.num_actions,
            'feature_dim': self.feature_dim,
            'prior_precision': self.prior_precision,
            'exploration_variance': self.exploration_variance,
            'total_rounds': self.total_rounds,
            'arms': [
                {
                    'precision_matrix': arm.precision_matrix.tolist(),
                    'covariance_matrix': arm.covariance_matrix.tolist(),
                    'b_vector': arm.b_vector.tolist(),
                    'theta_hat': arm.theta_hat.tolist(),
                    'pull_count': arm.pull_count,
                    'cumulative_reward': arm.cumulative_reward,
                }
                for arm in self.arms
            ],
            'action_history': self.action_history,
            'reward_history': self.reward_history,
        }
        return state
    
    def load_state(self, state: Dict[str, Any]):
        """Load bandit state from checkpoint."""
        self.total_rounds = state['total_rounds']
        self.action_history = state['action_history']
        self.reward_history = state['reward_history']
        
        for i, arm_state in enumerate(state['arms']):
            self.arms[i].precision_matrix = np.array(arm_state['precision_matrix'])
            self.arms[i].covariance_matrix = np.array(arm_state['covariance_matrix'])
            self.arms[i].b_vector = np.array(arm_state['b_vector'])
            self.arms[i].theta_hat = np.array(arm_state['theta_hat'])
            self.arms[i].pull_count = arm_state['pull_count']
            self.arms[i].cumulative_reward = arm_state['cumulative_reward']


class DisjointLinearTS(LinearThompsonSampling):
    """
    Alias for LinearThompsonSampling with disjoint models.
    
    Each action has independent parameters (no parameter sharing).
    This is the model used in the dissertation.
    """
    pass


class HybridLinearTS(LinearThompsonSampling):
    """
    Hybrid Linear Thompson Sampling with shared + action-specific features.
    
    Model: r = phi_shared^T @ theta_shared + phi_action^T @ theta_action
    
    This variant can improve sample efficiency when actions share
    some common structure.
    """
    
    def __init__(
        self,
        num_actions: int,
        shared_dim: int,
        action_dim: int,
        **kwargs
    ):
        """
        Initialize Hybrid Linear TS.
        
        Args:
            num_actions: Number of arms
            shared_dim: Dimension of shared features
            action_dim: Dimension of action-specific features
            **kwargs: Additional arguments for base class
        """
        self.shared_dim = shared_dim
        self.action_dim = action_dim
        
        # Total feature dimension = shared + action-specific
        total_dim = shared_dim + action_dim
        super().__init__(num_actions, total_dim, **kwargs)
    
    def build_context(
        self,
        shared_features: np.ndarray,
        action_features: np.ndarray
    ) -> np.ndarray:
        """
        Build full context from shared and action features.
        
        Args:
            shared_features: Features common to all actions
            action_features: Action-specific features
        
        Returns:
            Concatenated feature vector
        """
        return np.concatenate([shared_features, action_features])


if __name__ == "__main__":
    # Test the implementation
    np.random.seed(42)
    
    # Simple test case
    num_actions = 4
    feature_dim = 5
    num_rounds = 100
    
    # True parameters (unknown to agent)
    true_thetas = np.random.randn(num_actions, feature_dim)
    
    # Create bandit
    bandit = LinearThompsonSampling(
        num_actions=num_actions,
        feature_dim=feature_dim,
        prior_precision=0.1,
        exploration_variance=5.0,
        seed=42
    )
    
    # Run simulation
    cumulative_regret = 0.0
    regrets = []
    
    for t in range(num_rounds):
        # Generate context
        context = np.random.randn(feature_dim)
        context[0] = 1.0  # Bias term
        
        # Oracle: best action
        true_rewards = [context @ theta for theta in true_thetas]
        best_action = np.argmax(true_rewards)
        best_reward = true_rewards[best_action]
        
        # Agent selects action
        selected_action = bandit.select_action(context)
        
        # Generate reward (true reward + noise)
        true_reward = context @ true_thetas[selected_action]
        noise = np.random.randn() * 0.1
        observed_reward = true_reward + noise
        
        # Update bandit
        bandit.update(selected_action, context, observed_reward)
        
        # Track regret
        regret = best_reward - true_reward
        cumulative_regret += regret
        regrets.append(cumulative_regret)
    
    print("Linear Thompson Sampling Test")
    print(f"Total rounds: {num_rounds}")
    print(f"Cumulative regret: {cumulative_regret:.2f}")
    print(f"Average regret: {cumulative_regret / num_rounds:.4f}")
    
    stats = bandit.get_action_statistics()
    print(f"\nPull counts: {stats['pull_counts']}")
    print(f"Mean rewards: {[f'{r:.3f}' for r in stats['mean_rewards']]}")
