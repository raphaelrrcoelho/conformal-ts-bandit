"""
Conformalized Quantile Regression (CQR).

This module implements CQR with:
- Online quantile regression with pinball loss
- Conformal calibration for distribution-free coverage
- Rolling calibration window for non-stationarity
- Warm-up bounds for initial stability

Reference:
    Romano, Patterson, Candès (2019) - Conformalized Quantile Regression
    Barber et al. (2023) - Conformal prediction beyond exchangeability
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import deque


@dataclass
class QuantileModel:
    """State for a single quantile regression model."""
    
    # Model parameters
    weights: np.ndarray
    
    # Quantile level (e.g., 0.05 or 0.95)
    quantile: float
    
    # Learning rate
    learning_rate: float
    
    # L2 regularization
    l2_reg: float
    
    # Training statistics
    num_updates: int = 0
    cumulative_loss: float = 0.0


class OnlineQuantileRegression:
    """
    Online quantile regression using stochastic gradient descent.
    
    Minimizes pinball loss:
        L_tau(y, q) = (y - q) * [tau - I(y < q)]
    """
    
    def __init__(
        self,
        feature_dim: int,
        quantile: float,
        learning_rate: float = 0.02,
        l2_reg: float = 1e-4,
        seed: Optional[int] = None
    ):
        """
        Initialize online quantile regression.
        
        Args:
            feature_dim: Dimension of input features
            quantile: Target quantile (e.g., 0.05 for 5th percentile)
            learning_rate: SGD learning rate
            l2_reg: L2 regularization coefficient
            seed: Random seed for initialization
        """
        self.feature_dim = feature_dim
        self.quantile = quantile
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        
        rng = np.random.default_rng(seed)
        
        # Initialize weights near zero
        self.weights = rng.normal(0, 0.01, size=feature_dim)
        self.num_updates = 0
        self.cumulative_loss = 0.0
    
    def predict(self, x: np.ndarray) -> float:
        """
        Predict quantile for given features.
        
        Args:
            x: Feature vector
        
        Returns:
            Predicted quantile value
        """
        x = np.asarray(x).flatten()
        return float(x @ self.weights)
    
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Predict quantiles for batch of features.
        
        Args:
            X: Feature matrix (n_samples, feature_dim)
        
        Returns:
            Predicted quantile values
        """
        return X @ self.weights
    
    def update(self, x: np.ndarray, y: float):
        """
        Update model with single observation using SGD.
        
        Uses pinball loss gradient:
            grad = x * (tau - I(y < q_hat))
        
        Args:
            x: Feature vector
            y: True target value
        """
        x = np.asarray(x).flatten()
        
        # Current prediction
        q_hat = self.predict(x)
        
        # Pinball loss
        residual = y - q_hat
        if y < q_hat:
            loss = -residual * (self.quantile - 1)
        else:
            loss = residual * self.quantile
        
        # Gradient of pinball loss
        if y < q_hat:
            grad = x * (self.quantile - 1)
        else:
            grad = x * self.quantile
        
        # SGD update with L2 regularization
        self.weights -= self.learning_rate * (-grad + self.l2_reg * self.weights)
        
        # Tracking
        self.num_updates += 1
        self.cumulative_loss += loss
    
    def update_batch(self, X: np.ndarray, y: np.ndarray):
        """
        Update model with batch of observations.
        
        Args:
            X: Feature matrix (n_samples, feature_dim)
            y: Target values (n_samples,)
        """
        for xi, yi in zip(X, y):
            self.update(xi, yi)
    
    def get_state(self) -> Dict[str, Any]:
        """Get model state for checkpointing."""
        return {
            'weights': self.weights.tolist(),
            'quantile': self.quantile,
            'learning_rate': self.learning_rate,
            'l2_reg': self.l2_reg,
            'num_updates': self.num_updates,
            'cumulative_loss': self.cumulative_loss,
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load model state from checkpoint."""
        self.weights = np.array(state['weights'])
        self.num_updates = state['num_updates']
        self.cumulative_loss = state['cumulative_loss']


class ConformizedQuantileRegression:
    """
    Conformalized Quantile Regression for prediction intervals.
    
    Maintains separate models for lower and upper quantiles,
    then applies conformal calibration to achieve target coverage.
    """
    
    def __init__(
        self,
        feature_dim: int,
        coverage_target: float = 0.90,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95,
        learning_rate: float = 0.02,
        l2_reg: float = 1e-4,
        calibration_window: int = 250,
        warmup_min_obs: int = 20,
        warmup_interval_min: float = 0.002,
        warmup_interval_max: float = 0.05,
        seed: Optional[int] = None
    ):
        """
        Initialize CQR.
        
        Args:
            feature_dim: Dimension of input features
            coverage_target: Target coverage probability (e.g., 0.90)
            lower_quantile: Lower quantile level (e.g., 0.05)
            upper_quantile: Upper quantile level (e.g., 0.95)
            learning_rate: SGD learning rate for quantile models
            l2_reg: L2 regularization coefficient
            calibration_window: Rolling window size for calibration
            warmup_min_obs: Minimum observations before calibration
            warmup_interval_min: Minimum interval width during warmup
            warmup_interval_max: Maximum interval width during warmup
            seed: Random seed
        """
        self.feature_dim = feature_dim
        self.coverage_target = coverage_target
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.calibration_window = calibration_window
        self.warmup_min_obs = warmup_min_obs
        self.warmup_interval_min = warmup_interval_min
        self.warmup_interval_max = warmup_interval_max
        
        rng = np.random.default_rng(seed)
        
        # Quantile regression models
        self.lower_model = OnlineQuantileRegression(
            feature_dim, lower_quantile, learning_rate, l2_reg,
            seed=rng.integers(0, 2**31)
        )
        self.upper_model = OnlineQuantileRegression(
            feature_dim, upper_quantile, learning_rate, l2_reg,
            seed=rng.integers(0, 2**31)
        )
        
        # Calibration buffer for nonconformity scores
        self.calibration_buffer: deque = deque(maxlen=calibration_window)
        
        # Cached calibration quantile
        self._calibration_quantile: Optional[float] = None
        self._buffer_dirty: bool = True
        
        # Statistics
        self.num_predictions = 0
        self.num_calibration_updates = 0
    
    def predict_raw(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Get raw quantile predictions (before calibration).
        
        Args:
            x: Feature vector
        
        Returns:
            (lower_quantile, upper_quantile) predictions
        """
        lower = self.lower_model.predict(x)
        upper = self.upper_model.predict(x)
        
        # Ensure lower <= upper
        if lower > upper:
            mid = (lower + upper) / 2
            lower, upper = mid, mid
        
        return lower, upper
    
    def predict_interval(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Get calibrated prediction interval.
        
        Expands raw quantile predictions by calibration factor:
            [q_lower - Q, q_upper + Q]
        
        where Q is the (1-alpha) quantile of nonconformity scores.
        
        Args:
            x: Feature vector
        
        Returns:
            (lower_bound, upper_bound) calibrated interval
        """
        lower_raw, upper_raw = self.predict_raw(x)
        
        # Get calibration adjustment
        Q = self._get_calibration_quantile()
        
        # Apply calibration
        lower = lower_raw - Q
        upper = upper_raw + Q
        
        # During warmup, apply bounds
        if len(self.calibration_buffer) < self.warmup_min_obs:
            width = upper - lower
            if width < self.warmup_interval_min:
                expansion = (self.warmup_interval_min - width) / 2
                lower -= expansion
                upper += expansion
            elif width > self.warmup_interval_max:
                shrinkage = (width - self.warmup_interval_max) / 2
                lower += shrinkage
                upper -= shrinkage
        
        self.num_predictions += 1
        return lower, upper
    
    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get calibrated intervals for batch of features.
        
        Args:
            X: Feature matrix (n_samples, feature_dim)
        
        Returns:
            (lower_bounds, upper_bounds) arrays
        """
        lower_raw = self.lower_model.predict_batch(X)
        upper_raw = self.upper_model.predict_batch(X)
        
        # Ensure lower <= upper
        swap_mask = lower_raw > upper_raw
        if swap_mask.any():
            mid = (lower_raw[swap_mask] + upper_raw[swap_mask]) / 2
            lower_raw[swap_mask] = mid
            upper_raw[swap_mask] = mid
        
        Q = self._get_calibration_quantile()
        
        return lower_raw - Q, upper_raw + Q
    
    def update(self, x: np.ndarray, y: float):
        """
        Update models and calibration with new observation.
        
        Args:
            x: Feature vector
            y: True target value
        """
        # Get predictions before update (for conformity score)
        lower_pred, upper_pred = self.predict_raw(x)
        
        # Compute nonconformity score
        # R = max(q_lower - y, y - q_upper)
        score = max(lower_pred - y, y - upper_pred)
        
        # Update calibration buffer
        self.calibration_buffer.append(score)
        self._buffer_dirty = True
        self.num_calibration_updates += 1
        
        # Update quantile models
        self.lower_model.update(x, y)
        self.upper_model.update(x, y)
    
    def update_batch(self, X: np.ndarray, y: np.ndarray):
        """
        Update models with batch of observations.
        
        Args:
            X: Feature matrix
            y: Target values
        """
        for xi, yi in zip(X, y):
            self.update(xi, yi)
    
    def _get_calibration_quantile(self) -> float:
        """
        Get the calibration quantile from nonconformity scores.
        
        Returns the (1-alpha) quantile of scores in buffer.
        """
        if not self._buffer_dirty and self._calibration_quantile is not None:
            return self._calibration_quantile
        
        if len(self.calibration_buffer) == 0:
            return 0.0
        
        scores = np.array(self.calibration_buffer)
        n = len(scores)
        
        # Finite-sample correction: use (n+1)*(1-alpha) index
        # Following Romano et al. (2019)
        k = int(np.ceil((n + 1) * self.coverage_target))
        k = min(k, n)
        
        # Get k-th order statistic
        sorted_scores = np.sort(scores)
        self._calibration_quantile = sorted_scores[k - 1]
        self._buffer_dirty = False
        
        return self._calibration_quantile
    
    def get_coverage_statistics(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute coverage statistics on held-out data.
        
        Args:
            X: Feature matrix
            y: True target values
        
        Returns:
            Dictionary with coverage, width, and score statistics
        """
        lower, upper = self.predict_batch(X)
        
        # Coverage
        covered = (y >= lower) & (y <= upper)
        coverage = np.mean(covered)
        
        # Width statistics
        widths = upper - lower
        mean_width = np.mean(widths)
        median_width = np.median(widths)
        
        # Interval scores
        scores = self._compute_interval_scores(lower, upper, y)
        mean_score = np.mean(scores)
        
        return {
            'coverage': coverage,
            'mean_width': mean_width,
            'median_width': median_width,
            'mean_interval_score': mean_score,
            'num_samples': len(y),
        }
    
    def _compute_interval_scores(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        y: np.ndarray,
        alpha: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute interval scores.
        
        IS(l, u, y) = (u - l) + (2/alpha) * [(l - y)*I(y < l) + (y - u)*I(y > u)]
        
        Args:
            lower: Lower bounds
            upper: Upper bounds
            y: True values
            alpha: Miscoverage rate (default: 1 - coverage_target)
        
        Returns:
            Interval scores for each sample
        """
        if alpha is None:
            alpha = 1 - self.coverage_target
        
        width = upper - lower
        
        # Penalty for below lower bound
        below = y < lower
        penalty_below = (lower - y) * below
        
        # Penalty for above upper bound
        above = y > upper
        penalty_above = (y - upper) * above
        
        scores = width + (2 / alpha) * (penalty_below + penalty_above)
        return scores
    
    def get_state(self) -> Dict[str, Any]:
        """Get full state for checkpointing."""
        return {
            'lower_model': self.lower_model.get_state(),
            'upper_model': self.upper_model.get_state(),
            'calibration_buffer': list(self.calibration_buffer),
            'num_predictions': self.num_predictions,
            'num_calibration_updates': self.num_calibration_updates,
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load state from checkpoint."""
        self.lower_model.load_state(state['lower_model'])
        self.upper_model.load_state(state['upper_model'])
        self.calibration_buffer = deque(
            state['calibration_buffer'],
            maxlen=self.calibration_window
        )
        self.num_predictions = state['num_predictions']
        self.num_calibration_updates = state['num_calibration_updates']
        self._buffer_dirty = True


class ActionSpecificCQR:
    """
    Maintains separate CQR models for each action/specification.
    
    This mirrors the disjoint structure in Linear Thompson Sampling.
    """
    
    def __init__(
        self,
        num_actions: int,
        feature_dim: int,
        coverage_target: float = 0.90,
        learning_rate: float = 0.02,
        l2_reg: float = 1e-4,
        calibration_window: int = 250,
        seed: Optional[int] = None
    ):
        """
        Initialize action-specific CQR models.
        
        Args:
            num_actions: Number of actions/specifications
            feature_dim: Dimension of input features
            coverage_target: Target coverage probability
            learning_rate: SGD learning rate
            l2_reg: L2 regularization
            calibration_window: Rolling calibration window size
            seed: Random seed
        """
        self.num_actions = num_actions
        self.feature_dim = feature_dim
        
        rng = np.random.default_rng(seed)
        
        # One CQR model per action
        self.models: List[ConformizedQuantileRegression] = []
        for _ in range(num_actions):
            model = ConformizedQuantileRegression(
                feature_dim=feature_dim,
                coverage_target=coverage_target,
                learning_rate=learning_rate,
                l2_reg=l2_reg,
                calibration_window=calibration_window,
                seed=rng.integers(0, 2**31)
            )
            self.models.append(model)
    
    def predict_interval(
        self,
        action: int,
        x: np.ndarray
    ) -> Tuple[float, float]:
        """
        Get prediction interval for specific action.
        
        Args:
            action: Action index
            x: Feature vector
        
        Returns:
            (lower, upper) calibrated interval
        """
        return self.models[action].predict_interval(x)
    
    def update(self, action: int, x: np.ndarray, y: float):
        """
        Update model for specific action.
        
        Args:
            action: Action index
            x: Feature vector
            y: True target value
        """
        self.models[action].update(x, y)
    
    def get_all_intervals(
        self,
        x: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Get intervals from all action models.
        
        Args:
            x: Feature vector
        
        Returns:
            List of (lower, upper) intervals for each action
        """
        return [model.predict_interval(x) for model in self.models]
    
    def get_state(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {
            'num_actions': self.num_actions,
            'models': [model.get_state() for model in self.models],
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load state from checkpoint."""
        for i, model_state in enumerate(state['models']):
            self.models[i].load_state(model_state)


if __name__ == "__main__":
    # Test the implementation
    np.random.seed(42)
    
    # Generate synthetic data with heteroscedastic noise
    n_samples = 500
    feature_dim = 5
    
    X = np.random.randn(n_samples, feature_dim)
    X[:, 0] = 1.0  # Bias term
    
    true_weights = np.random.randn(feature_dim)
    y = X @ true_weights + np.random.randn(n_samples) * (0.5 + 0.5 * np.abs(X[:, 1]))
    
    # Split data
    n_train = 300
    n_cal = 100
    n_test = 100
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_cal, y_cal = X[n_train:n_train+n_cal], y[n_train:n_train+n_cal]
    X_test, y_test = X[n_train+n_cal:], y[n_train+n_cal:]
    
    # Create CQR model
    cqr = ConformizedQuantileRegression(
        feature_dim=feature_dim,
        coverage_target=0.90,
        learning_rate=0.02,
        calibration_window=250,
        seed=42
    )
    
    # Train on training data
    print("Training quantile models...")
    cqr.update_batch(X_train, y_train)
    
    # Calibrate on calibration data
    print("Calibrating...")
    cqr.update_batch(X_cal, y_cal)
    
    # Evaluate on test data
    print("\nTest set evaluation:")
    stats = cqr.get_coverage_statistics(X_test, y_test)
    print(f"Coverage: {stats['coverage']:.2%} (target: 90%)")
    print(f"Mean interval width: {stats['mean_width']:.4f}")
    print(f"Mean interval score: {stats['mean_interval_score']:.4f}")
    
    # Test action-specific CQR
    print("\n--- Action-Specific CQR Test ---")
    action_cqr = ActionSpecificCQR(
        num_actions=4,
        feature_dim=feature_dim,
        coverage_target=0.90,
        seed=42
    )
    
    # Simulate different actions with different noise levels
    for t in range(n_train):
        action = t % 4
        noise_scale = 0.5 + 0.3 * action  # Different noise per action
        yi = X_train[t] @ true_weights + np.random.randn() * noise_scale
        action_cqr.update(action, X_train[t], yi)
    
    # Get intervals for all actions
    test_x = X_test[0]
    intervals = action_cqr.get_all_intervals(test_x)
    print("\nIntervals for each action:")
    for i, (lower, upper) in enumerate(intervals):
        print(f"  Action {i}: [{lower:.4f}, {upper:.4f}] (width: {upper-lower:.4f})")
