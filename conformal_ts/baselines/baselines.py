"""
Baseline Methods for Comparison.

Implements several baselines for comparing against Conformal Thompson Sampling:

1. LightGBM Quantile Regression - State-of-the-art for M5
2. Fixed Specification - Best single spec from validation
3. Equal-Weight Ensemble - Average across all specifications  
4. Oracle - Hindsight optimal (upper bound)
5. Random - Random specification selection
6. Simple Quantile - Online quantile regression without bandit

These match what top M5/GEFCom competitors used.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseBaseline(ABC):
    """Abstract base class for baselines."""
    
    @abstractmethod
    def fit(self, contexts: np.ndarray, targets: np.ndarray, **kwargs):
        """Fit the baseline model."""
        pass
    
    @abstractmethod
    def predict(self, context: np.ndarray) -> Tuple[float, float]:
        """Predict interval (lower, upper)."""
        pass
    
    @abstractmethod
    def update(self, context: np.ndarray, target: float):
        """Online update (if applicable)."""
        pass


class LightGBMQuantileBaseline(BaseBaseline):
    """
    LightGBM Quantile Regression baseline.
    
    This was the backbone of many top M5 solutions.
    Trains separate models for lower and upper quantiles.
    """
    
    def __init__(
        self,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        n_estimators: int = 100,
        min_child_samples: int = 20,
        seed: int = 42
    ):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        
        self.lgb_params = {
            'objective': 'quantile',
            'metric': 'quantile',
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'min_child_samples': min_child_samples,
            'verbose': -1,
            'seed': seed,
        }
        
        self.lower_model = None
        self.upper_model = None
        self._fitted = False
    
    def fit(
        self,
        contexts: np.ndarray,
        targets: np.ndarray,
        val_contexts: Optional[np.ndarray] = None,
        val_targets: Optional[np.ndarray] = None
    ):
        """
        Fit LightGBM quantile models.
        
        Args:
            contexts: Training features (n_samples, n_features)
            targets: Training targets (n_samples,)
            val_contexts: Optional validation features
            val_targets: Optional validation targets
        """
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("lightgbm not installed. Run: pip install lightgbm")
            raise
        
        logger.info(f"Fitting LightGBM quantile models on {len(targets)} samples")
        
        # Lower quantile model
        lower_params = {**self.lgb_params, 'alpha': self.lower_quantile}
        self.lower_model = lgb.LGBMRegressor(**lower_params)
        
        if val_contexts is not None:
            self.lower_model.fit(
                contexts, targets,
                eval_set=[(val_contexts, val_targets)],
            )
        else:
            self.lower_model.fit(contexts, targets)
        
        # Upper quantile model
        upper_params = {**self.lgb_params, 'alpha': self.upper_quantile}
        self.upper_model = lgb.LGBMRegressor(**upper_params)
        
        if val_contexts is not None:
            self.upper_model.fit(
                contexts, targets,
                eval_set=[(val_contexts, val_targets)],
            )
        else:
            self.upper_model.fit(contexts, targets)
        
        self._fitted = True
        logger.info("LightGBM models fitted successfully")
    
    def predict(self, context: np.ndarray) -> Tuple[float, float]:
        """Predict interval."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        context = np.atleast_2d(context)
        
        lower = float(self.lower_model.predict(context)[0])
        upper = float(self.upper_model.predict(context)[0])
        
        # Ensure valid interval
        if lower > upper:
            lower, upper = upper, lower
        
        return lower, upper
    
    def update(self, context: np.ndarray, target: float):
        """LightGBM doesn't support online updates - no-op."""
        pass
    
    def predict_quantiles(
        self,
        context: np.ndarray,
        quantiles: List[float]
    ) -> np.ndarray:
        """
        Predict multiple quantiles (requires refitting for each).
        
        For full quantile predictions, train separate models.
        Here we interpolate from lower/upper as approximation.
        """
        context = np.atleast_2d(context)
        
        lower = self.lower_model.predict(context)[0]
        upper = self.upper_model.predict(context)[0]
        
        # Linear interpolation (crude approximation)
        predictions = []
        for q in quantiles:
            if q <= self.lower_quantile:
                pred = lower
            elif q >= self.upper_quantile:
                pred = upper
            else:
                # Interpolate
                frac = (q - self.lower_quantile) / (self.upper_quantile - self.lower_quantile)
                pred = lower + frac * (upper - lower)
            predictions.append(pred)
        
        return np.array(predictions)


class MultiQuantileLightGBM(BaseBaseline):
    """
    LightGBM with multiple quantile models for full distribution.
    
    Trains separate model for each quantile level.
    """
    
    def __init__(
        self,
        quantiles: List[float] = None,
        n_estimators: int = 100,
        seed: int = 42
    ):
        self.quantiles = quantiles or [0.05, 0.25, 0.5, 0.75, 0.95]
        self.n_estimators = n_estimators
        self.seed = seed
        
        self.models: Dict[float, Any] = {}
        self._fitted = False
    
    def fit(self, contexts: np.ndarray, targets: np.ndarray, **kwargs):
        """Fit model for each quantile."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm required")
        
        for q in self.quantiles:
            logger.info(f"Fitting quantile {q:.2f}")
            
            model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=q,
                n_estimators=self.n_estimators,
                verbose=-1,
                seed=self.seed
            )
            model.fit(contexts, targets)
            self.models[q] = model
        
        self._fitted = True
    
    def predict(self, context: np.ndarray) -> Tuple[float, float]:
        """Predict 90% interval using 5th and 95th percentile models."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        
        context = np.atleast_2d(context)
        
        lower = float(self.models[0.05].predict(context)[0])
        upper = float(self.models[0.95].predict(context)[0])
        
        return min(lower, upper), max(lower, upper)
    
    def predict_all_quantiles(self, context: np.ndarray) -> Dict[float, float]:
        """Predict all trained quantiles."""
        context = np.atleast_2d(context)
        return {q: float(m.predict(context)[0]) for q, m in self.models.items()}
    
    def update(self, context: np.ndarray, target: float):
        """No online update for LightGBM."""
        pass


class FixedSpecificationBaseline(BaseBaseline):
    """
    Fixed specification baseline.
    
    Uses a single specification throughout, selected based on
    validation performance.
    """
    
    def __init__(
        self,
        specification: int,
        feature_dim: int,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95,
        learning_rate: float = 0.01
    ):
        self.specification = specification
        self.feature_dim = feature_dim
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.learning_rate = learning_rate
        
        # Online quantile regression weights
        self.lower_weights = np.zeros(feature_dim)
        self.upper_weights = np.zeros(feature_dim)
        
        self._n_updates = 0
    
    def fit(self, contexts: np.ndarray, targets: np.ndarray, **kwargs):
        """Fit via batch gradient descent on training data."""
        for i in range(len(contexts)):
            self.update(contexts[i], targets[i])
    
    def predict(self, context: np.ndarray) -> Tuple[float, float]:
        """Predict interval."""
        context = np.asarray(context).flatten()
        
        lower = float(np.dot(context, self.lower_weights))
        upper = float(np.dot(context, self.upper_weights))
        
        # Ensure valid interval
        return min(lower, upper), max(lower, upper)
    
    def update(self, context: np.ndarray, target: float):
        """Online update using pinball loss gradient."""
        context = np.asarray(context).flatten()
        
        # Lower quantile update
        lower_pred = np.dot(context, self.lower_weights)
        lower_grad = context * (self.lower_quantile - (target < lower_pred))
        self.lower_weights -= self.learning_rate * lower_grad
        
        # Upper quantile update
        upper_pred = np.dot(context, self.upper_weights)
        upper_grad = context * (self.upper_quantile - (target < upper_pred))
        self.upper_weights -= self.learning_rate * upper_grad
        
        self._n_updates += 1


class EqualWeightEnsemble(BaseBaseline):
    """
    Equal-weight ensemble across all specifications.
    
    Maintains separate quantile models for each specification
    and averages predictions.
    """
    
    def __init__(
        self,
        num_specifications: int,
        feature_dim: int,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95,
        learning_rate: float = 0.01
    ):
        self.num_specifications = num_specifications
        self.feature_dim = feature_dim
        
        # Create one model per specification
        self.models = [
            FixedSpecificationBaseline(
                specification=i,
                feature_dim=feature_dim,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
                learning_rate=learning_rate
            )
            for i in range(num_specifications)
        ]
    
    def fit(self, contexts: np.ndarray, targets: np.ndarray, **kwargs):
        """Fit all specification models."""
        for model in self.models:
            model.fit(contexts, targets)
    
    def predict(self, context: np.ndarray) -> Tuple[float, float]:
        """Predict by averaging across specifications."""
        lowers = []
        uppers = []
        
        for model in self.models:
            l, u = model.predict(context)
            lowers.append(l)
            uppers.append(u)
        
        # Average intervals
        return float(np.mean(lowers)), float(np.mean(uppers))
    
    def update(self, context: np.ndarray, target: float):
        """Update all models."""
        for model in self.models:
            model.update(context, target)


class OracleBaseline(BaseBaseline):
    """
    Oracle baseline with hindsight optimal specification selection.
    
    This provides an upper bound on performance - selects the
    specification that would have had the best interval score
    for each prediction.
    
    Note: Can only be computed in hindsight, not used for actual prediction.
    """
    
    def __init__(
        self,
        num_specifications: int,
        feature_dim: int
    ):
        self.num_specifications = num_specifications
        self.feature_dim = feature_dim
        
        # Track performance per specification
        self.spec_scores: Dict[int, List[float]] = {
            i: [] for i in range(num_specifications)
        }
        
        # Models per specification
        self.models = [
            FixedSpecificationBaseline(spec, feature_dim)
            for spec in range(num_specifications)
        ]
    
    def fit(self, contexts: np.ndarray, targets: np.ndarray, **kwargs):
        """Fit all specification models."""
        for model in self.models:
            model.fit(contexts, targets)
    
    def predict(self, context: np.ndarray) -> Tuple[float, float]:
        """
        Oracle prediction - returns best interval in hindsight.
        
        In practice, returns the prediction from the best-performing
        specification on recent data.
        """
        # Use specification with best recent performance
        best_spec = 0
        best_avg_score = float('inf')
        
        for spec, scores in self.spec_scores.items():
            if scores:
                avg = np.mean(scores[-100:])  # Last 100 scores
                if avg < best_avg_score:
                    best_avg_score = avg
                    best_spec = spec
        
        return self.models[best_spec].predict(context)
    
    def update(self, context: np.ndarray, target: float):
        """Update all models and track their scores."""
        from conformal_ts.evaluation.metrics import interval_score
        
        for spec, model in enumerate(self.models):
            lower, upper = model.predict(context)
            score = interval_score(
                np.array([lower]),
                np.array([upper]),
                np.array([target])
            )[0]
            self.spec_scores[spec].append(score)
            model.update(context, target)
    
    def get_oracle_score(
        self,
        contexts: np.ndarray,
        targets: np.ndarray,
        spec_for_each: np.ndarray
    ) -> float:
        """
        Compute oracle score given hindsight-optimal specifications.
        
        Args:
            contexts: Feature matrix
            targets: True targets
            spec_for_each: Optimal specification for each sample
            
        Returns:
            Average interval score with oracle selection
        """
        from conformal_ts.evaluation.metrics import interval_score
        
        scores = []
        for i, (ctx, target, spec) in enumerate(zip(contexts, targets, spec_for_each)):
            lower, upper = self.models[spec].predict(ctx)
            score = interval_score(
                np.array([lower]),
                np.array([upper]),
                np.array([target])
            )[0]
            scores.append(score)
        
        return float(np.mean(scores))


class RandomSpecificationBaseline(BaseBaseline):
    """
    Random specification selection baseline.
    
    Randomly selects specification at each step.
    """
    
    def __init__(
        self,
        num_specifications: int,
        feature_dim: int,
        seed: int = 42
    ):
        self.num_specifications = num_specifications
        self.feature_dim = feature_dim
        self.rng = np.random.default_rng(seed)
        
        self.models = [
            FixedSpecificationBaseline(spec, feature_dim)
            for spec in range(num_specifications)
        ]
    
    def fit(self, contexts: np.ndarray, targets: np.ndarray, **kwargs):
        """Fit all models."""
        for model in self.models:
            model.fit(contexts, targets)
    
    def predict(self, context: np.ndarray) -> Tuple[float, float]:
        """Randomly select specification and predict."""
        spec = self.rng.integers(0, self.num_specifications)
        return self.models[spec].predict(context)
    
    def update(self, context: np.ndarray, target: float):
        """Update all models."""
        for model in self.models:
            model.update(context, target)


class AdaptiveConformalInference(BaseBaseline):
    """
    Adaptive Conformal Inference (ACI) baseline.
    
    Implements Gibbs & Candès (2021) ACI which adapts the
    miscoverage rate to maintain coverage under distribution shift.
    
    This is the main conformal-only baseline without bandit selection.
    """
    
    def __init__(
        self,
        feature_dim: int,
        target_coverage: float = 0.90,
        gamma: float = 0.01,  # Learning rate for alpha adaptation
        calibration_window: int = 250,
        learning_rate: float = 0.01
    ):
        self.feature_dim = feature_dim
        self.target_coverage = target_coverage
        self.target_alpha = 1 - target_coverage
        self.gamma = gamma
        self.calibration_window = calibration_window
        
        # Current miscoverage rate
        self.alpha_t = self.target_alpha
        
        # Quantile regression model
        self.lower_quantile = self.target_alpha / 2
        self.upper_quantile = 1 - self.target_alpha / 2
        
        self.lower_weights = np.zeros(feature_dim)
        self.upper_weights = np.zeros(feature_dim)
        self.learning_rate = learning_rate
        
        # Calibration scores
        self.conformity_scores: List[float] = []
        
        # Coverage tracking
        self.coverage_history: List[bool] = []
    
    def fit(self, contexts: np.ndarray, targets: np.ndarray, **kwargs):
        """Fit initial model and build calibration set."""
        for i in range(len(contexts)):
            self.update(contexts[i], targets[i])
    
    def predict(self, context: np.ndarray) -> Tuple[float, float]:
        """Predict with adaptive coverage."""
        context = np.asarray(context).flatten()
        
        # Raw quantile predictions
        lower_raw = np.dot(context, self.lower_weights)
        upper_raw = np.dot(context, self.upper_weights)
        
        # Conformal correction
        if len(self.conformity_scores) >= 20:
            # Get conformalization quantile at current alpha
            scores = np.array(self.conformity_scores[-self.calibration_window:])
            correction = np.quantile(scores, 1 - self.alpha_t)
        else:
            correction = 0.0
        
        lower = lower_raw - correction
        upper = upper_raw + correction
        
        return float(min(lower, upper)), float(max(lower, upper))
    
    def update(self, context: np.ndarray, target: float):
        """Update model and adapt alpha."""
        context = np.asarray(context).flatten()
        
        # Get current prediction
        lower, upper = self.predict(context)
        
        # Check coverage
        covered = lower <= target <= upper
        self.coverage_history.append(covered)
        
        # Update alpha using ACI rule
        # If we covered, decrease alpha (tighten intervals)
        # If we missed, increase alpha (widen intervals)
        err_t = 1 - int(covered)
        self.alpha_t = self.alpha_t + self.gamma * (self.target_alpha - err_t)
        self.alpha_t = np.clip(self.alpha_t, 0.01, 0.5)
        
        # Compute conformity score
        lower_raw = np.dot(context, self.lower_weights)
        upper_raw = np.dot(context, self.upper_weights)
        conformity = max(lower_raw - target, target - upper_raw)
        self.conformity_scores.append(conformity)
        
        # Keep window size
        if len(self.conformity_scores) > self.calibration_window:
            self.conformity_scores = self.conformity_scores[-self.calibration_window:]
        
        # Update quantile regression
        lower_pred = np.dot(context, self.lower_weights)
        lower_grad = context * (self.lower_quantile - (target < lower_pred))
        self.lower_weights -= self.learning_rate * lower_grad
        
        upper_pred = np.dot(context, self.upper_weights)
        upper_grad = context * (self.upper_quantile - (target < upper_pred))
        self.upper_weights -= self.learning_rate * upper_grad


def create_baselines(
    num_specifications: int,
    feature_dim: int,
    use_lightgbm: bool = True,
    seed: int = 42
) -> Dict[str, BaseBaseline]:
    """
    Create standard set of baselines for comparison.
    
    Args:
        num_specifications: Number of specifications in bandit
        feature_dim: Feature dimension
        use_lightgbm: Include LightGBM baseline (requires package)
        seed: Random seed
        
    Returns:
        Dictionary of baseline name -> baseline instance
    """
    baselines = {
        'fixed_best': FixedSpecificationBaseline(
            specification=0,  # Will be set to best on validation
            feature_dim=feature_dim
        ),
        'ensemble': EqualWeightEnsemble(
            num_specifications=num_specifications,
            feature_dim=feature_dim
        ),
        'random': RandomSpecificationBaseline(
            num_specifications=num_specifications,
            feature_dim=feature_dim,
            seed=seed
        ),
        'aci': AdaptiveConformalInference(
            feature_dim=feature_dim,
            target_coverage=0.90
        ),
    }
    
    if use_lightgbm:
        try:
            import lightgbm
            baselines['lightgbm'] = LightGBMQuantileBaseline(seed=seed)
        except ImportError:
            logger.warning("LightGBM not available, skipping baseline")
    
    return baselines


if __name__ == "__main__":
    # Test baselines
    logging.basicConfig(level=logging.INFO)
    
    np.random.seed(42)
    
    # Generate test data
    n_train = 1000
    n_test = 200
    feature_dim = 10
    
    X_train = np.random.randn(n_train, feature_dim)
    y_train = X_train[:, 0] * 2 + X_train[:, 1] + np.random.randn(n_train) * 0.5
    
    X_test = np.random.randn(n_test, feature_dim)
    y_test = X_test[:, 0] * 2 + X_test[:, 1] + np.random.randn(n_test) * 0.5
    
    # Test each baseline
    baselines = create_baselines(
        num_specifications=8,
        feature_dim=feature_dim,
        use_lightgbm=True
    )
    
    for name, baseline in baselines.items():
        print(f"\n{'='*50}")
        print(f"Testing {name}")
        print('='*50)
        
        # Fit
        baseline.fit(X_train, y_train)
        
        # Evaluate
        coverages = []
        widths = []
        
        for x, y in zip(X_test, y_test):
            lower, upper = baseline.predict(x)
            coverages.append(lower <= y <= upper)
            widths.append(upper - lower)
            baseline.update(x, y)
        
        print(f"Coverage: {np.mean(coverages):.2%}")
        print(f"Mean width: {np.mean(widths):.3f}")
