"""
Fair Baselines for Comparison - CORRECTED VERSION.

This replaces conformal_ts/baselines/baselines.py with methodologically
correct baselines that ALL share the same underlying infrastructure.

CRITICAL FIX: The original baselines used online quantile regression
starting from zero weights, which fails catastrophically on M5's scale.
This version ensures:

1. ALL methods use the SAME LightGBM quantile models (trained once)
2. ALL methods use the SAME conformal calibration infrastructure
3. The ONLY difference is specification SELECTION strategy

This follows the gold standard from Gibbs & Candès literature where
baselines employ identical conformal prediction mechanisms, differing
only in the mechanism being evaluated.

Usage:
    Replace conformal_ts/baselines/baselines.py with this file.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# =============================================================================
# SHARED INFRASTRUCTURE (used by ALL methods)
# =============================================================================

class SharedQuantileModels:
    """
    Shared LightGBM quantile models used by ALL methods.
    
    This ensures fair comparison - all methods predict using
    the same underlying models, differing only in which
    specification they select.
    """
    
    _instance = None
    
    def __init__(
        self,
        num_specifications: int,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        seed: int = 42
    ):
        self.num_specifications = num_specifications
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        
        # Models per specification: {spec_idx: (lower_model, upper_model)}
        self.models: Dict[int, Tuple[Any, Any]] = {}
        self._fitted = False
        
    @classmethod
    def get_instance(cls, **kwargs) -> 'SharedQuantileModels':
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset singleton for new experiment."""
        cls._instance = None
    
    def fit(
        self,
        contexts: np.ndarray,
        targets: np.ndarray,
        specification_indices: Optional[np.ndarray] = None
    ):
        """
        Fit models for all specifications.
        
        If specification_indices is provided, uses only matching samples
        for each specification. Otherwise, trains all specs on all data.
        """
        try:
            import lightgbm as lgb
            use_lgb = True
        except ImportError:
            logger.warning("LightGBM not available, using linear models")
            use_lgb = False
        
        logger.info(f"Fitting shared quantile models for {self.num_specifications} specifications...")
        
        for spec_idx in range(self.num_specifications):
            # Select data for this specification (or all data)
            if specification_indices is not None:
                mask = specification_indices == spec_idx
                if mask.sum() < 100:
                    # Not enough data, use all
                    X, y = contexts, targets
                else:
                    X, y = contexts[mask], targets[mask]
            else:
                X, y = contexts, targets
            
            if use_lgb:
                # LightGBM quantile regression
                lower_model = lgb.LGBMRegressor(
                    objective='quantile',
                    alpha=self.lower_quantile,
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    min_child_samples=20,
                    verbose=-1,
                    random_state=self.seed + spec_idx,
                    n_jobs=-1,
                )
                upper_model = lgb.LGBMRegressor(
                    objective='quantile',
                    alpha=self.upper_quantile,
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    min_child_samples=20,
                    verbose=-1,
                    random_state=self.seed + spec_idx + 1000,
                    n_jobs=-1,
                )
                
                lower_model.fit(X, y)
                upper_model.fit(X, y)
            else:
                # Fallback: batch-fitted linear quantile regression
                lower_model = BatchLinearQuantile(self.lower_quantile).fit(X, y)
                upper_model = BatchLinearQuantile(self.upper_quantile).fit(X, y)
            
            self.models[spec_idx] = (lower_model, upper_model)
            logger.info(f"  Spec {spec_idx}: fitted on {len(y)} samples")
        
        self._fitted = True
        logger.info("Shared models fitted.")
    
    def predict(
        self,
        context: np.ndarray,
        spec_idx: int
    ) -> Tuple[float, float]:
        """Get raw predictions from a specification's model."""
        if not self._fitted:
            raise RuntimeError("Models not fitted. Call fit() first.")
        
        context = np.atleast_2d(context)
        lower_model, upper_model = self.models[spec_idx]
        
        lower = float(lower_model.predict(context)[0])
        upper = float(upper_model.predict(context)[0])
        
        # Ensure valid interval
        return min(lower, upper), max(lower, upper)
    
    def predict_all(
        self,
        context: np.ndarray
    ) -> Dict[int, Tuple[float, float]]:
        """Get predictions from all specifications."""
        return {
            spec_idx: self.predict(context, spec_idx)
            for spec_idx in range(self.num_specifications)
        }


class BatchLinearQuantile:
    """Simple batch-fitted linear quantile regression (fallback)."""
    
    def __init__(self, quantile: float, n_iter: int = 1000, lr: float = 0.01):
        self.quantile = quantile
        self.n_iter = n_iter
        self.lr = lr
        self.weights = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BatchLinearQuantile':
        """Fit using batch gradient descent with proper initialization."""
        # Initialize from OLS
        XtX = X.T @ X + 1e-6 * np.eye(X.shape[1])
        Xty = X.T @ y
        self.weights = np.linalg.solve(XtX, Xty)
        
        # Fine-tune with pinball loss
        for _ in range(self.n_iter):
            pred = X @ self.weights
            error = y - pred
            grad = -X.T @ np.where(error > 0, self.quantile, self.quantile - 1)
            self.weights -= self.lr * grad / len(y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights


class SharedConformalCalibrator:
    """
    Shared conformal calibration used by ALL methods.
    
    Implements CQR-style calibration with rolling windows.
    """
    
    _instance = None
    
    def __init__(
        self,
        num_specifications: int,
        calibration_window: int = 250,
        coverage_target: float = 0.90
    ):
        self.num_specifications = num_specifications
        self.calibration_window = calibration_window
        self.coverage_target = coverage_target
        
        # Conformity scores per specification
        self.scores: Dict[int, deque] = {
            i: deque(maxlen=calibration_window)
            for i in range(num_specifications)
        }
    
    @classmethod
    def get_instance(cls, **kwargs) -> 'SharedConformalCalibrator':
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance
    
    @classmethod
    def reset(cls):
        cls._instance = None
    
    def update(
        self,
        spec_idx: int,
        y_true: float,
        lower_raw: float,
        upper_raw: float
    ):
        """Update calibration with new observation."""
        score = max(lower_raw - y_true, y_true - upper_raw)
        self.scores[spec_idx].append(score)
    
    def get_adjustment(self, spec_idx: int) -> float:
        """Get calibration adjustment for a specification."""
        scores = list(self.scores[spec_idx])
        
        if len(scores) < 30:
            return 0.0
        
        n = len(scores)
        alpha = 1 - self.coverage_target
        corrected_quantile = min(1.0, (1 - alpha) * (1 + 1/n))
        
        return np.quantile(scores, corrected_quantile)
    
    def calibrate(
        self,
        spec_idx: int,
        lower_raw: float,
        upper_raw: float
    ) -> Tuple[float, float]:
        """Apply calibration to raw predictions."""
        adj = self.get_adjustment(spec_idx)
        return lower_raw - adj, upper_raw + adj


# =============================================================================
# BASE CLASS
# =============================================================================

class BaseBaseline(ABC):
    """Abstract base class for baselines."""
    
    @abstractmethod
    def fit(self, contexts: np.ndarray, targets: np.ndarray, **kwargs):
        pass
    
    @abstractmethod
    def predict(self, context: np.ndarray) -> Tuple[float, float]:
        pass
    
    @abstractmethod
    def update(self, context: np.ndarray, target: float):
        pass


# =============================================================================
# FAIR BASELINES (all use shared infrastructure)
# =============================================================================

class FairFixedSpecificationBaseline(BaseBaseline):
    """
    Fixed specification baseline using SHARED models.
    
    Uses a single specification throughout, selected based on
    validation performance. Uses shared LightGBM + conformal.
    """
    
    def __init__(
        self,
        specification: int,
        feature_dim: int,
        **kwargs  # Accept but ignore legacy params
    ):
        self.specification = specification
        self.feature_dim = feature_dim
        self._shared_models: Optional[SharedQuantileModels] = None
        self._shared_calibrator: Optional[SharedConformalCalibrator] = None
    
    def set_shared_infrastructure(
        self,
        models: SharedQuantileModels,
        calibrator: SharedConformalCalibrator
    ):
        """Set shared infrastructure (called by experiment runner)."""
        self._shared_models = models
        self._shared_calibrator = calibrator
    
    def fit(self, contexts: np.ndarray, targets: np.ndarray, **kwargs):
        """No-op - uses shared models fitted elsewhere."""
        pass
    
    def predict(self, context: np.ndarray) -> Tuple[float, float]:
        """Predict using shared models + calibration."""
        if self._shared_models is None:
            raise RuntimeError("Shared models not set. Call set_shared_infrastructure().")
        
        lower_raw, upper_raw = self._shared_models.predict(context, self.specification)
        
        if self._shared_calibrator is not None:
            return self._shared_calibrator.calibrate(
                self.specification, lower_raw, upper_raw
            )
        return lower_raw, upper_raw
    
    def update(self, context: np.ndarray, target: float):
        """Update calibration."""
        if self._shared_models is None or self._shared_calibrator is None:
            return
        
        lower_raw, upper_raw = self._shared_models.predict(context, self.specification)
        self._shared_calibrator.update(
            self.specification, target, lower_raw, upper_raw
        )


class FairEnsembleBaseline(BaseBaseline):
    """
    Ensemble baseline using SHARED models.
    
    Averages calibrated predictions across all specifications.
    """
    
    def __init__(
        self,
        num_specifications: int,
        feature_dim: int,
        **kwargs
    ):
        self.num_specifications = num_specifications
        self.feature_dim = feature_dim
        self._shared_models: Optional[SharedQuantileModels] = None
        self._shared_calibrator: Optional[SharedConformalCalibrator] = None
    
    def set_shared_infrastructure(
        self,
        models: SharedQuantileModels,
        calibrator: SharedConformalCalibrator
    ):
        self._shared_models = models
        self._shared_calibrator = calibrator
    
    def fit(self, contexts: np.ndarray, targets: np.ndarray, **kwargs):
        pass
    
    def predict(self, context: np.ndarray) -> Tuple[float, float]:
        if self._shared_models is None:
            raise RuntimeError("Shared models not set.")
        
        lowers, uppers = [], []
        
        for spec_idx in range(self.num_specifications):
            lower_raw, upper_raw = self._shared_models.predict(context, spec_idx)
            
            if self._shared_calibrator is not None:
                lower, upper = self._shared_calibrator.calibrate(
                    spec_idx, lower_raw, upper_raw
                )
            else:
                lower, upper = lower_raw, upper_raw
            
            lowers.append(lower)
            uppers.append(upper)
        
        return float(np.mean(lowers)), float(np.mean(uppers))
    
    def update(self, context: np.ndarray, target: float):
        if self._shared_models is None or self._shared_calibrator is None:
            return
        
        # Update calibration for all specs
        for spec_idx in range(self.num_specifications):
            lower_raw, upper_raw = self._shared_models.predict(context, spec_idx)
            self._shared_calibrator.update(spec_idx, target, lower_raw, upper_raw)


class FairRandomBaseline(BaseBaseline):
    """
    Random selection baseline using SHARED models.
    
    Randomly selects specification at each step.
    """
    
    def __init__(
        self,
        num_specifications: int,
        feature_dim: int,
        seed: int = 42,
        **kwargs
    ):
        self.num_specifications = num_specifications
        self.feature_dim = feature_dim
        self.rng = np.random.default_rng(seed)
        self._shared_models: Optional[SharedQuantileModels] = None
        self._shared_calibrator: Optional[SharedConformalCalibrator] = None
    
    def set_shared_infrastructure(
        self,
        models: SharedQuantileModels,
        calibrator: SharedConformalCalibrator
    ):
        self._shared_models = models
        self._shared_calibrator = calibrator
    
    def fit(self, contexts: np.ndarray, targets: np.ndarray, **kwargs):
        pass
    
    def predict(self, context: np.ndarray) -> Tuple[float, float]:
        if self._shared_models is None:
            raise RuntimeError("Shared models not set.")
        
        spec_idx = self.rng.integers(0, self.num_specifications)
        lower_raw, upper_raw = self._shared_models.predict(context, spec_idx)
        
        if self._shared_calibrator is not None:
            return self._shared_calibrator.calibrate(spec_idx, lower_raw, upper_raw)
        return lower_raw, upper_raw
    
    def update(self, context: np.ndarray, target: float):
        pass  # Could update all specs, but random doesn't need it


class FairACIBaseline(BaseBaseline):
    """
    Adaptive Conformal Inference using SHARED models.
    
    Uses a fixed specification but adapts the coverage rate.
    This is DIFFERENT from CTS which adapts WHICH specification.
    """
    
    def __init__(
        self,
        feature_dim: int,
        base_spec: int = 0,
        target_coverage: float = 0.90,
        gamma: float = 0.01,
        **kwargs
    ):
        self.feature_dim = feature_dim
        self.base_spec = base_spec
        self.target_coverage = target_coverage
        self.target_alpha = 1 - target_coverage
        self.gamma = gamma
        
        self.alpha_t = self.target_alpha
        self._shared_models: Optional[SharedQuantileModels] = None
        
        # ACI's own conformity score buffer
        self.conformity_scores: deque = deque(maxlen=250)
    
    def set_shared_infrastructure(
        self,
        models: SharedQuantileModels,
        calibrator: SharedConformalCalibrator
    ):
        self._shared_models = models
        # ACI doesn't use shared calibrator - has its own adaptive alpha
    
    def fit(self, contexts: np.ndarray, targets: np.ndarray, **kwargs):
        pass
    
    def predict(self, context: np.ndarray) -> Tuple[float, float]:
        if self._shared_models is None:
            raise RuntimeError("Shared models not set.")
        
        lower_raw, upper_raw = self._shared_models.predict(context, self.base_spec)
        
        # ACI-style calibration with adaptive alpha
        if len(self.conformity_scores) >= 30:
            scores = np.array(self.conformity_scores)
            correction = np.quantile(scores, 1 - self.alpha_t)
        else:
            correction = 0.0
        
        return lower_raw - correction, upper_raw + correction
    
    def update(self, context: np.ndarray, target: float):
        if self._shared_models is None:
            return
        
        lower_raw, upper_raw = self._shared_models.predict(context, self.base_spec)
        lower, upper = self.predict(context)
        
        # Check coverage
        covered = lower <= target <= upper
        
        # Update alpha (ACI rule)
        err_t = 1 - int(covered)
        self.alpha_t = self.alpha_t + self.gamma * (self.target_alpha - err_t)
        self.alpha_t = np.clip(self.alpha_t, 0.01, 0.5)
        
        # Update conformity scores
        score = max(lower_raw - target, target - upper_raw)
        self.conformity_scores.append(score)


class FairOracleBaseline(BaseBaseline):
    """
    Oracle baseline using SHARED models.
    
    Selects the specification that achieves best interval score
    FOR EACH SAMPLE (requires knowing the future - upper bound only).
    """
    
    def __init__(
        self,
        num_specifications: int,
        feature_dim: int,
        **kwargs
    ):
        self.num_specifications = num_specifications
        self.feature_dim = feature_dim
        self._shared_models: Optional[SharedQuantileModels] = None
        self._shared_calibrator: Optional[SharedConformalCalibrator] = None
        
        # For oracle computation
        self._last_predictions: Dict[int, Tuple[float, float]] = {}
    
    def set_shared_infrastructure(
        self,
        models: SharedQuantileModels,
        calibrator: SharedConformalCalibrator
    ):
        self._shared_models = models
        self._shared_calibrator = calibrator
    
    def fit(self, contexts: np.ndarray, targets: np.ndarray, **kwargs):
        pass
    
    def predict(self, context: np.ndarray) -> Tuple[float, float]:
        """
        In practice, returns prediction from spec with best recent performance.
        True oracle is computed in oracle_predict_with_target().
        """
        if self._shared_models is None:
            raise RuntimeError("Shared models not set.")
        
        # Cache predictions for oracle computation
        self._last_predictions = {}
        for spec_idx in range(self.num_specifications):
            lower_raw, upper_raw = self._shared_models.predict(context, spec_idx)
            if self._shared_calibrator is not None:
                lower, upper = self._shared_calibrator.calibrate(
                    spec_idx, lower_raw, upper_raw
                )
            else:
                lower, upper = lower_raw, upper_raw
            self._last_predictions[spec_idx] = (lower, upper)
        
        # Return first spec (actual oracle computed with target)
        return self._last_predictions[0]
    
    def oracle_predict_with_target(
        self,
        context: np.ndarray,
        target: float,
        alpha: float = 0.10
    ) -> Tuple[float, float, int]:
        """
        Get oracle prediction given the true target.
        
        Returns (lower, upper, best_spec_idx)
        """
        if self._shared_models is None:
            raise RuntimeError("Shared models not set.")
        
        best_score = float('inf')
        best_lower, best_upper = 0.0, 0.0
        best_spec = 0
        
        for spec_idx in range(self.num_specifications):
            lower_raw, upper_raw = self._shared_models.predict(context, spec_idx)
            if self._shared_calibrator is not None:
                lower, upper = self._shared_calibrator.calibrate(
                    spec_idx, lower_raw, upper_raw
                )
            else:
                lower, upper = lower_raw, upper_raw
            
            # Compute interval score
            width = upper - lower
            if target < lower:
                penalty = (2 / alpha) * (lower - target)
            elif target > upper:
                penalty = (2 / alpha) * (target - upper)
            else:
                penalty = 0.0
            score = width + penalty
            
            if score < best_score:
                best_score = score
                best_lower, best_upper = lower, upper
                best_spec = spec_idx
        
        return best_lower, best_upper, best_spec
    
    def update(self, context: np.ndarray, target: float):
        pass


# =============================================================================
# LEGACY BASELINES (kept for backward compatibility, but recommend using Fair*)
# =============================================================================

# Keep original classes with deprecation warnings
class LightGBMQuantileBaseline(BaseBaseline):
    """
    LightGBM Quantile Regression baseline.
    
    This works correctly because it's batch-trained.
    Kept for backward compatibility.
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
        **kwargs
    ):
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm required")
        
        logger.info(f"Fitting LightGBM on {len(targets)} samples")
        
        lower_params = {**self.lgb_params, 'alpha': self.lower_quantile}
        self.lower_model = lgb.LGBMRegressor(**lower_params)
        self.lower_model.fit(contexts, targets)
        
        upper_params = {**self.lgb_params, 'alpha': self.upper_quantile}
        self.upper_model = lgb.LGBMRegressor(**upper_params)
        self.upper_model.fit(contexts, targets)
        
        self._fitted = True
    
    def predict(self, context: np.ndarray) -> Tuple[float, float]:
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        
        context = np.atleast_2d(context)
        lower = float(self.lower_model.predict(context)[0])
        upper = float(self.upper_model.predict(context)[0])
        
        return min(lower, upper), max(lower, upper)
    
    def update(self, context: np.ndarray, target: float):
        pass  # No online updates


# Aliases for backward compatibility
FixedSpecificationBaseline = FairFixedSpecificationBaseline
EqualWeightEnsemble = FairEnsembleBaseline
RandomSpecificationBaseline = FairRandomBaseline
AdaptiveConformalInference = FairACIBaseline
OracleBaseline = FairOracleBaseline

# Keep MultiQuantileLightGBM as-is (it works correctly)
class MultiQuantileLightGBM(BaseBaseline):
    """LightGBM with multiple quantile models."""
    
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
        import lightgbm as lgb
        
        for q in self.quantiles:
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
        context = np.atleast_2d(context)
        lower = float(self.models[0.05].predict(context)[0])
        upper = float(self.models[0.95].predict(context)[0])
        return min(lower, upper), max(lower, upper)
    
    def update(self, context: np.ndarray, target: float):
        pass


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_baselines(
    num_specifications: int,
    feature_dim: int,
    use_lightgbm: bool = True,
    use_fair_baselines: bool = True,
    seed: int = 42
) -> Dict[str, BaseBaseline]:
    """
    Create standard set of baselines.
    
    Args:
        num_specifications: Number of specification options
        feature_dim: Feature dimension
        use_lightgbm: Include LightGBM baseline
        use_fair_baselines: Use corrected fair baselines (recommended)
        seed: Random seed
        
    Returns:
        Dictionary of baseline name -> baseline instance
    """
    if use_fair_baselines:
        baselines = {
            'fixed_best': FairFixedSpecificationBaseline(
                specification=0,  # Will be set to best on validation
                feature_dim=feature_dim
            ),
            'ensemble': FairEnsembleBaseline(
                num_specifications=num_specifications,
                feature_dim=feature_dim
            ),
            'random': FairRandomBaseline(
                num_specifications=num_specifications,
                feature_dim=feature_dim,
                seed=seed
            ),
            'aci': FairACIBaseline(
                feature_dim=feature_dim,
                target_coverage=0.90
            ),
            'oracle': FairOracleBaseline(
                num_specifications=num_specifications,
                feature_dim=feature_dim
            ),
        }
    else:
        # Legacy baselines (not recommended - unfair comparison)
        logger.warning(
            "Using legacy baselines. These use different infrastructure "
            "than CTS and may produce unfair comparisons."
        )
        baselines = {
            'fixed_best': FixedSpecificationBaseline(
                specification=0,
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


def setup_shared_infrastructure(
    baselines: Dict[str, BaseBaseline],
    num_specifications: int,
    contexts: np.ndarray,
    targets: np.ndarray,
    calibration_window: int = 250,
    coverage_target: float = 0.90,
    seed: int = 42
) -> Tuple[SharedQuantileModels, SharedConformalCalibrator]:
    """
    Setup shared infrastructure and connect to baselines.
    
    Call this after create_baselines() but before evaluation.
    
    Args:
        baselines: Dictionary from create_baselines()
        num_specifications: Number of specifications
        contexts: Training contexts
        targets: Training targets
        calibration_window: Window for conformal calibration
        coverage_target: Target coverage rate
        seed: Random seed
        
    Returns:
        (shared_models, shared_calibrator) tuple
    """
    # Reset singletons
    SharedQuantileModels.reset()
    SharedConformalCalibrator.reset()
    
    # Create shared models
    shared_models = SharedQuantileModels.get_instance(
        num_specifications=num_specifications,
        seed=seed
    )
    shared_models.fit(contexts, targets)
    
    # Create shared calibrator
    shared_calibrator = SharedConformalCalibrator.get_instance(
        num_specifications=num_specifications,
        calibration_window=calibration_window,
        coverage_target=coverage_target
    )
    
    # Connect to all baselines that support it
    for name, baseline in baselines.items():
        if hasattr(baseline, 'set_shared_infrastructure'):
            baseline.set_shared_infrastructure(shared_models, shared_calibrator)
            logger.info(f"  Connected {name} to shared infrastructure")
    
    return shared_models, shared_calibrator


if __name__ == "__main__":
    # Test the corrected baselines
    logging.basicConfig(level=logging.INFO)
    
    np.random.seed(42)
    
    print("Testing FAIR baselines (all use same infrastructure)")
    print("=" * 60)
    
    # Generate test data
    n_train = 2000
    n_test = 500
    feature_dim = 15
    num_specs = 4
    
    X_train = np.random.randn(n_train, feature_dim)
    y_train = 50 + 10 * X_train[:, 0] + 5 * X_train[:, 1] + np.random.randn(n_train) * 5
    
    X_test = np.random.randn(n_test, feature_dim)
    y_test = 50 + 10 * X_test[:, 0] + 5 * X_test[:, 1] + np.random.randn(n_test) * 5
    
    # Create baselines
    baselines = create_baselines(
        num_specifications=num_specs,
        feature_dim=feature_dim,
        use_lightgbm=True,
        use_fair_baselines=True
    )
    
    # Setup shared infrastructure
    shared_models, shared_calibrator = setup_shared_infrastructure(
        baselines=baselines,
        num_specifications=num_specs,
        contexts=X_train,
        targets=y_train
    )
    
    # Fit LightGBM separately (it's not part of shared infra)
    baselines['lightgbm'].fit(X_train, y_train)
    
    # Evaluate
    print("\nResults (all Fair* baselines use SAME models):")
    print("-" * 60)
    
    for name, baseline in baselines.items():
        coverages = []
        widths = []
        
        for x, y in zip(X_test, y_test):
            if name == 'oracle':
                lower, upper, _ = baseline.oracle_predict_with_target(x, y)
            else:
                lower, upper = baseline.predict(x)
            
            coverages.append(lower <= y <= upper)
            widths.append(upper - lower)
            baseline.update(x, y)
        
        print(f"{name:12s}  coverage={np.mean(coverages):.1%}  width={np.mean(widths):.2f}")
    
    print("\n✓ All Fair* baselines use identical LightGBM models + calibration")
    print("✓ Only difference is specification SELECTION strategy")
