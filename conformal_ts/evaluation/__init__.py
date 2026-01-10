"""Evaluation metrics for probabilistic forecasting."""

from .competition_metrics import (
    # Core metrics
    pinball_loss,
    interval_score,
    coverage_rate,
    crps,
    
    # Competition-specific
    weighted_scaled_pinball_loss,
    m5_wspl_from_intervals,
    gefcom_pinball_score,
    
    # Statistical tests
    diebold_mariano_test,
    bootstrap_confidence_interval,
    compare_methods,
    
    # Comprehensive evaluation
    full_evaluation,
    EvaluationResults,
    
    # Constants
    M5_QUANTILES,
    GEFCOM_QUANTILES,
)

# Keep original metrics available
try:
    from .metrics import (
        interval_score as original_interval_score,
        coverage_rate as original_coverage_rate,
        mean_interval_width,
        winkler_score,
        quantile_score,
    )
except ImportError:
    pass

__all__ = [
    'pinball_loss',
    'interval_score',
    'coverage_rate',
    'crps',
    'weighted_scaled_pinball_loss',
    'm5_wspl_from_intervals',
    'gefcom_pinball_score',
    'diebold_mariano_test',
    'bootstrap_confidence_interval',
    'compare_methods',
    'full_evaluation',
    'EvaluationResults',
    'M5_QUANTILES',
    'GEFCOM_QUANTILES',
]