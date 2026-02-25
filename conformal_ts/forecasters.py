"""
Diverse Forecaster Specifications for Conformal Thompson Sampling.

Provides 5 structurally different forecasters that give CTS genuinely
different failure modes to exploit:

| # | Name           | Wins when...         |
|---|----------------|----------------------|
| 0 | Naive          | Random walk          |
| 1 | Seasonal Naive | Periodic data        |
| 2 | Rolling Mean   | Mean-reverting       |
| 3 | SES (α=0.3)   | Level shifts         |
| 4 | Linear Trend   | Trending data        |

Each forecaster implements ``forecast_all(series)`` which pre-computes
ALL one-step-ahead forecasts in a single O(T) (or O(T*w)) pass,
avoiding O(T²) recomputation in the scoring loop.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

__all__ = [
    "ForecasterSpec",
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "RollingMeanForecaster",
    "SESForecaster",
    "LinearTrendForecaster",
    "make_default_forecasters",
]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

@dataclass
class ForecasterSpec:
    """Base class for a forecaster specification.

    Subclasses must implement :meth:`forecast_all`.
    """

    name: str
    min_history: int

    def forecast_all(self, series: np.ndarray) -> np.ndarray:
        """Pre-compute forecasts[t] = forecast for y[t] using y[:t].

        Returns an array of length T.  Entries where the forecaster
        cannot yet produce a forecast (insufficient history) are NaN.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

class NaiveForecaster(ForecasterSpec):
    """Forecast = last observed value: y_hat[t] = y[t-1]."""

    def __init__(self):
        super().__init__(name="Naive", min_history=1)

    def forecast_all(self, series: np.ndarray) -> np.ndarray:
        T = len(series)
        forecasts = np.full(T, np.nan)
        forecasts[1:] = series[:-1]
        return forecasts


class SeasonalNaiveForecaster(ForecasterSpec):
    """Forecast = value from one seasonal period ago: y_hat[t] = y[t-period]."""

    def __init__(self, period: int = 24):
        super().__init__(name=f"SeasonalNaive({period})", min_history=period)
        self.period = period

    def forecast_all(self, series: np.ndarray) -> np.ndarray:
        T = len(series)
        forecasts = np.full(T, np.nan)
        p = self.period
        forecasts[p:] = series[:-p]
        return forecasts


class RollingMeanForecaster(ForecasterSpec):
    """Forecast = mean of the last *window* observations (cumsum trick, O(T))."""

    def __init__(self, window: int = 50):
        super().__init__(name=f"RollingMean({window})", min_history=window)
        self.window = window

    def forecast_all(self, series: np.ndarray) -> np.ndarray:
        T = len(series)
        w = self.window
        forecasts = np.full(T, np.nan)
        cs = np.concatenate(([0.0], np.cumsum(series)))
        # forecast[t] = mean(series[t-w : t]) = (cs[t] - cs[t-w]) / w
        for t in range(w, T):
            forecasts[t] = (cs[t] - cs[t - w]) / w
        return forecasts


class SESForecaster(ForecasterSpec):
    """Simple Exponential Smoothing: level = α*y + (1-α)*level."""

    def __init__(self, alpha: float = 0.3):
        super().__init__(name=f"SES({alpha})", min_history=2)
        self.alpha = alpha

    def forecast_all(self, series: np.ndarray) -> np.ndarray:
        T = len(series)
        a = self.alpha
        forecasts = np.full(T, np.nan)
        level = series[0]
        for t in range(1, T):
            forecasts[t] = level  # forecast made BEFORE observing y[t]
            level = a * series[t] + (1 - a) * level
        return forecasts


class LinearTrendForecaster(ForecasterSpec):
    """Sliding-window OLS: fit y = a + b*x on the last *window* points, extrapolate one step."""

    def __init__(self, window: int = 20):
        super().__init__(name=f"LinearTrend({window})", min_history=window)
        self.window = window

    def forecast_all(self, series: np.ndarray) -> np.ndarray:
        T = len(series)
        w = self.window
        forecasts = np.full(T, np.nan)

        # Pre-compute x-coordinates (0, 1, ..., w-1) stats (constant)
        x = np.arange(w, dtype=np.float64)
        x_mean = x.mean()
        ss_xx = np.sum((x - x_mean) ** 2)

        for t in range(w, T):
            y_win = series[t - w : t]
            y_mean = y_win.mean()
            ss_xy = np.sum((x - x_mean) * (y_win - y_mean))
            b = ss_xy / ss_xx if ss_xx > 0 else 0.0
            a = y_mean - b * x_mean
            # Extrapolate to x = w (one step beyond the window)
            forecasts[t] = a + b * w
        return forecasts


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_default_forecasters(
    seasonal_period: int = 24,
    rolling_window: int = 50,
    trend_window: int = 20,
) -> List[ForecasterSpec]:
    """Create the default set of 5 diverse forecasters.

    Parameters
    ----------
    seasonal_period : int
        Period for the seasonal naive forecaster (e.g. 24 for hourly
        data with a daily cycle, 96 for 15-min data).
    rolling_window : int
        Window size for the rolling mean forecaster.
    trend_window : int
        Window size for the linear trend forecaster.

    Returns
    -------
    list of ForecasterSpec
        [Naive, SeasonalNaive, RollingMean, SES, LinearTrend]
    """
    return [
        NaiveForecaster(),
        SeasonalNaiveForecaster(period=seasonal_period),
        RollingMeanForecaster(window=rolling_window),
        SESForecaster(alpha=0.3),
        LinearTrendForecaster(window=trend_window),
    ]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== forecasters self-test ===\n")

    rng = np.random.default_rng(42)

    # Synthetic series: trend + seasonality + noise
    T = 500
    t = np.arange(T, dtype=np.float64)
    series = 100 + 0.05 * t + 10 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 2, T)

    forecasters = make_default_forecasters(seasonal_period=24, rolling_window=50, trend_window=20)

    for fc in forecasters:
        preds = fc.forecast_all(series)
        valid = ~np.isnan(preds)
        n_valid = valid.sum()
        if n_valid > 0:
            residuals = series[valid] - preds[valid]
            rmse = np.sqrt(np.mean(residuals ** 2))
            mae = np.mean(np.abs(residuals))
        else:
            rmse = mae = float("nan")
        print(
            f"  {fc.name:<25s}  min_hist={fc.min_history:>3d}  "
            f"valid={n_valid:>4d}/{T}  RMSE={rmse:.3f}  MAE={mae:.3f}"
        )

    # Verify shapes and NaN placement
    for fc in forecasters:
        preds = fc.forecast_all(series)
        assert preds.shape == (T,), f"{fc.name}: wrong shape {preds.shape}"
        assert np.isnan(preds[0]), f"{fc.name}: preds[0] should be NaN"
        assert not np.isnan(preds[fc.min_history]), (
            f"{fc.name}: preds[{fc.min_history}] should not be NaN"
        )

    print("\nAll assertions passed.")
    print("Self-test complete.")
