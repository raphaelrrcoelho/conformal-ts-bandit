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
forecasts in a single O(T) (or O(T*w)) pass, avoiding O(T²)
recomputation in the scoring loop.

Multi-step-ahead forecasting (horizon > 1):
    ``forecasts[t]`` = prediction for ``y[t]`` using only ``y[:t-h]``
    (information available *h* steps before the target).  This keeps
    the scoring loop in ``build_scores_matrix_with_cqr`` unchanged --
    it still compares ``forecasts[t]`` against ``series[t]``.
"""

import math
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
    """Forecast = last observed value h steps back: y_hat[t] = y[t-h]."""

    def __init__(self, horizon: int = 1):
        h = max(1, horizon)
        super().__init__(name="Naive", min_history=h)
        self.horizon = h

    def forecast_all(self, series: np.ndarray) -> np.ndarray:
        T = len(series)
        h = self.horizon
        forecasts = np.full(T, np.nan)
        forecasts[h:] = series[:-h]
        return forecasts


class SeasonalNaiveForecaster(ForecasterSpec):
    """Forecast = value from the most recent complete seasonal cycle.

    At horizon h, the look-back is ``period * ceil(h / period)`` steps,
    ensuring we always reference a full-period-ago observation that is
    available at the forecast origin (t - h).
    """

    def __init__(self, period: int = 24, horizon: int = 1):
        h = max(1, horizon)
        lag = period * math.ceil(h / period)
        super().__init__(name=f"SeasonalNaive({period})", min_history=lag)
        self.period = period
        self.horizon = h
        self.lag = lag

    def forecast_all(self, series: np.ndarray) -> np.ndarray:
        T = len(series)
        lag = self.lag
        forecasts = np.full(T, np.nan)
        forecasts[lag:] = series[:-lag]
        return forecasts


class RollingMeanForecaster(ForecasterSpec):
    """Forecast = mean of *window* observations ending h steps back."""

    def __init__(self, window: int = 50, horizon: int = 1):
        h = max(1, horizon)
        super().__init__(name=f"RollingMean({window})", min_history=window + h)
        self.window = window
        self.horizon = h

    def forecast_all(self, series: np.ndarray) -> np.ndarray:
        T = len(series)
        w = self.window
        h = self.horizon
        forecasts = np.full(T, np.nan)
        cs = np.concatenate(([0.0], np.cumsum(series)))
        # forecast[t] = mean(series[t-h-w : t-h])
        #             = (cs[t-h] - cs[t-h-w]) / w
        for t in range(w + h, T):
            forecasts[t] = (cs[t - h] - cs[t - h - w]) / w
        return forecasts


class SESForecaster(ForecasterSpec):
    """Simple Exponential Smoothing: level = α*y + (1-α)*level.

    At horizon h, the forecast for y[t] uses the SES level computed
    through y[t-h] (the last observation available at origin t-h).
    """

    def __init__(self, alpha: float = 0.3, horizon: int = 1):
        h = max(1, horizon)
        super().__init__(name=f"SES({alpha})", min_history=h + 1)
        self.alpha = alpha
        self.horizon = h

    def forecast_all(self, series: np.ndarray) -> np.ndarray:
        T = len(series)
        a = self.alpha
        h = self.horizon
        forecasts = np.full(T, np.nan)

        # Pre-compute SES levels: levels[t] = level after observing y[t]
        levels = np.empty(T)
        levels[0] = series[0]
        for t in range(1, T):
            levels[t] = a * series[t] + (1 - a) * levels[t - 1]

        # forecast[t] = level at t-h (computed through y[t-h])
        # Available for t >= h+1 (need at least y[0] to compute levels[0],
        # then use levels[t-h] where t-h >= 0, but we also need t-h >= 0
        # so t >= h; we require min_history = h+1 for the first real forecast)
        for t in range(h, T):
            # levels[t-h] is the level after observing y[t-h].
            # For h=1: forecasts[t] = levels[t-1], same as before.
            forecasts[t] = levels[t - h]

        return forecasts


class LinearTrendForecaster(ForecasterSpec):
    """Sliding-window OLS: fit y = a + b*x on *window* points ending h steps back, extrapolate h steps."""

    def __init__(self, window: int = 20, horizon: int = 1):
        h = max(1, horizon)
        super().__init__(name=f"LinearTrend({window})", min_history=window + h)
        self.window = window
        self.horizon = h

    def forecast_all(self, series: np.ndarray) -> np.ndarray:
        T = len(series)
        w = self.window
        h = self.horizon
        forecasts = np.full(T, np.nan)

        # Pre-compute x-coordinates (0, 1, ..., w-1) stats (constant)
        x = np.arange(w, dtype=np.float64)
        x_mean = x.mean()
        ss_xx = np.sum((x - x_mean) ** 2)

        for t in range(w + h, T):
            # Window: series[t-h-w : t-h]
            y_win = series[t - h - w : t - h]
            y_mean = y_win.mean()
            ss_xy = np.sum((x - x_mean) * (y_win - y_mean))
            b = ss_xy / ss_xx if ss_xx > 0 else 0.0
            a = y_mean - b * x_mean
            # Extrapolate h steps beyond the window end: x = w - 1 + h
            forecasts[t] = a + b * (w - 1 + h)
        return forecasts


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_default_forecasters(
    seasonal_period: int = 24,
    rolling_window: int = 50,
    trend_window: int = 20,
    horizon: int = 1,
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
    horizon : int
        Forecast horizon.  At horizon *h*, each forecaster uses only
        information through y[t-h] to predict y[t].

    Returns
    -------
    list of ForecasterSpec
        [Naive, SeasonalNaive, RollingMean, SES, LinearTrend]
    """
    return [
        NaiveForecaster(horizon=horizon),
        SeasonalNaiveForecaster(period=seasonal_period, horizon=horizon),
        RollingMeanForecaster(window=rolling_window, horizon=horizon),
        SESForecaster(alpha=0.3, horizon=horizon),
        LinearTrendForecaster(window=trend_window, horizon=horizon),
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

    for horizon in [1, 24]:
        print(f"--- horizon = {horizon} ---")
        forecasters = make_default_forecasters(
            seasonal_period=24, rolling_window=50, trend_window=20,
            horizon=horizon,
        )

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
            # Verify no data leakage: forecast[t] should only use y[:t-h]
            # For NaiveForecaster: forecast[h] = y[0], so preds[h] == series[0]
            if isinstance(fc, NaiveForecaster):
                assert preds[horizon] == series[0], (
                    f"Naive(h={horizon}): preds[{horizon}] should be series[0]"
                )

        print()

    # Extra check: at h=1, SES should match the old behaviour
    ses_h1 = SESForecaster(alpha=0.3, horizon=1)
    preds_h1 = ses_h1.forecast_all(series)
    assert np.isnan(preds_h1[0])
    assert preds_h1[1] == series[0], "SES h=1: forecast[1] should be series[0]"

    print("All assertions passed.")
    print("Self-test complete.")
