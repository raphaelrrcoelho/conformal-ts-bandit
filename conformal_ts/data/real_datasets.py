"""
Real Public Dataset Loaders for Bandit Experiments.

Downloads, caches, and prepares publicly available time series datasets
for use with Conformal Thompson Sampling experiments.  Each dataset is
converted into a (contexts, scores_matrix) representation where:

- contexts:       (T, D) feature matrix  -- bandit context at each step
- scores_matrix:  (T, K) interval scores -- one per specification (lookback window)

The specification space consists of different lookback windows for a simple
rolling-statistics forecaster.  At each timestep *t* and for each spec *k*
(lookback window w_k):

    1. Compute rolling_mean and rolling_std over the last w_k observations.
    2. Form a prediction interval [mean - z * std, mean + z * std].
    3. Evaluate interval_score(lower, upper, y_true, alpha).

This produces genuinely different performance across windows depending on
the local regime (trend, volatility, seasonality).

Datasets
--------
1. **ETTh1** -- Electricity Transformer Temperature (hourly), 7 features
   URL: https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv

2. **ETTh2** -- Second transformer station (hourly), 7 features
   URL: https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv

3. **Electricity** -- UCI 370-client electricity consumption (15-min)
   URL: https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip

4. **AustralianElecDemand** -- Synthetic Australian-style electricity demand
   (deterministic seasonal + stochastic; no external download needed)

5. **Weather** -- 21 meteorological indicators (10-min, resampled to hourly)
   URL: https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv
   (We reuse ETTm1 at minute granularity as a weather-proxy dataset.)

References
----------
Zhou et al. (2021) "Informer: Beyond Efficient Transformer for Long
Sequence Time-Series Forecasting", AAAI.
"""

import gzip
import io
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from conformal_ts.evaluation.metrics import interval_score

logger = logging.getLogger(__name__)

__all__ = [
    "RealDatasetConfig",
    "RealDatasetLoader",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RealDatasetConfig:
    """Metadata and download information for a single real dataset."""

    name: str
    url: str
    target_col: str
    freq: str  # pandas frequency alias: 'H', '15min', 'D', etc.
    description: str = ""
    date_col: str = "date"
    separator: str = ","
    decimal: str = "."
    header: int = 0
    # Lookback windows (in native time-steps) used as the specification space.
    # Defaults are overridden per-dataset at registration time.
    lookback_windows: List[int] = field(default_factory=lambda: [24, 48, 96, 168, 336])
    # Number of clients to sample for very large multi-client datasets.
    sample_clients: Optional[int] = None
    # Whether the URL points to a zip file.
    is_zip: bool = False
    # Filename inside the zip (if applicable).
    zip_member: Optional[str] = None
    # Whether the URL points to a gzip file.
    is_gzip: bool = False
    # If True, file has no header row (headerless CSV/TSV).
    no_header: bool = False
    # Seasonal period (in native timesteps) for the seasonal naive forecaster.
    seasonal_period: Optional[int] = None


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, RealDatasetConfig] = {
    "ETTh1": RealDatasetConfig(
        name="ETTh1",
        url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        target_col="OT",
        freq="H",
        description="Electricity Transformer Temperature -- station 1, hourly.",
        lookback_windows=[24, 48, 96, 168, 336],
        seasonal_period=24,
    ),
    "ETTh2": RealDatasetConfig(
        name="ETTh2",
        url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
        target_col="OT",
        freq="H",
        description="Electricity Transformer Temperature -- station 2, hourly.",
        lookback_windows=[24, 48, 96, 168, 336],
        seasonal_period=24,
    ),
    "Electricity": RealDatasetConfig(
        name="Electricity",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip",
        target_col="MT_001",  # Will be replaced by sampled client columns
        freq="15min",
        description="UCI 370-client electricity consumption, 15-min resolution.",
        separator=";",
        decimal=",",
        header=0,
        lookback_windows=[96, 192, 384, 672, 1344],  # Scaled for 15-min
        sample_clients=5,
        is_zip=True,
        zip_member="LD2011_2014.txt",
        seasonal_period=96,
    ),
    "AustralianElecDemand": RealDatasetConfig(
        name="AustralianElecDemand",
        url="",  # Synthesised locally -- no download needed
        target_col="demand",
        freq="30min",
        description="Synthetic Australian-style half-hourly electricity demand.",
        lookback_windows=[48, 96, 192, 336, 672],  # Scaled for 30-min
        seasonal_period=48,
    ),
    "ETTm1": RealDatasetConfig(
        name="ETTm1",
        url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
        target_col="OT",
        freq="15min",
        description="Electricity Transformer Temperature -- station 1, 15-min.",
        lookback_windows=[96, 192, 384, 672, 1344],
        seasonal_period=96,
    ),
    "ETTm2": RealDatasetConfig(
        name="ETTm2",
        url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
        target_col="OT",
        freq="15min",
        description="Electricity Transformer Temperature -- station 2, 15-min.",
        lookback_windows=[96, 192, 384, 672, 1344],
        seasonal_period=96,
    ),
    "ExchangeRate": RealDatasetConfig(
        name="ExchangeRate",
        url="https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz",
        target_col="col_0",
        freq="D",
        description="Daily exchange rates of 8 countries, 1990-2016.",
        lookback_windows=[7, 14, 30, 60, 120],
        is_gzip=True,
        no_header=True,
        seasonal_period=7,
    ),
    "Traffic": RealDatasetConfig(
        name="Traffic",
        url="https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/traffic/traffic.txt.gz",
        target_col="col_0",
        freq="H",
        description="California freeway occupancy rates, 862 sensors, hourly.",
        lookback_windows=[24, 48, 96, 168, 336],
        is_gzip=True,
        no_header=True,
        seasonal_period=24,
    ),
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class RealDatasetLoader:
    """Downloads, caches, and prepares real datasets for bandit experiments."""

    REGISTRY: Dict[str, RealDatasetConfig] = _REGISTRY

    def __init__(self, cache_dir: str = "./cache/datasets"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def list_available() -> List[str]:
        """Return the names of all registered datasets."""
        return list(_REGISTRY.keys())

    def load(self, name: str) -> Optional[pd.DataFrame]:
        """
        Load a dataset by name, downloading and caching if needed.

        Parameters
        ----------
        name : str
            One of the keys in ``REGISTRY``.

        Returns
        -------
        pd.DataFrame or None
            DataFrame indexed by datetime with numeric columns.
            Returns ``None`` if the download fails.
        """
        if name not in self.REGISTRY:
            raise ValueError(
                f"Unknown dataset '{name}'. "
                f"Available: {self.list_available()}"
            )

        config = self.REGISTRY[name]

        # Special case: Australian demand is generated, not downloaded.
        if name == "AustralianElecDemand":
            return self._generate_australian_demand()

        cache_path = self.cache_dir / f"{name}.csv"
        if cache_path.exists():
            logger.info("Loading cached %s from %s", name, cache_path)
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return df

        logger.info("Downloading %s from %s ...", name, config.url)
        try:
            df = self._download(config)
        except Exception as exc:
            logger.error("Failed to download %s: %s", name, exc)
            return None

        if df is not None:
            df.to_csv(cache_path)
            logger.info("Cached %s to %s", name, cache_path)

        return df

    def prepare_bandit_experiment(
        self,
        name: str,
        num_specs: Optional[int] = None,
        alpha: float = 0.10,
        z: float = 1.96,
        warmup_fraction: float = 0.05,
    ) -> Optional[Dict]:
        """
        Prepare a dataset for the bandit specification-selection experiment.

        Parameters
        ----------
        name : str
            Dataset name (key in REGISTRY).
        num_specs : int, optional
            If given, use only the first ``num_specs`` lookback windows.
        alpha : float
            Miscoverage rate for the interval score (default 0.10 = 90 %).
        z : float
            Number of standard deviations for the prediction interval
            (default 1.96 ~ 97.5 % two-sided).
        warmup_fraction : float
            Fraction of timesteps to skip at the start (before the longest
            lookback window has enough data).

        Returns
        -------
        dict or None
            ``contexts``      -- np.ndarray, shape (T, D)
            ``scores_matrix`` -- np.ndarray, shape (T, K)
            ``lookback_windows`` -- list[int]
            ``timestamps``    -- pd.DatetimeIndex
            ``target_values`` -- np.ndarray, shape (T,)
            ``dataset_name``  -- str
            ``freq``          -- str
            Returns ``None`` when the dataset cannot be loaded.
        """
        df = self.load(name)
        if df is None:
            return None

        config = self.REGISTRY[name]
        target_col = config.target_col

        # For Electricity we average sampled client columns into a target.
        if name == "Electricity" and target_col not in df.columns:
            target_col = df.columns[0]

        if target_col not in df.columns:
            logger.error(
                "Target column '%s' not found in %s. Columns: %s",
                target_col, name, list(df.columns),
            )
            return None

        target = df[target_col].values.astype(np.float64)

        lookback_windows = list(config.lookback_windows)
        if num_specs is not None:
            lookback_windows = lookback_windows[:num_specs]

        max_lb = max(lookback_windows)
        warmup = max(int(len(target) * warmup_fraction), max_lb + 1)

        T_total = len(target) - warmup
        if T_total <= 0:
            logger.error("Dataset %s is too short for the requested lookback windows.", name)
            return None

        # --- Build scores_matrix (T, K) -----------------------------------
        K = len(lookback_windows)
        scores_matrix = np.full((T_total, K), np.nan)

        for k, w in enumerate(lookback_windows):
            for t_idx in range(T_total):
                t = warmup + t_idx
                history = target[t - w : t]
                if len(history) < 2:
                    continue
                mu = np.mean(history)
                sigma = np.std(history, ddof=1)
                sigma = max(sigma, 1e-8)
                lower = mu - z * sigma
                upper = mu + z * sigma
                y_true = target[t]
                scores_matrix[t_idx, k] = interval_score(
                    np.array([lower]),
                    np.array([upper]),
                    np.array([y_true]),
                    alpha=alpha,
                )[0]

        # --- Build context features (T, D) --------------------------------
        contexts = self.build_context_features(
            df, target, warmup, T_total, config,
        )

        timestamps = df.index[warmup : warmup + T_total]
        target_values = target[warmup : warmup + T_total]

        return {
            "contexts": contexts,
            "scores_matrix": scores_matrix,
            "lookback_windows": lookback_windows,
            "timestamps": timestamps,
            "target_values": target_values,
            "dataset_name": name,
            "freq": config.freq,
        }

    def prepare_raw_series(
        self,
        name: str,
        num_specs: Optional[int] = None,
        warmup_fraction: float = 0.05,
    ) -> Optional[Dict]:
        """
        Return raw series data for external CQR scoring.

        Unlike :meth:`prepare_bandit_experiment`, this does **not** build
        scores or context features -- it just loads the dataset, identifies
        the target column, computes the warmup, and returns the pieces
        that the caller needs to compose CQR scoring with rich contexts.

        Parameters
        ----------
        name : str
            Dataset name (key in REGISTRY).
        num_specs : int, optional
            If given, use only the first ``num_specs`` lookback windows.
        warmup_fraction : float
            Fraction of timesteps to skip at the start.

        Returns
        -------
        dict or None
            ``df``               -- pd.DataFrame (full dataset)
            ``target``           -- np.ndarray, shape (N,) raw target values
            ``lookback_windows`` -- list[int]
            ``config``           -- RealDatasetConfig
            ``warmup``           -- int  (number of steps to skip)
            ``T_total``          -- int  (number of scoreable steps)
        """
        df = self.load(name)
        if df is None:
            return None

        config = self.REGISTRY[name]
        target_col = config.target_col

        # For Electricity we average sampled client columns into a target.
        if name == "Electricity" and target_col not in df.columns:
            target_col = df.columns[0]

        if target_col not in df.columns:
            logger.error(
                "Target column '%s' not found in %s. Columns: %s",
                target_col, name, list(df.columns),
            )
            return None

        target = df[target_col].values.astype(np.float64)

        lookback_windows = list(config.lookback_windows)
        if num_specs is not None:
            lookback_windows = lookback_windows[:num_specs]

        max_lb = max(lookback_windows)
        warmup = max(int(len(target) * warmup_fraction), max_lb + 1)

        T_total = len(target) - warmup
        if T_total <= 0:
            logger.error(
                "Dataset %s is too short for the requested lookback windows.",
                name,
            )
            return None

        return {
            "df": df,
            "target": target,
            "lookback_windows": lookback_windows,
            "config": config,
            "warmup": warmup,
            "T_total": T_total,
        }

    # ------------------------------------------------------------------
    # Private helpers -- downloading
    # ------------------------------------------------------------------

    def _download(self, config: RealDatasetConfig) -> Optional[pd.DataFrame]:
        """Download a single dataset and return a cleaned DataFrame."""
        import urllib.request

        url = config.url
        req = urllib.request.Request(url, headers={"User-Agent": "conformal_ts/1.0"})

        with urllib.request.urlopen(req, timeout=120) as resp:
            raw_bytes = resp.read()

        if config.is_zip:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                member = config.zip_member or zf.namelist()[0]
                with zf.open(member) as f:
                    raw_bytes = f.read()

        if config.is_gzip:
            raw_bytes = gzip.decompress(raw_bytes)

        # Parse CSV
        text = raw_bytes.decode("utf-8", errors="replace")
        header_arg = None if config.no_header else config.header
        df = pd.read_csv(
            io.StringIO(text),
            sep=config.separator,
            decimal=config.decimal,
            header=header_arg,
            low_memory=False,
        )

        # Auto-name columns for headerless files
        if config.no_header:
            df.columns = [f"col_{i}" for i in range(len(df.columns))]

        # Post-process per dataset
        if config.name == "Electricity":
            df = self._process_electricity(df, config)
        elif config.no_header:
            df = self._process_headerless(df, config)
        else:
            df = self._process_generic(df, config)

        return df

    @staticmethod
    def _process_generic(
        df: pd.DataFrame, config: RealDatasetConfig
    ) -> pd.DataFrame:
        """Process ETT-style CSVs (date + numeric columns)."""
        date_col = config.date_col
        if date_col not in df.columns:
            # Guess: first column that looks like a date
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    date_col = col
                    break
            else:
                # Fall back to first column
                date_col = df.columns[0]

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()

        # Keep only numeric columns
        df = df.select_dtypes(include=[np.number])

        # Drop rows that are entirely NaN
        df = df.dropna(how="all")

        # Forward-fill small gaps, then drop remaining NaN rows
        df = df.ffill(limit=4).dropna()

        return df

    @staticmethod
    def _process_headerless(
        df: pd.DataFrame, config: RealDatasetConfig
    ) -> pd.DataFrame:
        """Process headerless numeric files (exchange_rate, traffic).

        These have no date column -- we synthesize a DatetimeIndex from
        the configured frequency so downstream code works uniformly.
        """
        df = df.select_dtypes(include=[np.number])
        df = df.dropna(how="all")

        # Synthesize datetime index
        freq_map = {"D": "D", "H": "h", "15min": "15min", "30min": "30min"}
        pd_freq = freq_map.get(config.freq, "h")
        df.index = pd.date_range("2000-01-01", periods=len(df), freq=pd_freq)
        df.index.name = "date"

        return df

    def _process_electricity(
        self, df: pd.DataFrame, config: RealDatasetConfig
    ) -> pd.DataFrame:
        """
        Process the UCI Electricity dataset.

        The raw file uses ``;`` as separator and ``,`` as decimal.
        We sample a few clients, aggregate to reduce noise, and
        set a DatetimeIndex.
        """
        # The first column is the datetime index (unnamed or '')
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()

        # Ensure all columns are numeric (handles any remaining string values)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                )
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop columns that are all-zero or all-NaN (some clients have no data)
        df = df.loc[:, (df != 0).any(axis=0)]
        df = df.dropna(axis=1, how="all")

        # Sample a handful of clients for tractability
        n_sample = config.sample_clients or 5
        rng = np.random.default_rng(42)
        chosen = rng.choice(df.columns, size=min(n_sample, len(df.columns)), replace=False)
        df = df[chosen].copy()

        # Forward-fill short gaps
        df = df.ffill(limit=4).dropna()

        return df

    @staticmethod
    def _generate_australian_demand() -> pd.DataFrame:
        """
        Generate a realistic half-hourly Australian electricity demand series.

        The series has:
        - Strong daily and weekly seasonality
        - Temperature-driven summer/winter peaks
        - Stochastic volatility (regime-like behaviour)
        - Occasional demand spikes (heatwaves / cold snaps)
        """
        rng = np.random.default_rng(2024)

        start = pd.Timestamp("2015-01-01")
        periods = 48 * 365 * 3  # 3 years of half-hourly data
        index = pd.date_range(start, periods=periods, freq="30min")

        hour_frac = index.hour + index.minute / 60.0
        day_of_year = index.dayofyear
        day_of_week = index.dayofweek  # 0=Mon

        # -- Base load (MW) ------------------------------------------------
        base_load = 5000.0

        # Daily pattern: morning + evening peaks
        daily = (
            800 * np.exp(-((hour_frac - 8.5) ** 2) / 4)
            + 1200 * np.exp(-((hour_frac - 18.5) ** 2) / 3)
            - 600 * np.exp(-((hour_frac - 3.0) ** 2) / 6)
        )

        # Weekly: lower on weekends
        weekend_factor = np.where(day_of_week >= 5, -500.0, 0.0)

        # Seasonal: summer + winter heating/cooling peaks (southern hemisphere)
        seasonal = (
            1000 * np.cos(2 * np.pi * (day_of_year - 15) / 365)   # Summer peak Jan
            + 400 * np.cos(2 * np.pi * (day_of_year - 195) / 365)  # Winter peak Jul
        )

        # Temperature proxy (drives volatility)
        temperature = (
            25 + 12 * np.sin(2 * np.pi * (day_of_year - 15) / 365)
            + 5 * np.sin(2 * np.pi * hour_frac / 24)
            + rng.normal(0, 2, size=periods)
        )

        # Temperature-demand interaction (cooling above 30, heating below 10)
        temp_effect = np.where(
            temperature > 30,
            150 * (temperature - 30),
            np.where(temperature < 10, 100 * (10 - temperature), 0.0),
        )

        # Random walk for slow-moving demand level
        level_noise = np.cumsum(rng.normal(0, 5, size=periods))
        level_noise -= np.mean(level_noise)

        # Occasional spikes (heatwaves etc.)
        spike_mask = rng.random(periods) < 0.002
        spikes = spike_mask * rng.uniform(500, 2000, size=periods)

        demand = (
            base_load
            + daily
            + weekend_factor
            + seasonal
            + temp_effect
            + level_noise
            + spikes
            + rng.normal(0, 150, size=periods)
        )
        demand = np.maximum(demand, 1000)  # Floor

        df = pd.DataFrame(
            {"demand": demand, "temperature": temperature},
            index=index,
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers -- context features
    # ------------------------------------------------------------------

    @staticmethod
    def build_context_features(
        df: pd.DataFrame,
        target: np.ndarray,
        warmup: int,
        T: int,
        config: RealDatasetConfig,
    ) -> np.ndarray:
        """
        Build the (T, D) feature matrix used as bandit context.

        Features:
            0. bias (constant 1)
            1. recent mean      (last 24 native steps, or 24)
            2. recent std       (last 24 native steps)
            3. recent trend     (mean last 12 - mean prev 12)
            4. longer mean      (last 168 native steps, capped)
            5. longer std       (last 168 native steps, capped)
            6. hour-of-day sin  (if datetime index)
            7. hour-of-day cos
            8. day-of-week sin
            9. day-of-week cos
           10. lag 1
           11. lag 2
           12. lag 3
        """
        # Choose short and long context spans proportional to frequency
        freq_map = {"H": 1, "15min": 4, "30min": 2, "D": 1 / 24}
        steps_per_hour = freq_map.get(config.freq, 1)
        short_span = max(int(24 * steps_per_hour), 4)
        long_span = max(int(168 * steps_per_hour), short_span + 1)

        D = 13
        contexts = np.zeros((T, D), dtype=np.float32)

        idx = df.index
        has_datetime = hasattr(idx, "hour")

        for i in range(T):
            t = warmup + i

            # Bias
            contexts[i, 0] = 1.0

            # Short window statistics
            start_s = max(0, t - short_span)
            seg_short = target[start_s:t]
            if len(seg_short) >= 2:
                contexts[i, 1] = np.mean(seg_short)
                contexts[i, 2] = np.std(seg_short, ddof=1)
                half = len(seg_short) // 2
                contexts[i, 3] = np.mean(seg_short[half:]) - np.mean(seg_short[:half])
            else:
                contexts[i, 1] = target[t - 1] if t > 0 else 0.0
                contexts[i, 2] = 1e-4

            # Long window statistics
            start_l = max(0, t - long_span)
            seg_long = target[start_l:t]
            if len(seg_long) >= 2:
                contexts[i, 4] = np.mean(seg_long)
                contexts[i, 5] = np.std(seg_long, ddof=1)

            # Calendar features
            if has_datetime and t < len(idx):
                ts = idx[t]
                # Hour-of-day only for sub-daily frequencies;
                # for daily+ data hour is always 0 → constant features
                # (col 7 = cos(0) = 1 is collinear with bias).
                if steps_per_hour >= 1:
                    hour_frac = ts.hour + ts.minute / 60.0
                    contexts[i, 6] = np.sin(2 * np.pi * hour_frac / 24)
                    contexts[i, 7] = np.cos(2 * np.pi * hour_frac / 24)
                contexts[i, 8] = np.sin(2 * np.pi * ts.dayofweek / 7)
                contexts[i, 9] = np.cos(2 * np.pi * ts.dayofweek / 7)

            # Lag features
            if t >= 1:
                contexts[i, 10] = target[t - 1]
            if t >= 2:
                contexts[i, 11] = target[t - 2]
            if t >= 3:
                contexts[i, 12] = target[t - 3]

        # Standardise each column to zero-mean, unit-variance (except bias)
        for d in range(1, D):
            col = contexts[:, d]
            mu = np.mean(col)
            sigma = np.std(col)
            if sigma > 1e-8:
                contexts[:, d] = (col - mu) / sigma

        return contexts


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------

def load_real_dataset(name: str, cache_dir: str = "./cache/datasets") -> Optional[pd.DataFrame]:
    """Shortcut to load a dataset without explicitly creating a loader."""
    return RealDatasetLoader(cache_dir=cache_dir).load(name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    loader = RealDatasetLoader()

    print("Available datasets:", loader.list_available())
    print()

    for ds_name in loader.list_available():
        print(f"--- {ds_name} ---")
        df = loader.load(ds_name)
        if df is not None:
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns[:8])}{'...' if len(df.columns) > 8 else ''}")
            print(f"  Date range: {df.index.min()} to {df.index.max()}")
        else:
            print("  (download failed)")
        print()

    # Quick bandit-experiment test on the cheapest dataset
    print("=== Bandit experiment (AustralianElecDemand) ===")
    result = loader.prepare_bandit_experiment("AustralianElecDemand", num_specs=4)
    if result is not None:
        print(f"  contexts shape:      {result['contexts'].shape}")
        print(f"  scores_matrix shape: {result['scores_matrix'].shape}")
        print(f"  lookback windows:    {result['lookback_windows']}")
        print(f"  target_values shape: {result['target_values'].shape}")
        # Sanity: mean score per spec
        mean_scores = np.nanmean(result["scores_matrix"], axis=0)
        for w, s in zip(result["lookback_windows"], mean_scores):
            print(f"    window={w:>5d}  mean_IS={s:.2f}")
