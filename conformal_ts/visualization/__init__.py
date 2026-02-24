"""
Publication-Quality Plotting for Conformal Thompson Sampling.

This module provides visualization functions for analyzing and presenting
CTS experiment results, including:

- Cumulative regret curves
- Rolling coverage over time
- Specification selection heatmaps
- Interval width comparison
- Regime analysis
- Diagnostic scatter plots (non-stationarity vs CTS improvement)
- Multi-panel experiment summaries

All functions accept an optional ``ax`` parameter for subplot integration
and return the matplotlib Axes object they drew on. Use ``set_style()``
at the top of a script to configure matplotlib for publication-quality
figures.

Usage::

    from conformal_ts.visualization import (
        set_style, plot_cumulative_regret, plot_experiment_summary
    )

    set_style()
    fig, ax = plt.subplots()
    plot_cumulative_regret(results, methods=['cts', 'fixed_best', 'ensemble'], ax=ax)
    plt.show()
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple, Any, Union

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False


# ---------------------------------------------------------------------------
# Colour palette  -- colorblind-friendly (Wong 2011 + Tol muted)
# ---------------------------------------------------------------------------

METHOD_COLORS: Dict[str, str] = {
    # Primary methods
    'cts':          '#0072B2',   # blue
    'CTS':          '#0072B2',
    # Baselines
    'fixed_best':   '#E69F00',   # orange
    'Fixed':        '#E69F00',
    'fixed':        '#E69F00',
    'ensemble':     '#009E73',   # teal / green
    'Ensemble':     '#009E73',
    'random':       '#CC79A7',   # pink
    'Random':       '#CC79A7',
    'aci':          '#D55E00',   # vermillion
    'ACI':          '#D55E00',
    'oracle':       '#56B4E9',   # sky blue
    'Oracle':       '#56B4E9',
    'lightgbm':     '#F0E442',   # yellow
    'LightGBM':     '#F0E442',
    'ucb':          '#999999',   # grey
    'UCB':          '#999999',
    'round_robin':  '#882255',   # wine
    'RoundRobin':   '#882255',
}

# Ordered palette for unnamed methods (falls back here)
_PALETTE = [
    '#0072B2', '#E69F00', '#009E73', '#CC79A7',
    '#D55E00', '#56B4E9', '#F0E442', '#999999',
    '#882255', '#332288', '#44AA99', '#AA4499',
]


def _color_for(method: str, idx: int = 0) -> str:
    """Return a colour for *method*, falling back to the palette."""
    return METHOD_COLORS.get(method, _PALETTE[idx % len(_PALETTE)])


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

def set_style(
    context: str = 'paper',
    font_scale: float = 1.1,
    use_tex: bool = False,
) -> None:
    """
    Configure matplotlib / seaborn for publication-quality figures.

    Call this once at the beginning of a plotting script or notebook.

    Parameters
    ----------
    context : str
        One of ``'paper'``, ``'notebook'``, ``'talk'``, ``'poster'``.
        Passed to ``seaborn.set_context`` when seaborn is available.
    font_scale : float
        Multiplicative scaling for all font sizes.
    use_tex : bool
        If ``True``, enable LaTeX rendering (requires a TeX installation).
    """
    if _HAS_SEABORN:
        sns.set_context(context, font_scale=font_scale)
        sns.set_style('whitegrid', {
            'axes.edgecolor': '0.15',
            'axes.linewidth': 0.8,
            'grid.linestyle': '--',
            'grid.alpha': 0.35,
        })
    else:
        mpl.rcParams.update({
            'axes.grid': True,
            'grid.linestyle': '--',
            'grid.alpha': 0.35,
            'axes.edgecolor': '0.15',
            'axes.linewidth': 0.8,
        })

    mpl.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'font.size': 10 * font_scale,
        'axes.titlesize': 11 * font_scale,
        'axes.labelsize': 10 * font_scale,
        'xtick.labelsize': 9 * font_scale,
        'ytick.labelsize': 9 * font_scale,
        'legend.fontsize': 9 * font_scale,
        'legend.framealpha': 0.85,
        'legend.edgecolor': '0.6',
        'text.usetex': use_tex,
        'figure.figsize': (6.5, 4.0),
    })


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _ensure_ax(ax: Optional[plt.Axes]) -> plt.Axes:
    """Return the given Axes, or create a new figure if ``ax`` is None."""
    if ax is None:
        _, ax = plt.subplots()
    return ax


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute a centred rolling mean, padding edges with NaN."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) < window:
        return np.full_like(arr, np.nan)
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode='same')
    # Mark edges where the full window is not available
    half = window // 2
    smoothed[:half] = np.nan
    smoothed[-(half):] = np.nan
    return smoothed


def _extract_scores(
    results: Dict[str, Any],
    method: str,
    alpha: float = 0.10,
) -> np.ndarray:
    """
    Extract per-step interval scores for *method* from a results dict.

    Supports two common layouts emitted by the experiment runners:

    1. ``results[method]['scores']``  -- already-computed score array
    2. ``results[method]['lowers']``, ``results[method]['uppers']``,
       ``results[method]['targets']``  -- raw predictions that need scoring
    """
    data = results[method]

    if 'scores' in data:
        return np.asarray(data['scores'])

    # Compute from raw predictions
    from ..evaluation.metrics import interval_score as _is
    lowers = np.asarray(data['lowers'])
    uppers = np.asarray(data['uppers'])
    targets = np.asarray(data['targets'])
    return _is(lowers, uppers, targets, alpha)


def _extract_coverage_series(
    results: Dict[str, Any],
    method: str,
) -> np.ndarray:
    """Return a boolean array where ``True`` = target covered."""
    data = results[method]

    if 'coverages' in data:
        return np.asarray(data['coverages'], dtype=bool)

    lowers = np.asarray(data['lowers'])
    uppers = np.asarray(data['uppers'])
    targets = np.asarray(data['targets'])
    return (targets >= lowers) & (targets <= uppers)


def _extract_widths(
    results: Dict[str, Any],
    method: str,
) -> np.ndarray:
    """Return per-step interval widths."""
    data = results[method]

    if 'widths' in data:
        return np.asarray(data['widths'])

    lowers = np.asarray(data['lowers'])
    uppers = np.asarray(data['uppers'])
    return uppers - lowers


# ---------------------------------------------------------------------------
# 1. Cumulative regret
# ---------------------------------------------------------------------------

def plot_cumulative_regret(
    results: Dict[str, Any],
    methods: List[str],
    oracle_key: str = 'oracle',
    alpha: float = 0.10,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot cumulative regret curves for each method relative to an oracle.

    Regret at step *t* is defined as
    ``score(method, t) - score(oracle, t)`` (lower score is better, so
    positive regret means the method is worse than the oracle).

    Parameters
    ----------
    results : dict
        Experiment results dict keyed by method name.  Each entry must
        contain either a ``'scores'`` array or ``'lowers'``/``'uppers'``/
        ``'targets'`` arrays.
    methods : list of str
        Method names to plot (the oracle itself is excluded automatically).
    oracle_key : str
        Key in *results* for the oracle / best-in-hindsight baseline.
    alpha : float
        Miscoverage rate used for interval score computation.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created if ``None``.
    **kwargs
        Forwarded to ``ax.plot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ensure_ax(ax)

    oracle_scores = _extract_scores(results, oracle_key, alpha)

    for i, method in enumerate(methods):
        if method == oracle_key:
            continue
        scores = _extract_scores(results, method, alpha)
        n = min(len(scores), len(oracle_scores))
        regret = np.cumsum(scores[:n] - oracle_scores[:n])
        color = _color_for(method, i)
        ax.plot(
            np.arange(n), regret,
            label=method, color=color, linewidth=1.5,
            **kwargs,
        )

    ax.set_xlabel('Time step')
    ax.set_ylabel('Cumulative regret')
    ax.set_title('Cumulative Regret vs Oracle')
    ax.legend(frameon=True)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
    return ax


# ---------------------------------------------------------------------------
# 2. Coverage over time
# ---------------------------------------------------------------------------

def plot_coverage_over_time(
    results: Dict[str, Any],
    methods: List[str],
    target_alpha: float = 0.10,
    window: int = 100,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot rolling empirical coverage for each method.

    A horizontal dashed line marks the target coverage ``1 - target_alpha``.

    Parameters
    ----------
    results : dict
        Experiment results dict keyed by method name.
    methods : list of str
        Which methods to include.
    target_alpha : float
        Miscoverage level; the target line is drawn at ``1 - target_alpha``.
    window : int
        Rolling window size for the coverage average.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ensure_ax(ax)

    target_cov = 1.0 - target_alpha
    ax.axhline(
        target_cov, color='black', linewidth=1.0, linestyle='--',
        label=f'Target ({target_cov:.0%})', zorder=0,
    )

    for i, method in enumerate(methods):
        covered = _extract_coverage_series(results, method).astype(float)
        rolling_cov = _rolling_mean(covered, window)
        color = _color_for(method, i)
        ax.plot(
            np.arange(len(rolling_cov)), rolling_cov,
            label=method, color=color, linewidth=1.3,
        )

    ax.set_xlabel('Time step')
    ax.set_ylabel('Coverage rate')
    ax.set_title(f'Rolling Coverage (window={window})')
    ax.set_ylim(max(0.0, target_cov - 0.25), min(1.05, target_cov + 0.20))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.legend(frameon=True)
    return ax


# ---------------------------------------------------------------------------
# 3. Selection heatmap
# ---------------------------------------------------------------------------

def plot_selection_heatmap(
    selections: np.ndarray,
    spec_names: Optional[List[str]] = None,
    window: int = 50,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Heatmap showing which specification was selected over time.

    The x-axis represents time (binned into windows of size *window*), the
    y-axis lists specifications, and colour intensity shows the selection
    frequency within each window.

    Parameters
    ----------
    selections : array-like of int
        1-D array of selected specification indices, one per time step.
    spec_names : list of str, optional
        Human-readable labels for each specification.  If ``None``,
        generic labels ``Spec 0``, ``Spec 1``, ... are used.
    window : int
        Bin size along the time axis.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ensure_ax(ax)
    selections = np.asarray(selections)
    n_specs = int(selections.max()) + 1
    T = len(selections)
    n_bins = T // window + (1 if T % window else 0)

    if spec_names is None:
        spec_names = [f'Spec {k}' for k in range(n_specs)]

    # Build frequency matrix  (n_specs x n_bins)
    freq = np.zeros((n_specs, n_bins))
    for b in range(n_bins):
        start = b * window
        end = min(start + window, T)
        chunk = selections[start:end]
        counts = np.bincount(chunk, minlength=n_specs)
        freq[:, b] = counts / len(chunk)

    if _HAS_SEABORN:
        sns.heatmap(
            freq,
            ax=ax,
            cmap='YlOrRd',
            vmin=0, vmax=1,
            yticklabels=spec_names,
            cbar_kws={'label': 'Selection frequency'},
        )
    else:
        im = ax.imshow(freq, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_yticks(range(n_specs))
        ax.set_yticklabels(spec_names)
        plt.colorbar(im, ax=ax, label='Selection frequency')

    # x-tick labels: show original time indices at a few positions
    n_ticks = min(8, n_bins)
    tick_positions = np.linspace(0, n_bins - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(p * window) for p in tick_positions])

    ax.set_xlabel('Time step')
    ax.set_ylabel('Specification')
    ax.set_title('Specification Selection Over Time')
    return ax


# ---------------------------------------------------------------------------
# 4. Interval width over time
# ---------------------------------------------------------------------------

def plot_interval_width(
    results: Dict[str, Any],
    methods: List[str],
    window: int = 100,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot rolling average interval width for each method.

    Narrower intervals (at equal coverage) indicate higher efficiency.

    Parameters
    ----------
    results : dict
        Experiment results dict keyed by method name.
    methods : list of str
        Which methods to include.
    window : int
        Rolling window size.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ensure_ax(ax)

    for i, method in enumerate(methods):
        widths = _extract_widths(results, method)
        rolling_w = _rolling_mean(widths, window)
        color = _color_for(method, i)
        ax.plot(
            np.arange(len(rolling_w)), rolling_w,
            label=method, color=color, linewidth=1.3,
        )

    ax.set_xlabel('Time step')
    ax.set_ylabel('Interval width')
    ax.set_title(f'Rolling Interval Width (window={window})')
    ax.legend(frameon=True)
    return ax


# ---------------------------------------------------------------------------
# 5. Regime analysis
# ---------------------------------------------------------------------------

def plot_regime_analysis(
    scores_matrix: np.ndarray,
    spec_names: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Show which specification is best at each time step, revealing regimes.

    The plot has two layers:

    * Coloured background regions indicating the best specification at
      each step.
    * An overlaid line showing the performance gap between the best and
      worst specification (right y-axis), providing visual evidence for
      regime shifts.

    Parameters
    ----------
    scores_matrix : np.ndarray
        2-D array of shape ``(n_specs, T)`` where ``scores_matrix[k, t]``
        is the interval score for specification *k* at time *t*.
        Lower is better.
    spec_names : list of str, optional
        Labels for each specification.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ensure_ax(ax)
    scores_matrix = np.asarray(scores_matrix)
    n_specs, T = scores_matrix.shape

    if spec_names is None:
        spec_names = [f'Spec {k}' for k in range(n_specs)]

    best_spec = np.argmin(scores_matrix, axis=0)

    # Coloured regions: pcolormesh needs 2-D data -- stretch to a thin bar
    region_data = best_spec.reshape(1, -1)
    cmap = mpl.colors.ListedColormap(
        [_PALETTE[k % len(_PALETTE)] for k in range(n_specs)]
    )
    bounds = np.arange(-0.5, n_specs)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    ax.pcolormesh(
        np.arange(T + 1),
        [0, 1],
        region_data,
        cmap=cmap, norm=norm, alpha=0.45, shading='flat',
    )

    # Overlay: performance gap
    ax2 = ax.twinx()
    gap = np.max(scores_matrix, axis=0) - np.min(scores_matrix, axis=0)
    gap_smooth = _rolling_mean(gap, window=max(T // 50, 10))
    ax2.plot(
        np.arange(T), gap_smooth,
        color='black', linewidth=1.2, alpha=0.7,
        label='Best-worst gap',
    )
    ax2.set_ylabel('Score gap (max - min)')
    ax2.legend(loc='upper right', frameon=True)

    # Legend for specification colours
    handles = [
        mpl.patches.Patch(
            facecolor=_PALETTE[k % len(_PALETTE)], alpha=0.55,
            label=spec_names[k],
        )
        for k in range(n_specs)
    ]
    ax.legend(handles=handles, loc='upper left', frameon=True, ncol=2)

    ax.set_xlabel('Time step')
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.set_title('Best Specification Over Time (Regime Analysis)')
    return ax


# ---------------------------------------------------------------------------
# 6. Diagnostic correlation (non-stationarity vs CTS improvement)
# ---------------------------------------------------------------------------

def plot_diagnostic_correlation(
    reports_dict: Dict[str, Dict[str, Any]],
    improvements_dict: Dict[str, float],
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Scatter plot of non-stationarity index vs CTS improvement.

    Each point represents one dataset.  The plot visualises the core
    diagnostic finding: higher non-stationarity correlates with larger
    CTS improvement over a fixed baseline.

    Parameters
    ----------
    reports_dict : dict
        Mapping ``dataset_name -> diagnostic_report``.  Each report must
        contain a ``'non_stationarity_index'`` (or ``'nonstationarity_index'``)
        key with a numeric value.
    improvements_dict : dict
        Mapping ``dataset_name -> float`` giving the percentage improvement
        of CTS over the fixed baseline (positive = CTS is better).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ensure_ax(ax)

    xs, ys, labels = [], [], []
    for name in reports_dict:
        report = reports_dict[name]
        # Accept either spelling
        ns_key = (
            'non_stationarity_index'
            if 'non_stationarity_index' in report
            else 'nonstationarity_index'
        )
        if ns_key not in report:
            continue
        if name not in improvements_dict:
            continue
        xs.append(report[ns_key])
        ys.append(improvements_dict[name])
        labels.append(name)

    xs = np.array(xs)
    ys = np.array(ys)

    ax.scatter(xs, ys, s=60, color='#0072B2', edgecolors='white', zorder=3)

    # Label each point
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(
            label,
            (x, y),
            textcoords='offset points',
            xytext=(6, 4),
            fontsize=8,
        )

    # Trend line (OLS)
    if len(xs) >= 3:
        coef = np.polyfit(xs, ys, 1)
        trend_x = np.linspace(xs.min() * 0.9, xs.max() * 1.1, 50)
        trend_y = np.polyval(coef, trend_x)
        ax.plot(
            trend_x, trend_y,
            color='#D55E00', linewidth=1.2, linestyle='--',
            label=f'Trend (slope={coef[0]:.2f})',
        )
        ax.legend(frameon=True)

    ax.axhline(0, color='grey', linewidth=0.5, linestyle='-')
    ax.set_xlabel('Non-stationarity index')
    ax.set_ylabel('CTS improvement over fixed (%)')
    ax.set_title('Diagnostic: Non-Stationarity vs CTS Benefit')
    return ax


# ---------------------------------------------------------------------------
# 7. Multi-panel experiment summary
# ---------------------------------------------------------------------------

def plot_experiment_summary(
    results: Dict[str, Any],
    methods: List[str],
    oracle_key: str = 'oracle',
    target_alpha: float = 0.10,
    window: int = 100,
    selections: Optional[np.ndarray] = None,
    spec_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Multi-panel summary figure combining four diagnostic plots.

    Layout (2x2):

    * Top-left:     Cumulative regret
    * Top-right:    Rolling coverage
    * Bottom-left:  Selection heatmap (if *selections* provided, else
                    interval width comparison)
    * Bottom-right: Rolling interval width

    Parameters
    ----------
    results : dict
        Experiment results dict keyed by method name.
    methods : list of str
        Which methods to include.
    oracle_key : str
        Key for the oracle baseline in *results*.
    target_alpha : float
        Miscoverage level for the coverage plot.
    window : int
        Rolling window size for smoothing.
    selections : array-like of int, optional
        CTS specification selections (for the heatmap panel).  If ``None``
        the bottom-left panel shows interval width with a larger window.
    spec_names : list of str, optional
        Specification labels for the heatmap.
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Experiment Summary', fontsize=14, fontweight='bold', y=0.98)

    # Top-left: cumulative regret
    if oracle_key in results:
        plot_cumulative_regret(
            results, methods,
            oracle_key=oracle_key, alpha=target_alpha, ax=axes[0, 0],
        )
    else:
        # Without oracle, plot raw cumulative scores
        for i, method in enumerate(methods):
            scores = _extract_scores(results, method, target_alpha)
            ax = axes[0, 0]
            ax.plot(
                np.arange(len(scores)), np.cumsum(scores),
                label=method, color=_color_for(method, i), linewidth=1.3,
            )
        axes[0, 0].set_xlabel('Time step')
        axes[0, 0].set_ylabel('Cumulative score')
        axes[0, 0].set_title('Cumulative Interval Score')
        axes[0, 0].legend(frameon=True)

    # Top-right: coverage over time
    plot_coverage_over_time(
        results, methods,
        target_alpha=target_alpha, window=window, ax=axes[0, 1],
    )

    # Bottom-left: selection heatmap or wider-window width comparison
    if selections is not None:
        plot_selection_heatmap(
            selections, spec_names=spec_names,
            window=max(window, 50), ax=axes[1, 0],
        )
    else:
        plot_interval_width(
            results, methods, window=window * 2, ax=axes[1, 0],
        )
        axes[1, 0].set_title(f'Interval Width (window={window * 2})')

    # Bottom-right: interval width
    plot_interval_width(
        results, methods, window=window, ax=axes[1, 1],
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        fig.savefig(save_path)

    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Style
    'set_style',
    # Constants
    'METHOD_COLORS',
    # Individual plots
    'plot_cumulative_regret',
    'plot_coverage_over_time',
    'plot_selection_heatmap',
    'plot_interval_width',
    'plot_regime_analysis',
    'plot_diagnostic_correlation',
    # Multi-panel
    'plot_experiment_summary',
]
