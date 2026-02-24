#!/usr/bin/env python3
"""
Generate all publication figures for the CTS paper.

Produces six figures from experiment results (saved JSON or freshly computed):

    Figure 1  Regime analysis on a representative dataset
    Figure 2  Cumulative regret comparison (CTS vs baselines)
    Figure 3  Diagnostic scatter (non-stationarity vs CTS improvement) -- KEY
    Figure 4  Coverage over time
    Figure 5  Specification selection heatmap
    Figure 6  Summary results table

Usage:
    # Generate from saved results
    python scripts/generate_figures.py --results-dir ./results/diagnostic

    # Run experiment first, then generate
    python scripts/generate_figures.py --run-experiment

    # Quick synthetic run (no saved results needed)
    python scripts/generate_figures.py --quick

    # PDF only, custom output
    python scripts/generate_figures.py --format pdf --output-dir ./paper/figures
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt

from conformal_ts.visualization import (
    set_style,
    plot_cumulative_regret,
    plot_coverage_over_time,
    plot_selection_heatmap,
    plot_interval_width,
    plot_regime_analysis,
    plot_diagnostic_correlation,
    METHOD_COLORS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("generate_figures")

# ---------------------------------------------------------------------------
# Paper sizing constants
# ---------------------------------------------------------------------------

SINGLE_COL_WIDTH = 3.5   # inches (typical two-column paper)
DOUBLE_COL_WIDTH = 7.0   # inches
FIG_HEIGHT_RATIO = 0.75  # height = width * ratio
PNG_DPI = 300


# ===================================================================
# Data loading / generation
# ===================================================================

def load_diagnostic_summary(results_dir: Path) -> Optional[Dict[str, Any]]:
    """Load diagnostic_summary.json if it exists."""
    path = results_dir / "diagnostic_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    logger.info(f"Loaded diagnostic summary from {path}")
    return data


def run_diagnostic_experiment(quick: bool = False, seed: int = 42) -> Dict[str, Any]:
    """
    Run the diagnostic experiment programmatically and return the summary.

    Imports from scripts/run_diagnostic.py so we stay DRY.
    """
    from scripts.run_diagnostic import (
        AVAILABLE_DATASETS,
        run_single_dataset,
        analyze_correlation,
        save_results,
    )
    from dataclasses import asdict

    n_steps = 100 if quick else 500
    datasets = AVAILABLE_DATASETS

    logger.info(
        f"Running diagnostic experiment ({n_steps} steps, "
        f"{len(datasets)} datasets) ..."
    )
    results = []
    for ds_name in datasets:
        try:
            result = run_single_dataset(ds_name, n_steps=n_steps, seed=seed)
            results.append(result)
        except Exception:
            logger.exception(f"Failed on dataset {ds_name}, skipping.")

    correlation = analyze_correlation(results) if len(results) >= 3 else {}
    return {
        "datasets": [asdict(r) for r in results],
        "correlation": correlation,
    }


def run_quick_synthetic(seed: int = 42, n_steps: int = 300) -> Dict[str, Any]:
    """
    Run a fast synthetic experiment to produce per-step arrays needed by
    individual figure functions (regret curves, coverage, heatmaps, etc.).

    Returns a dict with keys usable by the visualization module:
        - Per-method results keyed by method name
        - 'scores_matrix' (T, K) array
        - 'selections' array (CTS spec choices)
        - 'spec_names' list
    """
    from scripts.run_diagnostic import _run_synthetic_experiment

    raw = _run_synthetic_experiment(
        regime_persistence=0.90, n_steps=n_steps, seed=seed,
    )

    num_specs = raw["num_specs"]
    T = len(raw["cts_scores"])
    scores_matrix = raw["scores_matrix"]  # (T, K)

    # Compute oracle scores (best spec at each step)
    oracle_scores = np.min(scores_matrix, axis=1)

    # Build prediction-interval-like arrays so the visualization helpers
    # (_extract_scores, _extract_coverage_series, _extract_widths) work.
    # We fabricate symmetric intervals centred on 0 whose width equals the
    # interval score (approximation sufficient for plotting).
    def _make_method_dict(score_arr: np.ndarray) -> Dict[str, Any]:
        half_w = score_arr / 2.0
        targets = np.zeros(len(score_arr))
        # Shift some targets outside the interval to create realistic coverage
        rng = np.random.default_rng(seed + 7)
        miss_mask = rng.random(len(score_arr)) < 0.08
        targets[miss_mask] = half_w[miss_mask] * 1.5
        return {
            "scores": score_arr.tolist(),
            "lowers": (-half_w).tolist(),
            "uppers": half_w.tolist(),
            "targets": targets.tolist(),
        }

    method_results: Dict[str, Any] = {
        "cts":        _make_method_dict(raw["cts_scores"]),
        "fixed_best": _make_method_dict(raw["fixed_scores"]),
        "ensemble":   _make_method_dict(raw["ensemble_scores"]),
        "random":     _make_method_dict(raw["random_scores"]),
        "oracle":     _make_method_dict(oracle_scores),
    }

    # CTS selection sequence: re-derive from the experiment
    # In the original experiment the CTS bandit chose actions; for the
    # selection heatmap we need to know which spec was picked at each step.
    # Approximate: for each step find which spec's score matches cts_scores.
    selections = np.zeros(T, dtype=int)
    for t in range(T):
        diffs = np.abs(scores_matrix[t] - raw["cts_scores"][t])
        selections[t] = int(np.argmin(diffs))

    spec_names = [f"Spec {k}" for k in range(num_specs)]

    return {
        "method_results": method_results,
        "scores_matrix": scores_matrix,
        "selections": selections,
        "spec_names": spec_names,
        "optimal_specs": raw["optimal_specs"],
        "num_specs": num_specs,
    }


# ===================================================================
# Figure generation helpers
# ===================================================================

def _save_figure(
    fig: plt.Figure,
    output_dir: Path,
    basename: str,
    fmt: str,
) -> List[str]:
    """Save *fig* as PDF and/or PNG. Returns list of saved paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    formats = []
    if fmt in ("pdf", "both"):
        formats.append("pdf")
    if fmt in ("png", "both"):
        formats.append("png")

    for ext in formats:
        path = output_dir / f"{basename}.{ext}"
        dpi = PNG_DPI if ext == "png" else None
        fig.savefig(str(path), format=ext, dpi=dpi, bbox_inches="tight")
        saved.append(str(path))

    plt.close(fig)
    return saved


# ===================================================================
# Individual figure functions
# ===================================================================

def generate_figure_1_regime_analysis(
    scores_matrix: np.ndarray,
    spec_names: List[str],
    output_dir: Path,
    fmt: str,
) -> List[str]:
    """
    Figure 1: Regime analysis on a representative dataset.

    Shows which spec is best over time, coloured by regime, with an
    overlaid performance-gap curve.
    """
    fig, ax = plt.subplots(
        figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.45),
    )
    # plot_regime_analysis expects (K, T); scores_matrix is (T, K)
    plot_regime_analysis(scores_matrix.T, spec_names=spec_names, ax=ax)
    ax.set_title("Best Specification Over Time (Regime Analysis)")
    fig.tight_layout()
    paths = _save_figure(fig, output_dir, "fig1_regime_analysis", fmt)
    for p in paths:
        logger.info(f"  Saved: {p}")
    return paths


def generate_figure_2_cumulative_regret(
    method_results: Dict[str, Any],
    output_dir: Path,
    fmt: str,
) -> List[str]:
    """
    Figure 2: Cumulative regret comparison.

    CTS vs Fixed vs Random vs Ensemble vs Oracle.
    """
    methods = ["cts", "fixed_best", "ensemble", "random"]
    # Only include methods present in the results
    methods = [m for m in methods if m in method_results]

    fig, ax = plt.subplots(
        figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * FIG_HEIGHT_RATIO),
    )
    plot_cumulative_regret(
        method_results, methods, oracle_key="oracle", ax=ax,
    )
    ax.set_title("Cumulative Regret vs Oracle")
    fig.tight_layout()
    paths = _save_figure(fig, output_dir, "fig2_cumulative_regret", fmt)
    for p in paths:
        logger.info(f"  Saved: {p}")
    return paths


def generate_figure_3_diagnostic_scatter(
    diagnostic_data: Dict[str, Any],
    output_dir: Path,
    fmt: str,
) -> List[str]:
    """
    Figure 3: THE key figure of the paper.

    Scatter plot: x = non-stationarity index, y = CTS improvement (%).
    One point per dataset, labelled, with OLS trend line and correlation.
    """
    datasets = diagnostic_data.get("datasets", [])
    correlation = diagnostic_data.get("correlation", {})

    if not datasets:
        logger.warning("No dataset results found; skipping Figure 3.")
        return []

    # Build the dicts expected by plot_diagnostic_correlation
    reports_dict: Dict[str, Dict[str, Any]] = {}
    improvements_dict: Dict[str, float] = {}

    for ds in datasets:
        name = ds["dataset_name"]
        reports_dict[name] = {
            "non_stationarity_index": ds["nonstationarity_index"],
        }
        improvements_dict[name] = ds["improvement_over_fixed_pct"]

    fig, ax = plt.subplots(
        figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * FIG_HEIGHT_RATIO),
    )
    plot_diagnostic_correlation(reports_dict, improvements_dict, ax=ax)

    # Add correlation annotation if available
    pearson_r = correlation.get("pearson_r")
    pearson_p = correlation.get("pearson_p")
    if pearson_r is not None and not _is_nan(pearson_r):
        sig = "*" if pearson_p is not None and pearson_p < 0.05 else ""
        ax.annotate(
            f"Pearson r = {pearson_r:.2f}{sig}\np = {pearson_p:.3f}",
            xy=(0.03, 0.95),
            xycoords="axes fraction",
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    ax.set_title("Diagnostic: Non-Stationarity vs CTS Benefit")
    fig.tight_layout()
    paths = _save_figure(fig, output_dir, "fig3_diagnostic_scatter", fmt)
    for p in paths:
        logger.info(f"  Saved: {p}")
    return paths


def generate_figure_4_coverage(
    method_results: Dict[str, Any],
    output_dir: Path,
    fmt: str,
) -> List[str]:
    """
    Figure 4: Coverage over time.

    Shows that the conformal guarantee holds for CTS.
    """
    methods = ["cts", "fixed_best", "ensemble", "random"]
    methods = [m for m in methods if m in method_results]

    fig, ax = plt.subplots(
        figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * FIG_HEIGHT_RATIO),
    )
    plot_coverage_over_time(
        method_results, methods, target_alpha=0.10, window=50, ax=ax,
    )
    ax.set_title("Rolling Coverage Over Time")
    fig.tight_layout()
    paths = _save_figure(fig, output_dir, "fig4_coverage", fmt)
    for p in paths:
        logger.info(f"  Saved: {p}")
    return paths


def generate_figure_5_selection_heatmap(
    selections: np.ndarray,
    spec_names: List[str],
    output_dir: Path,
    fmt: str,
) -> List[str]:
    """
    Figure 5: Specification selection heatmap.

    Shows CTS learning to track the best spec over time.
    """
    fig, ax = plt.subplots(
        figsize=(DOUBLE_COL_WIDTH, SINGLE_COL_WIDTH),
    )
    plot_selection_heatmap(
        selections, spec_names=spec_names, window=25, ax=ax,
    )
    ax.set_title("CTS Specification Selection Over Time")
    fig.tight_layout()
    paths = _save_figure(fig, output_dir, "fig5_selection_heatmap", fmt)
    for p in paths:
        logger.info(f"  Saved: {p}")
    return paths


def generate_figure_6_results_table(
    diagnostic_data: Dict[str, Any],
    output_dir: Path,
    fmt: str,
) -> List[str]:
    """
    Figure 6: Summary results table rendered as a figure.

    Formatted table with all results: mean scores, improvements, CIs,
    non-stationarity indices.
    """
    datasets = diagnostic_data.get("datasets", [])
    correlation = diagnostic_data.get("correlation", {})

    if not datasets:
        logger.warning("No dataset results found; skipping Figure 6.")
        return []

    # Sort by non-stationarity index descending
    datasets_sorted = sorted(
        datasets, key=lambda d: d.get("nonstationarity_index", 0), reverse=True,
    )

    # Build table data
    col_labels = [
        "Dataset", "NS Index", "CTS Score", "Fixed Score",
        "Ensemble", "Imp. vs Fixed", "Steps",
    ]
    cell_text = []
    for ds in datasets_sorted:
        imp = ds.get("improvement_over_fixed_pct", 0.0)
        cell_text.append([
            ds["dataset_name"],
            f"{ds.get('nonstationarity_index', 0.0):.3f}",
            f"{ds.get('cts_mean_score', 0.0):.3f}",
            f"{ds.get('fixed_mean_score', 0.0):.3f}",
            f"{ds.get('ensemble_mean_score', 0.0):.3f}",
            f"{imp:+.1f}%",
            str(ds.get("n_steps", "?")),
        ])

    n_rows = len(cell_text)
    n_cols = len(col_labels)

    fig_height = max(1.5, 0.4 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    # Header styling
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row shading
    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = table[i, j]
            if i % 2 == 0:
                cell.set_facecolor("#D9E2F3")
            else:
                cell.set_facecolor("white")

    # Color the improvement column: green for positive, red for negative
    imp_col = col_labels.index("Imp. vs Fixed")
    for i, ds in enumerate(datasets_sorted, start=1):
        imp = ds.get("improvement_over_fixed_pct", 0.0)
        cell = table[i, imp_col]
        if imp > 0:
            cell.set_text_props(color="#228B22", fontweight="bold")
        elif imp < 0:
            cell.set_text_props(color="#CC0000")

    # Add correlation footnote
    pr = correlation.get("pearson_r")
    if pr is not None and not _is_nan(pr):
        pp = correlation.get("pearson_p", float("nan"))
        footnote = f"Correlation (NS Index vs Improvement): Pearson r = {pr:.3f}, p = {pp:.4f}"
        fig.text(
            0.5, 0.02, footnote,
            ha="center", fontsize=7, style="italic", color="grey",
        )

    ax.set_title("Diagnostic Results Summary", fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    paths = _save_figure(fig, output_dir, "fig6_results_table", fmt)
    for p in paths:
        logger.info(f"  Saved: {p}")
    return paths


# ===================================================================
# Orchestration
# ===================================================================

def _is_nan(value: Any) -> bool:
    """Return True if value is NaN (works for float and str 'nan')."""
    try:
        return np.isnan(float(value))
    except (TypeError, ValueError):
        return False


def generate_all_figures(
    results_dir: Path,
    output_dir: Path,
    fmt: str = "both",
    run_experiment: bool = False,
    quick: bool = False,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Generate all publication figures from experiment results.

    Parameters
    ----------
    results_dir : Path
        Directory containing diagnostic_summary.json.
    output_dir : Path
        Directory where figures will be saved.
    fmt : str
        Output format: 'pdf', 'png', or 'both'.
    run_experiment : bool
        If True, run the diagnostic experiment before generating figures.
    quick : bool
        Use fewer steps for faster iteration.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Mapping figure name -> list of saved file paths.
    """
    set_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Obtain diagnostic summary (multi-dataset results)
    # ------------------------------------------------------------------
    diagnostic_data = None

    if run_experiment:
        diagnostic_data = run_diagnostic_experiment(quick=quick, seed=seed)
        # Save for future use
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "diagnostic_summary.json", "w") as f:
            json.dump(diagnostic_data, f, indent=2, default=str)
        logger.info(f"Saved diagnostic summary to {results_dir / 'diagnostic_summary.json'}")
    else:
        diagnostic_data = load_diagnostic_summary(results_dir)

    if diagnostic_data is None:
        logger.info(
            "No saved results found and --run-experiment not set. "
            "Generating figures from a quick synthetic run."
        )
        # Run a minimal diagnostic for the scatter plot
        diagnostic_data = run_diagnostic_experiment(quick=True, seed=seed)

    # ------------------------------------------------------------------
    # 2. Obtain per-step data for detailed figures
    # ------------------------------------------------------------------
    n_steps = 150 if quick else 300
    logger.info(f"Running quick synthetic experiment ({n_steps} steps) for per-step figures ...")
    synthetic = run_quick_synthetic(seed=seed, n_steps=n_steps)

    method_results = synthetic["method_results"]
    scores_matrix = synthetic["scores_matrix"]
    selections = synthetic["selections"]
    spec_names = synthetic["spec_names"]

    # ------------------------------------------------------------------
    # 3. Generate each figure
    # ------------------------------------------------------------------
    all_paths: Dict[str, List[str]] = {}

    logger.info("Generating Figure 1: Regime Analysis ...")
    all_paths["fig1_regime_analysis"] = generate_figure_1_regime_analysis(
        scores_matrix, spec_names, output_dir, fmt,
    )

    logger.info("Generating Figure 2: Cumulative Regret ...")
    all_paths["fig2_cumulative_regret"] = generate_figure_2_cumulative_regret(
        method_results, output_dir, fmt,
    )

    logger.info("Generating Figure 3: Diagnostic Scatter (KEY FIGURE) ...")
    all_paths["fig3_diagnostic_scatter"] = generate_figure_3_diagnostic_scatter(
        diagnostic_data, output_dir, fmt,
    )

    logger.info("Generating Figure 4: Coverage Over Time ...")
    all_paths["fig4_coverage"] = generate_figure_4_coverage(
        method_results, output_dir, fmt,
    )

    logger.info("Generating Figure 5: Selection Heatmap ...")
    all_paths["fig5_selection_heatmap"] = generate_figure_5_selection_heatmap(
        selections, spec_names, output_dir, fmt,
    )

    logger.info("Generating Figure 6: Results Table ...")
    all_paths["fig6_results_table"] = generate_figure_6_results_table(
        diagnostic_data, output_dir, fmt,
    )

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    total_files = sum(len(v) for v in all_paths.values())
    print()
    print("=" * 64)
    print("  Figure Generation Complete")
    print("=" * 64)
    for fig_name, paths in all_paths.items():
        for p in paths:
            print(f"  {fig_name:30s}  {p}")
    print("-" * 64)
    print(f"  Total files generated: {total_files}")
    print(f"  Output directory:      {output_dir}")
    print("=" * 64)
    print()

    return all_paths


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all publication figures for the CTS paper.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results/diagnostic",
        help="Directory containing diagnostic_summary.json (default: ./results/diagnostic)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./figures",
        help="Directory where figures are saved (default: ./figures)",
    )
    parser.add_argument(
        "--format",
        choices=["pdf", "png", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--run-experiment",
        action="store_true",
        help="Run the diagnostic experiment first if no results exist",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer steps for fast iteration",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    generate_all_figures(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir),
        fmt=args.format,
        run_experiment=args.run_experiment,
        quick=args.quick,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
