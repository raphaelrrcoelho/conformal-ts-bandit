"""
Bandit Experiment Runner for Specification Selection.

Provides a clean, reusable function for running the bandit competition
on ANY dataset's scores matrix.  Separates concerns:

1. **Scores matrix construction** -- build_scores_matrix_from_series()
   converts a raw univariate time series into a (T, K) interval-score
   matrix using rolling-window forecasters with different lookback windows.

2. **Bandit competition** -- run_bandit_experiment() takes a pre-computed
   (T, K) scores matrix and a (T, D) context matrix and simulates how
   each method (CTS, Fixed(k), Random, Ensemble, Oracle) would select
   specs and what scores they would receive.

This design means the same bandit runner can be called from
run_diagnostic.py, run_gefcom.py, or any other script, and the
interval-score differences between specs emerge naturally from real
forecasting pipelines rather than being manufactured.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Online reward normalizer (Welford's algorithm)
# ---------------------------------------------------------------------------

class _RunningNormalizer:
    """Welford online mean/variance for reward normalization."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def std(self) -> float:
        if self.n < 2:
            return 1.0
        return max(np.sqrt(self.M2 / self.n), 1e-8)

    def normalize(self, x: float) -> float:
        return (x - self.mean) / self.std()


# ---------------------------------------------------------------------------
# Random Fourier Features (RFF) lifter
# ---------------------------------------------------------------------------

class _RFFLifter:
    """Lift D-dim context to M-dim nonlinear feature space via Random Fourier Features.

    phi_rff(x) = sqrt(2/M) * cos(W @ x + b)

    W (M x D) and b (M,) are drawn once and fixed.
    """

    def __init__(self, input_dim: int, rff_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((rff_dim, input_dim))
        self.b = rng.uniform(0, 2 * np.pi, size=rff_dim)
        self.scale = np.sqrt(2.0 / rff_dim)

    def lift(self, context: np.ndarray) -> np.ndarray:
        """Map context (D,) -> lifted features (M,)."""
        return self.scale * np.cos(self.W @ context + self.b)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BanditExperimentConfig:
    """Configuration for the bandit specification selection experiment."""

    num_specs: int = 4
    feature_dim: int = 8
    warmup_rounds: int = 5
    exploration_variance: float = 1.0
    prior_precision: float = 1.0
    seed: int = 42

    # Train/test split: if > 0, first train_steps are warmup-only
    # (always round-robin, not counted in reported results).
    train_steps: int = 0

    # Rolling CV window for the CV-Fixed baseline.
    cv_window: int = 100

    # ACI learning rate (gamma in Gibbs & Candès, 2021).
    aci_gamma: float = 0.01

    # Sliding-window size for the TS posterior (None = no forgetting).
    window_size: Optional[int] = None

    # Random Fourier Features dimension (None = no lifting).
    rff_dim: Optional[int] = None

    # AgACI (Zaffran et al., ICML 2022)
    agaci_gammas: List[float] = field(default_factory=lambda: [0.001, 0.005, 0.01, 0.05, 0.1])
    agaci_eta: float = 0.1  # expert weight learning rate

    # FACI (Gibbs & Candès, JMLR 2023)
    faci_gamma_init: float = 0.01
    faci_sigma: float = 0.01  # multiplicative update rate
    faci_gamma_min: float = 0.001
    faci_gamma_max: float = 0.2

    # CTS-CG (CTS + coverage guarantee via ACI overlay)
    cts_cg_gamma: float = 0.01


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BanditExperimentResult:
    """Results from running the bandit competition on a scores matrix."""

    # Inputs
    scores_matrix: np.ndarray  # (T, K)
    contexts: np.ndarray       # (T, D)

    # Per-method score arrays (T,)
    cts_scores: np.ndarray
    fixed_scores: Dict[int, np.ndarray]   # fixed_scores[k] = scores when always picking spec k
    random_scores: np.ndarray
    ensemble_scores: np.ndarray
    oracle_scores: np.ndarray

    # CV-Fixed baseline: rolling cross-validated best spec (T,)
    cv_fixed_scores: np.ndarray

    # ACI baseline (Gibbs & Candès, 2021)
    aci_scores: np.ndarray           # (T,) spec-0 scores scaled by alpha adjustment
    aci_alpha_history: np.ndarray    # (T,) running alpha_t trajectory

    # Selection tracking
    cts_selections: np.ndarray   # (T,) which spec CTS picked
    optimal_specs: np.ndarray    # (T,) which spec was best (oracle)

    # Coverage tracking (optional, requires intervals_matrix)
    cts_covered: Optional[np.ndarray] = None  # (T,) bool: did CTS interval cover?

    # Width tracking
    cts_widths: Optional[np.ndarray] = None   # (T,) interval widths for CTS
    aci_widths: Optional[np.ndarray] = None   # (T,) interval widths for ACI
    aci_covered: Optional[np.ndarray] = None  # (T,) bool: did ACI interval cover?

    # AgACI (Zaffran et al., ICML 2022)
    agaci_scores: Optional[np.ndarray] = None
    agaci_alpha_history: Optional[np.ndarray] = None
    agaci_covered: Optional[np.ndarray] = None
    agaci_widths: Optional[np.ndarray] = None

    # FACI (Gibbs & Candès, JMLR 2023)
    faci_scores: Optional[np.ndarray] = None
    faci_alpha_history: Optional[np.ndarray] = None
    faci_covered: Optional[np.ndarray] = None
    faci_widths: Optional[np.ndarray] = None

    # CTS-CG (CTS + coverage guarantee)
    cts_cg_scores: Optional[np.ndarray] = None
    cts_cg_alpha_history: Optional[np.ndarray] = None
    cts_cg_covered: Optional[np.ndarray] = None
    cts_cg_widths: Optional[np.ndarray] = None

    def best_fixed_scores(self) -> np.ndarray:
        """Return scores for the best single fixed spec (in hindsight)."""
        best_k = min(self.fixed_scores, key=lambda k: float(np.mean(self.fixed_scores[k])))
        return self.fixed_scores[best_k]

    def best_fixed_spec(self) -> int:
        """Return index of the best single fixed spec (in hindsight)."""
        return min(self.fixed_scores, key=lambda k: float(np.mean(self.fixed_scores[k])))

    def improvement_over_best_fixed(self) -> float:
        """CTS improvement percentage over best fixed spec (lower score is better)."""
        best_fixed_mean = float(np.mean(self.best_fixed_scores()))
        cts_mean = float(np.mean(self.cts_scores))
        if best_fixed_mean == 0:
            return 0.0
        return 100.0 * (best_fixed_mean - cts_mean) / (best_fixed_mean + 1e-12)

    def median_fixed_scores(self) -> np.ndarray:
        """Return scores for the median-performing fixed spec (no hindsight)."""
        ranked = sorted(self.fixed_scores.keys(),
                        key=lambda k: float(np.mean(self.fixed_scores[k])))
        median_k = ranked[len(ranked) // 2]
        return self.fixed_scores[median_k]

    def mean_fixed_scores(self) -> np.ndarray:
        """Average score across all fixed specs (expected perf of random fixed choice)."""
        return np.mean([self.fixed_scores[k] for k in sorted(self.fixed_scores)], axis=0)

    def improvement_over_random(self) -> float:
        """CTS improvement % over random selection."""
        rand_mean = float(np.mean(self.random_scores))
        cts_mean = float(np.mean(self.cts_scores))
        if rand_mean == 0:
            return 0.0
        return 100.0 * (rand_mean - cts_mean) / (rand_mean + 1e-12)

    def summary(self) -> Dict[str, float]:
        """Return a summary dictionary of mean scores for all methods."""
        result = {
            "cts_mean": float(np.mean(self.cts_scores)),
            "oracle_mean": float(np.mean(self.oracle_scores)),
            "random_mean": float(np.mean(self.random_scores)),
            "ensemble_mean": float(np.mean(self.ensemble_scores)),
            "best_fixed_mean": float(np.mean(self.best_fixed_scores())),
            "best_fixed_spec": self.best_fixed_spec(),
            "cv_fixed_mean": float(np.mean(self.cv_fixed_scores)),
            "aci_mean": float(np.mean(self.aci_scores)),
            "improvement_over_best_fixed_pct": self.improvement_over_best_fixed(),
            "improvement_over_random_pct": self.improvement_over_random(),
        }
        if self.cts_covered is not None:
            result["cts_coverage"] = float(np.mean(self.cts_covered))
        if self.aci_covered is not None:
            result["aci_coverage"] = float(np.mean(self.aci_covered))
        # Width metrics
        for name, arr in [("cts", self.cts_widths), ("aci", self.aci_widths),
                          ("agaci", self.agaci_widths), ("faci", self.faci_widths),
                          ("cts_cg", self.cts_cg_widths)]:
            if arr is not None:
                result[f"{name}_mean_width"] = float(np.mean(arr))
        # New baselines
        for name, scores, covered in [
            ("agaci", self.agaci_scores, self.agaci_covered),
            ("faci", self.faci_scores, self.faci_covered),
            ("cts_cg", self.cts_cg_scores, self.cts_cg_covered),
        ]:
            if scores is not None:
                result[f"{name}_mean"] = float(np.mean(scores))
            if covered is not None:
                result[f"{name}_coverage"] = float(np.mean(covered))
        for k, scores in self.fixed_scores.items():
            result[f"fixed_{k}_mean"] = float(np.mean(scores))
        return result


# ---------------------------------------------------------------------------
# Main bandit experiment runner
# ---------------------------------------------------------------------------

def run_bandit_experiment(
    scores_matrix: np.ndarray,
    contexts: np.ndarray,
    config: BanditExperimentConfig,
    targets: Optional[np.ndarray] = None,
    intervals_matrix: Optional[np.ndarray] = None,
    point_forecasts: Optional[np.ndarray] = None,
) -> BanditExperimentResult:
    """
    Run the bandit specification selection competition.

    Given a pre-computed (T, K) scores matrix and (T, D) context matrix,
    simulates how each method would select specs and what scores they'd
    get.

    Methods
    -------
    - **CTS** : Linear Thompson Sampling with context.  During warmup it
      cycles through specs round-robin; afterwards it samples from the
      posterior.  Reward = -interval_score (higher is better).
    - **Fixed(k)** : Always pick spec *k* (run for every k).
    - **CV-Fixed** : Cross-validated fixed -- at each step, pick the spec
      with the lowest average score in a rolling validation window.
    - **ACI** : Adaptive Conformal Inference (Gibbs & Candès, 2021) --
      uses spec 0 but adapts the miscoverage level alpha over time.
    - **AgACI** : Aggregated ACI (Zaffran et al., ICML 2022) --
      K experts with different gamma values, aggregated via exponential
      weights.
    - **FACI** : Fully Adaptive CI (Gibbs & Candès, JMLR 2023) --
      ACI with adaptive gamma via multiplicative updates.
    - **CTS-CG** : CTS with Coverage Guarantee -- CTS spec selection
      overlaid with ACI alpha adaptation for provable coverage.
    - **Random** : Uniform random selection each step.
    - **Ensemble** : Average score across all specs each step.
    - **Oracle** : Best spec at each step (lower bound on achievable
      score; requires future knowledge).

    Parameters
    ----------
    scores_matrix : ndarray, shape (T, K)
        Pre-computed interval scores.  Lower is better.
    contexts : ndarray, shape (T, D)
        Context / feature vectors for each timestep.
    config : BanditExperimentConfig
        Experiment hyper-parameters.
    targets : ndarray, shape (T,), optional
        True values for coverage tracking.
    intervals_matrix : ndarray, shape (T, K, 2), optional
        Lower/upper bounds per spec per timestep.  When provided together
        with *targets*, coverage is tracked for CTS's selected spec.
    point_forecasts : ndarray, shape (T,), optional
        Shared point forecast for ACI-family methods.  When None, falls
        back to midpoint of spec-0 interval.

    Returns
    -------
    BanditExperimentResult
    """
    from conformal_ts.models.linear_ts import LinearThompsonSampling

    T, K = scores_matrix.shape
    D = contexts.shape[1]

    assert contexts.shape[0] == T, (
        f"scores_matrix has {T} rows but contexts has {contexts.shape[0]}"
    )
    assert K == config.num_specs, (
        f"scores_matrix has {K} specs but config says {config.num_specs}"
    )
    assert D == config.feature_dim, (
        f"contexts has {D} features but config says {config.feature_dim}"
    )

    if intervals_matrix is not None:
        assert intervals_matrix.shape == (T, K, 2), (
            f"intervals_matrix shape {intervals_matrix.shape} != expected "
            f"({T}, {K}, 2)"
        )
    if targets is not None:
        assert targets.shape == (T,), (
            f"targets shape {targets.shape} != expected ({T},)"
        )

    # --- optional RFF lifting ---
    lifter: Optional[_RFFLifter] = None
    bandit_dim = D
    if config.rff_dim is not None:
        lifter = _RFFLifter(D, config.rff_dim, seed=config.seed)
        bandit_dim = config.rff_dim

    # --- set up bandit ---
    bandit = LinearThompsonSampling(
        num_actions=K,
        feature_dim=bandit_dim,
        prior_precision=config.prior_precision,
        exploration_variance=config.exploration_variance,
        seed=config.seed,
        window_size=config.window_size,
    )

    # Effective warmup = max(config warmup, train_steps)
    train_steps = max(config.train_steps, 0)
    warmup = max(min(config.warmup_rounds, T // 5), train_steps)
    rng = np.random.default_rng(config.seed + 999)
    normalizer = _RunningNormalizer()

    # --- allocate FULL-length arrays (T) for the simulation loop ---
    cts_scores_full = np.zeros(T)
    cts_selections_full = np.zeros(T, dtype=int)
    random_scores_full = np.zeros(T)
    ensemble_scores_full = np.zeros(T)
    oracle_scores_full = np.zeros(T)
    optimal_specs_full = np.zeros(T, dtype=int)
    cv_fixed_scores_full = np.zeros(T)
    aci_scores_full = np.zeros(T)
    aci_alpha_full = np.zeros(T)

    # Fixed baselines: one array per spec
    fixed_scores_full: Dict[int, np.ndarray] = {
        k: np.zeros(T) for k in range(K)
    }

    # Coverage tracking
    track_coverage = (targets is not None and intervals_matrix is not None)
    cts_covered_full: Optional[np.ndarray] = None
    if track_coverage:
        cts_covered_full = np.zeros(T, dtype=bool)

    # ACI state
    from collections import deque as _deque
    from conformal_ts.evaluation.metrics import interval_score as _interval_score
    alpha_nominal = 0.10  # default nominal miscoverage
    alpha_t = alpha_nominal
    cv_window = config.cv_window

    # Width / coverage arrays for ACI and CTS
    aci_widths_full = np.zeros(T)
    aci_covered_full = np.zeros(T, dtype=bool)
    cts_widths_full = np.zeros(T)

    # Shared residual buffer for all ACI-family methods (ACI, AgACI, FACI, CTS-CG).
    # Residuals are y_t - pf_t, independent of each method's alpha.
    residual_buffer: _deque = _deque(maxlen=50)

    # Helper: get point forecast at time t
    def _get_pf(t_idx: int) -> float:
        if point_forecasts is not None:
            pf = point_forecasts[t_idx]
            if not np.isnan(pf):
                return float(pf)
        if intervals_matrix is not None:
            return 0.5 * (intervals_matrix[t_idx, 0, 0] + intervals_matrix[t_idx, 0, 1])
        return 0.0

    # Helper: build ACI interval from residual buffer
    def _build_aci_interval(pf_t: float, res_buf: _deque, alpha_curr: float,
                            fallback_hw: float):
        if len(res_buf) >= 2:
            residuals = np.array(res_buf)
            n_res = len(residuals)
            q_idx = int(np.ceil((n_res + 1) * (1 - alpha_curr)))
            q_idx = min(q_idx, n_res) - 1  # 0-indexed
            q = float(np.sort(np.abs(residuals))[max(q_idx, 0)])
        else:
            q = fallback_hw
        return (pf_t - q, pf_t + q)

    # --- AgACI state (Zaffran et al., ICML 2022) ---
    n_experts = len(config.agaci_gammas)
    agaci_alpha_experts = np.full(n_experts, alpha_nominal)  # per-expert alpha_t
    agaci_weights = np.ones(n_experts) / n_experts  # normalized weights
    agaci_scores_full = np.zeros(T)
    agaci_alpha_full = np.zeros(T)
    agaci_covered_full = np.zeros(T, dtype=bool)
    agaci_widths_full = np.zeros(T)

    # --- FACI state (Gibbs & Candès, JMLR 2023) ---
    faci_alpha_t = alpha_nominal
    faci_gamma_t = config.faci_gamma_init
    faci_scores_full = np.zeros(T)
    faci_alpha_full = np.zeros(T)
    faci_covered_full = np.zeros(T, dtype=bool)
    faci_widths_full = np.zeros(T)

    # --- CTS-CG state (CTS + coverage guarantee) ---
    cts_cg_alpha_t = alpha_nominal
    cts_cg_scores_full = np.zeros(T)
    cts_cg_alpha_full = np.zeros(T)
    cts_cg_covered_full = np.zeros(T, dtype=bool)
    cts_cg_widths_full = np.zeros(T)

    # --- simulation loop ---
    for t in range(T):
        ctx = contexts[t]
        row = scores_matrix[t]

        # Oracle
        best_k = int(np.argmin(row))
        optimal_specs_full[t] = best_k
        oracle_scores_full[t] = row[best_k]

        # Optionally lift context for the bandit
        bandit_ctx = lifter.lift(ctx) if lifter is not None else ctx

        # CTS
        if t < warmup:
            cts_action = t % K
        else:
            cts_action = bandit.select_action(bandit_ctx)
        cts_selections_full[t] = cts_action
        cts_scores_full[t] = row[cts_action]
        # Full-information update: normalize all K rewards and update
        # every arm so CTS learns at the same rate as CV-Fixed.
        raw_rewards = -row  # negative scores (higher is better)
        # Normalize with OLD stats (predict-then-update pattern),
        # then update the normalizer.
        normalized = [np.clip(normalizer.normalize(v), -3.0, 3.0)
                      for v in raw_rewards]
        for val in raw_rewards:
            normalizer.update(val)
        bandit.update_all_arms(bandit_ctx, normalized, selected_action=cts_action)

        # Coverage + width tracking for CTS
        if track_coverage:
            lo = intervals_matrix[t, cts_action, 0]
            hi = intervals_matrix[t, cts_action, 1]
            cts_covered_full[t] = (lo <= targets[t] <= hi)
            cts_widths_full[t] = hi - lo

        # Fixed baselines
        for k in range(K):
            fixed_scores_full[k][t] = row[k]

        # CV-Fixed baseline: pick spec with lowest mean score in
        # the last cv_window steps.
        if t < cv_window:
            # Not enough history -- fall back to round-robin
            cv_action = t % K
        else:
            window_scores = scores_matrix[t - cv_window:t]  # (cv_window, K)
            cv_action = int(np.argmin(window_scores.mean(axis=0)))
        cv_fixed_scores_full[t] = row[cv_action]

        # --- Point forecast for ACI-family methods ---
        pf_t = _get_pf(t)
        fallback_hw = 0.5 * (intervals_matrix[t, 0, 1] - intervals_matrix[t, 0, 0]) if track_coverage else 1.0

        # ===== ACI baseline (Gibbs & Candès, 2021) =====
        aci_alpha_full[t] = alpha_t

        if track_coverage:
            aci_lo, aci_hi = _build_aci_interval(pf_t, residual_buffer, alpha_t, fallback_hw)
            aci_widths_full[t] = aci_hi - aci_lo
            aci_scores_full[t] = _interval_score(
                np.array([aci_lo]), np.array([aci_hi]),
                np.array([targets[t]]), alpha=alpha_nominal,
            )[0]
            aci_err_t = 0.0 if (aci_lo <= targets[t] <= aci_hi) else 1.0
            aci_covered_full[t] = (aci_err_t == 0.0)
        else:
            aci_scores_full[t] = row[0]
            if t > 0:
                past = scores_matrix[:t + 1, 0]
                aci_err_t = 0.0 if row[0] <= float(np.median(past)) else 1.0
            else:
                aci_err_t = 0.0

        alpha_t = alpha_t + config.aci_gamma * (alpha_nominal - aci_err_t)
        alpha_t = float(np.clip(alpha_t, 1e-6, 1.0 - 1e-6))

        # ===== AgACI (Zaffran et al., ICML 2022) =====
        # Aggregated alpha = weighted combination of expert alphas
        agaci_alpha_agg = float(np.dot(agaci_weights, agaci_alpha_experts))
        agaci_alpha_agg = float(np.clip(agaci_alpha_agg, 1e-6, 1.0 - 1e-6))
        agaci_alpha_full[t] = agaci_alpha_agg

        if track_coverage:
            ag_lo, ag_hi = _build_aci_interval(pf_t, residual_buffer, agaci_alpha_agg, fallback_hw)
            agaci_widths_full[t] = ag_hi - ag_lo
            agaci_scores_full[t] = _interval_score(
                np.array([ag_lo]), np.array([ag_hi]),
                np.array([targets[t]]), alpha=alpha_nominal,
            )[0]
            ag_err = 0.0 if (ag_lo <= targets[t] <= ag_hi) else 1.0
            agaci_covered_full[t] = (ag_err == 0.0)

            # Update each expert's alpha and compute per-expert loss
            y_t = targets[t]
            for e in range(n_experts):
                gamma_e = config.agaci_gammas[e]
                # Per-expert interval for loss computation
                e_lo, e_hi = _build_aci_interval(
                    pf_t, residual_buffer, agaci_alpha_experts[e], fallback_hw
                )
                # Interval score as expert loss — differentiates experts
                # by their actual interval quality (width + coverage penalty)
                loss_e = _interval_score(
                    np.array([e_lo]), np.array([e_hi]),
                    np.array([y_t]), alpha=alpha_nominal,
                )[0]
                agaci_weights[e] *= np.exp(-config.agaci_eta * loss_e)
                # Update expert alpha via ACI rule
                e_err = 0.0 if (e_lo <= y_t <= e_hi) else 1.0
                agaci_alpha_experts[e] += gamma_e * (alpha_nominal - e_err)
                agaci_alpha_experts[e] = float(np.clip(agaci_alpha_experts[e], 1e-6, 1.0 - 1e-6))

            # Renormalize weights
            w_sum = agaci_weights.sum()
            if w_sum > 0:
                agaci_weights /= w_sum
            else:
                agaci_weights = np.ones(n_experts) / n_experts
        else:
            agaci_scores_full[t] = row[0]

        # ===== FACI (Gibbs & Candès, JMLR 2023) =====
        faci_alpha_full[t] = faci_alpha_t

        if track_coverage:
            f_lo, f_hi = _build_aci_interval(pf_t, residual_buffer, faci_alpha_t, fallback_hw)
            faci_widths_full[t] = f_hi - f_lo
            faci_scores_full[t] = _interval_score(
                np.array([f_lo]), np.array([f_hi]),
                np.array([targets[t]]), alpha=alpha_nominal,
            )[0]
            f_err = 0.0 if (f_lo <= targets[t] <= f_hi) else 1.0
            faci_covered_full[t] = (f_err == 0.0)

            # Save pre-update alpha for gradient computation
            faci_alpha_pre = faci_alpha_t

            # Update alpha
            faci_alpha_t = faci_alpha_t + faci_gamma_t * (alpha_nominal - f_err)
            faci_alpha_t = float(np.clip(faci_alpha_t, 1e-6, 1.0 - 1e-6))

            # Adaptive gamma update (Gibbs & Candès, JMLR 2023, Algorithm 2)
            # gradient = -(alpha - err_t) * (alpha_t_pre - alpha) / gamma_t
            gradient_t = -(alpha_nominal - f_err) * (faci_alpha_pre - alpha_nominal) / (faci_gamma_t + 1e-8)
            faci_gamma_t = faci_gamma_t * np.exp(-config.faci_sigma * gradient_t)
            faci_gamma_t = float(np.clip(faci_gamma_t, config.faci_gamma_min, config.faci_gamma_max))
        else:
            faci_scores_full[t] = row[0]

        # ===== CTS-CG (CTS + coverage guarantee) =====
        # CTS selects the spec; ACI alpha adaptation provides coverage guarantee.
        # Use the CTS-selected spec's midpoint as the point forecast, and
        # build a conformal interval around it with the adaptive alpha.
        cts_cg_alpha_full[t] = cts_cg_alpha_t

        if track_coverage:
            # Point forecast from CTS-selected spec
            cg_pf = 0.5 * (intervals_matrix[t, cts_action, 0] + intervals_matrix[t, cts_action, 1])
            cg_lo, cg_hi = _build_aci_interval(cg_pf, residual_buffer, cts_cg_alpha_t, fallback_hw)
            cts_cg_widths_full[t] = cg_hi - cg_lo
            cts_cg_scores_full[t] = _interval_score(
                np.array([cg_lo]), np.array([cg_hi]),
                np.array([targets[t]]), alpha=alpha_nominal,
            )[0]
            cg_err = 0.0 if (cg_lo <= targets[t] <= cg_hi) else 1.0
            cts_cg_covered_full[t] = (cg_err == 0.0)

            # ACI alpha update
            cts_cg_alpha_t = cts_cg_alpha_t + config.cts_cg_gamma * (alpha_nominal - cg_err)
            cts_cg_alpha_t = float(np.clip(cts_cg_alpha_t, 1e-6, 1.0 - 1e-6))
        else:
            cts_cg_scores_full[t] = row[cts_action]

        # Update shared residual buffer (used by all ACI-family methods)
        if track_coverage:
            residual_buffer.append(targets[t] - pf_t)

        # Random
        random_scores_full[t] = row[rng.integers(0, K)]

        # Ensemble (average across specs)
        ensemble_scores_full[t] = float(np.mean(row))

    # --- apply train/test split ---
    s = train_steps  # start index for reported results
    scores_out = scores_matrix[s:]
    contexts_out = contexts[s:]

    cts_scores = cts_scores_full[s:]
    cts_selections = cts_selections_full[s:]
    random_scores = random_scores_full[s:]
    ensemble_scores = ensemble_scores_full[s:]
    oracle_scores = oracle_scores_full[s:]
    optimal_specs = optimal_specs_full[s:]
    cv_fixed_scores = cv_fixed_scores_full[s:]
    aci_scores = aci_scores_full[s:]
    aci_alpha_history = aci_alpha_full[s:]
    fixed_scores: Dict[int, np.ndarray] = {
        k: v[s:] for k, v in fixed_scores_full.items()
    }
    cts_covered: Optional[np.ndarray] = None
    if cts_covered_full is not None:
        cts_covered = cts_covered_full[s:]

    # Slice new arrays
    cts_widths = cts_widths_full[s:] if track_coverage else None
    aci_widths = aci_widths_full[s:] if track_coverage else None
    aci_covered = aci_covered_full[s:] if track_coverage else None
    agaci_scores = agaci_scores_full[s:]
    agaci_alpha_history = agaci_alpha_full[s:]
    agaci_covered = agaci_covered_full[s:] if track_coverage else None
    agaci_widths = agaci_widths_full[s:] if track_coverage else None
    faci_scores = faci_scores_full[s:]
    faci_alpha_history = faci_alpha_full[s:]
    faci_covered = faci_covered_full[s:] if track_coverage else None
    faci_widths = faci_widths_full[s:] if track_coverage else None
    cts_cg_scores = cts_cg_scores_full[s:]
    cts_cg_alpha_history = cts_cg_alpha_full[s:]
    cts_cg_covered = cts_cg_covered_full[s:] if track_coverage else None
    cts_cg_widths = cts_cg_widths_full[s:] if track_coverage else None

    return BanditExperimentResult(
        scores_matrix=scores_out,
        contexts=contexts_out,
        cts_scores=cts_scores,
        fixed_scores=fixed_scores,
        random_scores=random_scores,
        ensemble_scores=ensemble_scores,
        oracle_scores=oracle_scores,
        cv_fixed_scores=cv_fixed_scores,
        aci_scores=aci_scores,
        aci_alpha_history=aci_alpha_history,
        cts_selections=cts_selections,
        optimal_specs=optimal_specs,
        cts_covered=cts_covered,
        cts_widths=cts_widths,
        aci_widths=aci_widths,
        aci_covered=aci_covered,
        agaci_scores=agaci_scores,
        agaci_alpha_history=agaci_alpha_history,
        agaci_covered=agaci_covered,
        agaci_widths=agaci_widths,
        faci_scores=faci_scores,
        faci_alpha_history=faci_alpha_history,
        faci_covered=faci_covered,
        faci_widths=faci_widths,
        cts_cg_scores=cts_cg_scores,
        cts_cg_alpha_history=cts_cg_alpha_history,
        cts_cg_covered=cts_cg_covered,
        cts_cg_widths=cts_cg_widths,
    )


# ---------------------------------------------------------------------------
# Helper: build scores matrix from a univariate time series
# ---------------------------------------------------------------------------

def build_scores_matrix_from_series(
    series: np.ndarray,
    lookback_windows: List[int],
    alpha: float = 0.10,
    min_history: int = 50,
    return_intervals: bool = False,
) -> Tuple[np.ndarray, ...]:
    """
    Build a (T', K) scores matrix from a univariate time series.

    For each timestep *t* (starting from ``min_history``) and each
    lookback window *w_k*:

    1. Compute rolling mean and std over the last *w_k* observations.
    2. Form a prediction interval::

           [mean - z_{1-alpha/2} * std,  mean + z_{1-alpha/2} * std]

    3. Score against the true observation at *t* using the interval
       score (Gneiting & Raftery, 2007).

    Parameters
    ----------
    series : ndarray, shape (T,)
        Raw univariate time series.
    lookback_windows : list of int
        Each element defines a "spec" -- the lookback window length.
    alpha : float
        Miscoverage rate for the prediction interval (default 0.10
        for 90 % intervals).
    min_history : int
        Minimum number of observations before scoring begins.  The
        returned arrays start at index ``min_history`` of the original
        series.
    return_intervals : bool
        If True, also return ``targets`` and ``intervals_matrix`` so
        that downstream code can compute coverage.

    Returns
    -------
    scores_matrix : ndarray, shape (T', K)
        Interval scores.  T' = T - min_history.
    contexts : ndarray, shape (T', D)
        Context features derived from the series history.  D = 8.
    targets : ndarray, shape (T',)
        True values at each scored timestep.  Only returned when
        *return_intervals* is True.
    intervals_matrix : ndarray, shape (T', K, 2)
        ``intervals_matrix[t, k]`` = ``[lower, upper]`` for spec *k*
        at timestep *t*.  Only returned when *return_intervals* is True.
    """
    from conformal_ts.evaluation.metrics import interval_score

    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)

    T = len(series)
    K = len(lookback_windows)
    T_prime = T - min_history

    if T_prime <= 0:
        raise ValueError(
            f"Series length {T} is <= min_history {min_history}; "
            "nothing to score."
        )

    scores_matrix = np.zeros((T_prime, K))
    feature_dim = 8
    contexts = np.zeros((T_prime, feature_dim))
    targets_arr = np.zeros(T_prime)
    intervals_arr = np.zeros((T_prime, K, 2))

    for idx in range(T_prime):
        t = min_history + idx
        y_true = series[t]
        targets_arr[idx] = y_true

        # --- build context features from history ---
        ctx = np.zeros(feature_dim)
        ctx[0] = 1.0  # bias

        # Short-window statistics (last 5)
        short_win = series[max(0, t - 5):t]
        if len(short_win) > 0:
            ctx[1] = np.mean(short_win)
            ctx[2] = np.std(short_win) + 1e-8
        # Medium-window statistics (last 20)
        med_win = series[max(0, t - 20):t]
        if len(med_win) > 0:
            ctx[3] = np.mean(med_win)
            ctx[4] = np.std(med_win) + 1e-8
        # Long-window statistics (last 50)
        long_win = series[max(0, t - 50):t]
        if len(long_win) > 0:
            ctx[5] = np.mean(long_win)
            ctx[6] = np.std(long_win) + 1e-8
        # Last value
        ctx[7] = series[t - 1]

        contexts[idx] = ctx

        # --- score each spec ---
        for k, w in enumerate(lookback_windows):
            window = series[max(0, t - w):t]
            mu = float(np.mean(window))
            sigma = float(np.std(window)) + 1e-8

            lower_k = mu - z * sigma
            upper_k = mu + z * sigma

            intervals_arr[idx, k, 0] = lower_k
            intervals_arr[idx, k, 1] = upper_k

            scores_matrix[idx, k] = interval_score(
                np.array([lower_k]),
                np.array([upper_k]),
                np.array([y_true]),
                alpha=alpha,
            )[0]

    if return_intervals:
        return scores_matrix, contexts, targets_arr, intervals_arr

    return scores_matrix, contexts


# ---------------------------------------------------------------------------
# Helper: build scores matrix with conformal calibration (CQR-style)
# ---------------------------------------------------------------------------

def build_scores_matrix_with_cqr(
    series: np.ndarray,
    lookback_windows: Optional[List[int]] = None,
    alpha: float = 0.10,
    min_history: int = 100,
    calibration_window: int = 50,
    forecasters: Optional[List] = None,
    calibration_windows: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Build a (T', K) scores matrix using conformalized prediction intervals.

    Instead of the simple ``mean +/- z * std`` intervals used by
    :func:`build_scores_matrix_from_series`, this function computes
    proper conformal intervals following the split-conformal / CQR
    philosophy:

    1. Point forecast from either a :class:`ForecasterSpec` or a
       rolling mean over the last *w_k* values.
    2. Residuals are collected in a rolling calibration set of size
       *calibration_window*.
    3. The prediction interval at time *t* is::

           point +/- quantile(|residuals|, 1 - alpha)

       where the quantile is taken over the calibration window.

    This guarantees approximate marginal coverage of ``1 - alpha`` once
    the calibration window is full, without Gaussianity assumptions.

    Operating modes
    ---------------
    **Forecaster mode** (``forecasters`` provided, no ``calibration_windows``):
        K = len(forecasters).  Each spec is a different point forecaster,
        all sharing the same ``calibration_window``.

    **Calibration-selection mode** (``calibration_windows`` provided):
        K = len(calibration_windows).  A single ensemble point forecast
        (mean of all ``forecasters``, or rolling-mean if none given) is
        shared by every spec; the specs differ only in how many recent
        residuals they use for the conformal quantile.  This lets the
        bandit select the *adaptation speed* of the conformal calibration.

    **Lookback mode** (``lookback_windows`` only, backward compat):
        K = len(lookback_windows).  Rolling-mean point forecasts with
        a shared ``calibration_window``.

    Parameters
    ----------
    series : ndarray, shape (T,)
        Raw univariate time series.
    lookback_windows : list of int, optional
        Each element defines a "spec" -- the lookback window length.
        Used when *forecasters* is not provided (backward compat).
    alpha : float
        Miscoverage rate (default 0.10 for 90 % intervals).
    min_history : int
        Minimum number of observations before scoring begins (must be
        large enough that each lookback window and the calibration
        window have data).  Default 100.
    calibration_window : int
        Number of recent residuals used to compute the conformal
        quantile.  Default 50.  Ignored when *calibration_windows*
        is provided.
    forecasters : list of ForecasterSpec, optional
        When provided, use these diverse forecasters instead of
        rolling-mean lookback windows.  Each forecaster's
        ``forecast_all(series)`` is called once to pre-compute all
        point forecasts.
    calibration_windows : list of int, optional
        When provided, each element defines a spec whose conformal
        quantile is computed over that many recent residuals.  All
        specs share the same ensemble point forecast.

    Returns
    -------
    scores_matrix : ndarray, shape (T', K)
        Interval scores.  T' = T - min_history.
    contexts : ndarray, shape (T', D)
        Context features derived from the series history (D = 8).
    targets : ndarray, shape (T',)
        True values at each scored timestep.
    intervals_matrix : ndarray, shape (T', K, 2)
        ``intervals_matrix[t, k]`` = ``[lower, upper]`` for spec *k*
        at timestep *t*.
    point_forecasts : ndarray, shape (T',), or None
        Shared ensemble point forecast at each scored timestep.
        Available in calibration-selection mode (ensemble of diverse
        forecasters) and forecaster mode (nanmean across forecasters).
        None in lookback mode (no single shared forecast).
    """
    from conformal_ts.evaluation.metrics import interval_score
    from collections import deque

    T = len(series)

    # --- Determine mode and pre-compute forecasts ---
    calib_mode = calibration_windows is not None

    if calib_mode:
        # Calibration-selection mode: K = number of calibration windows,
        # single ensemble point forecast.
        K = len(calibration_windows)

        if forecasters is not None:
            all_fc_forecasts = [fc.forecast_all(series) for fc in forecasters]
            # Ensemble = nanmean across forecasters at each timestep
            stacked = np.column_stack(all_fc_forecasts)  # (T, n_fc)
            with np.errstate(all="ignore"):
                ensemble_forecast = np.nanmean(stacked, axis=1)  # (T,)
        else:
            # Fallback: rolling mean with window=50
            ensemble_forecast = np.full(T, np.nan)
            cs = np.concatenate(([0.0], np.cumsum(series)))
            _w = 50
            for _t in range(_w, T):
                ensemble_forecast[_t] = (cs[_t] - cs[_t - _w]) / _w

        all_forecasts = None  # not used per-spec
    elif forecasters is not None:
        K = len(forecasters)
        all_forecasts = [fc.forecast_all(series) for fc in forecasters]
        ensemble_forecast = None
    elif lookback_windows is not None:
        K = len(lookback_windows)
        all_forecasts = None
        ensemble_forecast = None
    else:
        raise ValueError(
            "Either lookback_windows, forecasters, or "
            "calibration_windows must be provided."
        )

    T_prime = T - min_history

    if T_prime <= 0:
        raise ValueError(
            f"Series length {T} is <= min_history {min_history}; "
            "nothing to score."
        )

    scores_matrix = np.zeros((T_prime, K))
    feature_dim = 8
    contexts = np.zeros((T_prime, feature_dim))
    targets_arr = np.zeros(T_prime)
    intervals_arr = np.zeros((T_prime, K, 2))

    # Per-spec rolling residual buffers for conformal calibration
    if calib_mode:
        residual_buffers: List[deque] = [
            deque(maxlen=cw) for cw in calibration_windows
        ]
    else:
        residual_buffers = [
            deque(maxlen=calibration_window) for _ in range(K)
        ]

    for idx in range(T_prime):
        t = min_history + idx
        y_true = series[t]
        targets_arr[idx] = y_true

        # --- build context features from history ---
        ctx = np.zeros(feature_dim)
        ctx[0] = 1.0  # bias

        short_win = series[max(0, t - 5):t]
        if len(short_win) > 0:
            ctx[1] = np.mean(short_win)
            ctx[2] = np.std(short_win) + 1e-8
        med_win = series[max(0, t - 20):t]
        if len(med_win) > 0:
            ctx[3] = np.mean(med_win)
            ctx[4] = np.std(med_win) + 1e-8
        long_win = series[max(0, t - 50):t]
        if len(long_win) > 0:
            ctx[5] = np.mean(long_win)
            ctx[6] = np.std(long_win) + 1e-8
        ctx[7] = series[t - 1]

        contexts[idx] = ctx

        # --- score each spec ---
        for k in range(K):
            # Get point forecast
            if calib_mode:
                mu_val = ensemble_forecast[t]
                if np.isnan(mu_val):
                    mu_val = series[t - 1]
                mu = float(mu_val)
            elif all_forecasts is not None:
                mu_val = all_forecasts[k][t]
                if np.isnan(mu_val):
                    # Forecaster can't produce a forecast yet; use last value
                    mu_val = series[t - 1]
                mu = float(mu_val)
            else:
                w = lookback_windows[k]
                window = series[max(0, t - w):t]
                mu = float(np.mean(window))

            # Conformal quantile from rolling residuals
            buf = residual_buffers[k]
            if len(buf) >= 2:
                residuals = np.array(buf)
                # Finite-sample correction: ceil((n+1)*(1-alpha)) / n
                n = len(residuals)
                q_idx = int(np.ceil((n + 1) * (1 - alpha)))
                q_idx = min(q_idx, n) - 1  # 0-indexed
                q = float(np.sort(np.abs(residuals))[q_idx])
            else:
                # Fallback: use std-based interval until calibration
                # buffer fills up
                if calib_mode or all_forecasts is not None:
                    # Use recent residuals from the series itself
                    recent = series[max(0, t - 50):t]
                    sigma = float(np.std(recent)) + 1e-8
                else:
                    w = lookback_windows[k]
                    window = series[max(0, t - w):t]
                    sigma = float(np.std(window)) + 1e-8
                from scipy.stats import norm as _norm
                q = _norm.ppf(1 - alpha / 2) * sigma

            lower_k = mu - q
            upper_k = mu + q

            intervals_arr[idx, k, 0] = lower_k
            intervals_arr[idx, k, 1] = upper_k

            scores_matrix[idx, k] = interval_score(
                np.array([lower_k]),
                np.array([upper_k]),
                np.array([y_true]),
                alpha=alpha,
            )[0]

            # Update residual buffer AFTER scoring (split-conformal:
            # calibrate on past, predict on present).
            buf.append(y_true - mu)

    # Build point_forecasts output
    if calib_mode:
        point_forecasts_out = ensemble_forecast[min_history:]
    elif all_forecasts is not None:
        # Forecaster mode: nanmean across forecasters
        stacked_fc = np.column_stack([fc[min_history:] for fc in all_forecasts])
        with np.errstate(all="ignore"):
            point_forecasts_out = np.nanmean(stacked_fc, axis=1)
    else:
        point_forecasts_out = None

    return scores_matrix, contexts, targets_arr, intervals_arr, point_forecasts_out


# ---------------------------------------------------------------------------
# Convenience: generate a regime-switching time series
# ---------------------------------------------------------------------------

def generate_regime_switching_series(
    n_steps: int,
    regime_persistence: float = 0.95,
    num_regimes: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a univariate time series with regime-switching dynamics.

    Each regime has its own mean, volatility, and AR(1) coefficient, so
    different lookback windows will naturally excel in different regimes.

    Parameters
    ----------
    n_steps : int
        Length of the output series.
    regime_persistence : float
        Probability of staying in the current regime at each step.
    num_regimes : int
        Number of distinct regimes.
    seed : int
        Random seed.

    Returns
    -------
    series : ndarray, shape (n_steps,)
    """
    rng = np.random.default_rng(seed)

    # Regime parameters -- deliberately varied so that different
    # lookback windows are better in different regimes.
    regime_means = rng.normal(0, 1.0, size=num_regimes)
    regime_vols = 0.3 + rng.uniform(0, 1.5, size=num_regimes)
    regime_ar = rng.uniform(-0.5, 0.8, size=num_regimes)

    series = np.zeros(n_steps)
    current_regime = rng.integers(0, num_regimes)

    for t in range(n_steps):
        mean = regime_means[current_regime]
        vol = regime_vols[current_regime]
        ar = regime_ar[current_regime]

        ar_contrib = ar * series[t - 1] if t > 0 else 0.0
        series[t] = mean + ar_contrib + rng.normal(0, vol)

        # Regime transition
        if rng.random() > regime_persistence:
            new_regime = rng.integers(0, num_regimes)
            while new_regime == current_regime and num_regimes > 1:
                new_regime = rng.integers(0, num_regimes)
            current_regime = new_regime

    return series


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== bandit_experiment self-test ===\n")

    # 1. Generate a regime-switching time series
    series = generate_regime_switching_series(
        n_steps=600, regime_persistence=0.90, seed=123,
    )
    print(f"Series length: {len(series)}, mean={series.mean():.3f}, std={series.std():.3f}")

    # 2. Build scores matrix with 4 lookback windows (basic)
    lookbacks = [10, 25, 50, 100]
    scores_matrix, contexts = build_scores_matrix_from_series(
        series, lookback_windows=lookbacks, alpha=0.10, min_history=50,
    )
    print(f"Scores matrix shape: {scores_matrix.shape}")
    print(f"Contexts shape:      {contexts.shape}")
    print(f"Mean scores per spec: {scores_matrix.mean(axis=0).round(3)}")

    # 2b. Also test return_intervals=True
    sm2, ctx2, tgt2, intv2 = build_scores_matrix_from_series(
        series, lookback_windows=lookbacks, alpha=0.10, min_history=50,
        return_intervals=True,
    )
    print(f"With return_intervals: targets shape={tgt2.shape}, "
          f"intervals shape={intv2.shape}")

    # 2c. Test CQR-based builder (now returns 5-tuple)
    sm_cqr, ctx_cqr, tgt_cqr, intv_cqr, pf_cqr = build_scores_matrix_with_cqr(
        series, lookback_windows=lookbacks, alpha=0.10, min_history=100,
        calibration_window=50,
    )
    print(f"CQR scores matrix shape: {sm_cqr.shape}")
    print(f"CQR mean scores per spec: {sm_cqr.mean(axis=0).round(3)}")
    print(f"CQR point_forecasts: {pf_cqr}")  # None for lookback mode

    # 3. Run bandit experiment (with coverage tracking)
    T, K = sm2.shape
    D = ctx2.shape[1]

    config = BanditExperimentConfig(
        num_specs=K,
        feature_dim=D,
        warmup_rounds=5,
        seed=123,
        train_steps=50,
    )
    result = run_bandit_experiment(
        sm2, ctx2, config,
        targets=tgt2,
        intervals_matrix=intv2,
    )

    summary = result.summary()
    print("\n--- Summary ---")
    for key, val in summary.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")

    print(f"\n  CTS improvement over best fixed: "
          f"{result.improvement_over_best_fixed():+.2f}%")

    if result.cts_covered is not None:
        print(f"  CTS coverage: {result.cts_covered.mean():.2%}")
    print(f"  ACI final alpha: {result.aci_alpha_history[-1]:.4f}")
    print(f"  Result arrays length (after train split): {len(result.cts_scores)}")

    # Verify new baselines
    print("\n--- New Baselines ---")
    if result.agaci_scores is not None:
        print(f"  AgACI mean score: {np.mean(result.agaci_scores):.4f}")
    if result.agaci_covered is not None:
        print(f"  AgACI coverage:   {np.mean(result.agaci_covered):.2%}")
    if result.faci_scores is not None:
        print(f"  FACI mean score:  {np.mean(result.faci_scores):.4f}")
    if result.faci_covered is not None:
        print(f"  FACI coverage:    {np.mean(result.faci_covered):.2%}")
    if result.cts_cg_scores is not None:
        print(f"  CTS-CG mean score: {np.mean(result.cts_cg_scores):.4f}")
    if result.cts_cg_covered is not None:
        print(f"  CTS-CG coverage:   {np.mean(result.cts_cg_covered):.2%}")
    if result.aci_covered is not None:
        print(f"  ACI coverage:      {np.mean(result.aci_covered):.2%}")
    if result.aci_widths is not None:
        print(f"  ACI mean width:    {np.mean(result.aci_widths):.4f}")
    if result.cts_widths is not None:
        print(f"  CTS mean width:    {np.mean(result.cts_widths):.4f}")

    print("\nSelf-test passed.")
