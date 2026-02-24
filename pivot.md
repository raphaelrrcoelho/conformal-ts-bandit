# Conformal Thompson Sampling (CTS) — Pivoted Research Codebase

## Research Pivot

This codebase implements the pivoted research strategy for the IMPA master's thesis:

**From**: CTS on M5 retail data (where specification selection doesn't matter much)
**To**: Three-pronged contribution:

1. **Theoretical**: Regret bounds for Linear TS with conformal rewards
2. **Diagnostic**: "When does adaptive specification selection help?" — multi-dataset analysis correlating non-stationarity with CTS improvement
3. **GEFCom2014 Benchmark**: Primary empirical demonstration on energy data where regimes genuinely shift

## Why This Pivot

- M5 retail data has stable seasonality → LightGBM extracts similar info regardless of lookback → specification selection ≈ irrelevant
- GEFCom2014 wind/solar have genuine regime shifts (weather fronts, cloud events) → different lookbacks are optimal at different times
- The "diagnostic" framing is a more honest and interesting paper than "we beat baselines by 3%"

## Architecture

```
cts_pivot/
├── core/                          # Dataset-agnostic core library
│   ├── linear_ts.py               # Linear Thompson Sampling (Sherman-Morrison O(d²))
│   ├── cqr.py                     # Conformalized Quantile Regression (shared infrastructure)
│   ├── metrics.py                 # Interval score, pinball loss, DM test, bootstrap CI
│   ├── specifications.py          # Action space definitions
│   └── agent.py                   # CTS agent (TS + CQR combined)
├── baselines/
│   └── __init__.py                # Fixed, Ensemble, Random, Oracle, UCB, RoundRobin, ACI
├── data/
│   └── gefcom2014.py              # GEFCom2014 loader + synthetic data generator
├── experiments/
│   └── runner.py                  # Fair comparison experiment runner
├── diagnostics/
│   └── __init__.py                # Non-stationarity metrics & diagnostic reports
├── configs/
│   └── default.yaml               # Default configuration
└── scripts/
    ├── smoke_test.py              # Verify everything works (8 tests)
    ├── run_gefcom.py              # Main GEFCom2014 experiment
    └── run_diagnostic.py          # Multi-dataset diagnostic comparison
```

## Key Design Principle: Fair Comparison

**ALL methods share the IDENTICAL CQR pipeline** (same LightGBM models, same conformal calibrators). They differ ONLY in specification selection logic. This follows the gold standard from Gibbs & Candès where fair comparisons isolate exactly the mechanism being evaluated.

```
Shared Infrastructure (identical for all methods):
  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
  │  LightGBM   │───►│   Conformal      │───►│   Calibrated    │
  │  Quantile   │    │   Calibrator     │    │   Intervals     │
  │  Models     │    │   (rolling 250)  │    │   [l_t, u_t]    │
  └─────────────┘    └──────────────────┘    └─────────────────┘

Selection Logic (ONLY difference between methods):
  CTS:        Thompson Sampling → spec_idx
  Fixed:      Always same spec → spec_idx  
  UCB:        Upper Confidence Bound → spec_idx
  Random:     Uniform random → spec_idx
  Ensemble:   Average across specs → blended interval
  ACI:        Fixed spec + adaptive calibration alpha
```

## Running

```bash
# Smoke test (verify everything works)
python scripts/smoke_test.py

# Main GEFCom2014 experiment (solar track)
python scripts/run_gefcom.py --track solar

# Wind track (stronger regime effects)
python scripts/run_gefcom.py --track wind

# Full diagnostic comparison
python scripts/run_diagnostic.py
```

## Metrics

| Metric | Use | Reference |
|--------|-----|-----------|
| Interval Score | Bandit reward + primary metric | Gneiting & Raftery (2007) |
| Pinball Loss | GEFCom2014 evaluation | Hong et al. (2016) |
| Coverage Rate | Conformal guarantee verification | Romano et al. (2019) |
| Diebold-Mariano | Statistical significance | Diebold & Mariano (1995) |
| Bootstrap CI | Confidence intervals | Moving-block bootstrap |

## Non-Stationarity Diagnostic

The diagnostic module computes a composite non-stationarity index:

- **Switch frequency**: How often does the optimal specification change?
- **Selection entropy**: How spread out are optimal spec selections?
- **Performance spread**: How different are specs from each other?
- **Changepoints**: How many regime shifts exist?

High index → CTS should significantly outperform fixed baselines
Low index → Fixed baselines are sufficient

## Next Steps

1. **Get real GEFCom2014 data** — replace synthetic generators with actual competition data
2. **Theory**: Derive regret bounds for Linear TS with conformal rewards (connects TS regret to conformal coverage)
3. **Add more datasets** for the diagnostic sweep (ENTSO-E prices, Monash repository)
4. **Plotting**: Add visualization module for regime analysis, selection heatmaps, cumulative regret curves
