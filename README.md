# Conformal Thompson Sampling for Adaptive Specification Selection

Implementation of Conformal Thompson Sampling (CTS) for adaptive specification selection in time series forecasting. This combines Linear Thompson Sampling with Conformalized Quantile Regression to dynamically choose optimal forecast specifications, with proper evaluation on benchmark datasets.


## Dataset Loaders
- **`conformal_ts/data/m5_real.py`** - Full M5 competition data loader
  - Downloads from Kaggle API
  - Handles 12-level hierarchy aggregation
  - Computes proper scaling factors for WSPL
  - Creates train/val/test splits matching competition

- **`conformal_ts/data/gefcom2014.py`** - GEFCom2014 energy forecasting
  - Solar and wind track support
  - Weather feature integration
  - Hourly probabilistic forecasting setup

## Proper Baselines
- **`conformal_ts/baselines/baselines.py`** - Competition-grade baselines
  - `LightGBMQuantileBaseline` - Top M5 method
  - `MultiQuantileLightGBM` - Full quantile distribution
  - `FixedSpecificationBaseline` - Best fixed spec
  - `EqualWeightEnsemble` - Ensemble averaging
  - `AdaptiveConformalInference` - ACI (Gibbs & Candès 2021)
  - `OracleBaseline` - Hindsight optimal (upper bound)
  - `RandomSpecificationBaseline` - Random selection

## Competition Metrics
- **`conformal_ts/evaluation/competition_metrics.py`**
  - WSPL (Weighted Scaled Pinball Loss) - M5 metric
  - Pinball loss across 99 quantiles - GEFCom metric
  - Interval score / Winkler score
  - CRPS approximation
  - Diebold-Mariano tests with HAC variance
  - Bootstrap confidence intervals

## Full Experiment Runner
- **`conformal_ts/experiments/run_full_experiment.py`**
  - Proper train/val/test splits
  - Deferred updates (respecting forecast horizons)
  - Checkpointing for long runs
  - Comprehensive logging
  - Result saving in JSON format

# Quick Start

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Download Data

```bash
# For M5 (requires Kaggle API credentials)
python setup_data.py --dataset m5

# For GEFCom (manual download required, see instructions)
python setup_data.py --dataset gefcom

# Run quick sanity check
python setup_data.py --test
```

## 3. Run Experiments

```bash
# Quick test on simulated data (~2 minutes)
python -m conformal_ts.experiments.run_full_experiment --quick --output ./results/quick_test

# Full M5 experiment (~1-2 hours with sampling)
python -m conformal_ts.experiments.run_full_experiment \
    --dataset m5 \
    --max-series 500 \
    --output ./results/m5

# Full scale M5 (several hours)
python -m conformal_ts.experiments.run_full_experiment \
    --dataset m5 \
    --max-series 5000 \
    --output ./results/m5_full

# GEFCom solar track
python -m conformal_ts.experiments.run_full_experiment \
    --dataset gefcom \
    --track solar \
    --output ./results/gefcom_solar
```

# Expected Output

After running, you'll get:

```
results/
├── results_20260109_120000.json    # Full results
├── results_latest.json             # Symlink to latest
└── agent_20260109_120000.json      # Saved agent state
```

The results JSON contains:

```json
{
  "config": {...},
  "training": {
    "total_steps": 150000,
    "training_time": 3600.5,
    "final_mean_score": 45.23,
    "final_coverage": 0.892,
    "action_distribution": [1234, 2345, ...]
  },
  "evaluation": {
    "cts": {
      "mean_score": 42.15,
      "score_ci_lower": 40.2,
      "score_ci_upper": 44.1,
      "coverage": 0.901,
      "wspl": 0.0234
    },
    "lightgbm": {...},
    "fixed_best": {...},
    "dm_tests": {
      "lightgbm": {
        "pct_improvement": 7.3,
        "p_value": 0.023,
        "significant_0.05": true
      }
    }
  }
}
```

# File Structure

```
conformal_ts/
├── models/
│   ├── linear_ts.py           # Linear Thompson Sampling
│   ├── cqr.py                 # Conformalized Quantile Regression
│   └── cts_agent.py           # Main CTS agent
├── data/
│   ├── synthetic.py           # Synthetic data generator
│   ├── m5_real.py             # M5 data loader
│   └── gefcom2014.py          # GEFCom2014 loader
├── baselines/
│   ├── __init__.py            # Package init
│   └── baselines.py           # All baseline methods
├── evaluation/
│   ├── metrics.py             # Original metrics
│   └── competition_metrics.py # Competition-grade metrics
└── experiments/
    ├── run_experiment.py      # Original experiment runner
    └── run_full_experiment.py # Full-scale runner
```

## For the Paper

When reporting results, include:

1. **Table 1**: Main comparison
   - Mean interval score with 95% CI
   - Coverage rate
   - WSPL (for M5)

2. **Table 2**: Statistical tests
   - DM test p-values vs each baseline
   - Improvement percentages

3. **Figure 1**: Learning curve
   - Interval score over training
   - Convergence behavior

4. **Figure 2**: Action distribution
   - Which specifications selected over time
   - Context-dependent patterns

## Troubleshooting

### Kaggle API Issues
```bash
# Install kaggle
pip install kaggle

# Set up credentials (download from kaggle.com/account)
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Memory Issues with Full M5
```bash
# Use sampling
python -m conformal_ts.experiments.run_full_experiment \
    --dataset m5 \
    --max-series 1000  # Reduce from 30K
```

### Missing LightGBM
```bash
pip install lightgbm

# Or run without it
python -m conformal_ts.experiments.run_full_experiment --no-lightgbm
```

## Citation

If you use this code, please cite:

```bibtex
@mastersthesis{coelho2025conformal,
  title={Causal Reinforcement Learning for Dynamic Factor Momentum Specification in Brazilian Debenture Markets},
  author={Coelho, Raphael R.},
  school={IMPA},
  year={2025}
}
```