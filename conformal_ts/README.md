# Conformal Thompson Sampling for Adaptive Specification Selection

Implementation of Conformal Thompson Sampling (CTS) for adaptive specification selection in time series forecasting. This combines Linear Thompson Sampling with Conformalized Quantile Regression to dynamically choose optimal forecast specifications.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from conformal_ts.models.cts_agent import ConformalThompsonSampling, CTSConfig

# Create agent
config = CTSConfig(
    num_actions=16,       # Number of specifications
    feature_dim=10,       # Context dimension
    coverage_target=0.90, # Target coverage
    warmup_rounds=50,     # Exploration phase
    seed=42
)
agent = ConformalThompsonSampling(config)

# Training loop
for t in range(1000):
    context = get_features()  # Your feature extraction
    
    # Agent selects specification and predicts interval
    action, lower, upper = agent.select_and_predict(context)
    
    # Use specification 'action' to generate forecast
    forecast = generate_forecast(action)
    
    # Observe outcome
    outcome = observe_outcome()
    
    # Update agent
    reward = agent.update(action, context, outcome)
```

## Running Experiments

### Synthetic Data
```bash
python -m conformal_ts.experiments.run_experiment --experiment synthetic --seed 42
```

### M5 Simulation
```bash
python -m conformal_ts.experiments.run_experiment --experiment m5_sim --seed 42
```

## Project Structure

```
conformal_ts/
├── config.py                    # Configuration dataclasses
├── models/
│   ├── linear_ts.py            # Linear Thompson Sampling
│   ├── cqr.py                  # Conformalized Quantile Regression
│   └── cts_agent.py            # Main CTS agent
├── data/
│   ├── synthetic.py            # Synthetic data generator
│   └── m5_loader.py            # M5 dataset loader
├── evaluation/
│   └── metrics.py              # Interval scores, DM tests
└── experiments/
    └── run_experiment.py       # Experiment pipeline
```

## Core Components

### Linear Thompson Sampling (`models/linear_ts.py`)
- Disjoint linear bandit: `r = φᵀθₐ + ε`
- Sherman-Morrison O(d²) updates
- Variance inflation for exploration
- Numerical stability safeguards

### Conformalized Quantile Regression (`models/cqr.py`)
- Online quantile regression with pinball loss
- Rolling calibration window
- Distribution-free coverage guarantees
- Action-specific models

### Interval Scores (`evaluation/metrics.py`)
- Proper scoring rule: `IS(l,u,y) = (u-l) + (2/α)·penalties`
- Diebold-Mariano tests
- Bootstrap confidence intervals

## Configuration

```python
from conformal_ts.config import get_default_config

# Get default config for M5
config = get_default_config("m5")

# Customize
config.ts_config.exploration_variance = 3.0
config.cqr_config.calibration_window = 200
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prior_precision` | 0.1 | Regularization (λ) |
| `exploration_variance` | 5.0 | Posterior sampling variance |
| `coverage_target` | 0.90 | Target coverage probability |
| `learning_rate` | 0.02 | CQR SGD learning rate |
| `calibration_window` | 250 | Rolling calibration buffer |
| `warmup_rounds` | 50 | Initial exploration rounds |

## References

- Agrawal & Goyal (2013) - Thompson Sampling for Contextual Bandits
- Romano, Patterson, Candès (2019) - Conformalized Quantile Regression
- Gneiting & Raftery (2007) - Strictly Proper Scoring Rules

## License

MIT
