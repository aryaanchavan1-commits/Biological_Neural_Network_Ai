# Reproducibility Guide

BIO-NN is designed for full reproducibility. Every experiment captures all state needed to reproduce results exactly.

## Setting Random Seeds

All random seeds are set automatically via `bio_nn.utils.seeding`:

```python
from bio_nn.utils.seeding import set_seed

set_seed(42)  # Sets torch, numpy, and random seeds
```

When running via config, seeds are set from the YAML:

```yaml
reproducibility:
  seed: 42
  deterministic: true  # torch.use_deterministic_algorithms(True)
```

## Locking Dependencies

### Export current environment

```bash
pip freeze > requirements.lock
```

### Install from lock file

```bash
pip install -r requirements.lock
```

### Using pyproject.toml

Core dependencies are pinned in `pyproject.toml`. For exact reproduction, install from the lock file.

## Running Experiments

### 1. Capture the configuration

The experiment runner automatically saves the full config to the results directory:

```
results/{experiment_name}/config.yaml
```

### 2. Run the experiment

```bash
python -m bio_nn.experiments.run --config configs/bio_v01.yaml
```

### 3. Check results

Results are saved to:

```
results/{experiment_name}/
├── config.yaml          # Exact config used
├── metrics.csv          # Per-epoch metrics
├── checkpoints/         # Model weights
├── logs/                # Experiment logs
└── plots/               # Generated visualizations
```

## Checking Results

### Load saved metrics

```python
import pandas as pd

metrics = pd.read_csv('results/experiment_name/metrics.csv')
print(metrics.head())
```

### Load saved model

```python
import torch

checkpoint = torch.load('results/experiment_name/checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Compare experiments

```bash
python -m bio_nn.experiments.compare \
  --experiments results/baseline results/bio_v01 \
  --metric accuracy
```

## Docker Reproducibility (Optional)

For full environment reproducibility:

```dockerfile
FROM python:3.10-slim
COPY requirements.lock .
RUN pip install -r requirements.lock
COPY . .
RUN pip install -e .
```

```bash
docker build -t bio-nn .
docker run bio-nn python -m bio_nn.experiments.run --config configs/bio_v01.yaml
```
