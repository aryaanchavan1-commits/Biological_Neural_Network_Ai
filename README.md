# BIO-NN: Biologically Inspired Neural Network Research Framework

## Overview

A modular, configurable research framework for experimenting with biologically-inspired neural network mechanisms. Tests whether biological mechanisms (synaptic plasticity, structural plasticity, sparse computation, neuromodulation) can improve continual learning, robustness, and efficiency.

## Key Features

- **Modular architecture** with replaceable components
- **YAML-based configuration** — no code changes needed
- **13+ neuron/plasticity/structure mechanisms** from computational neuroscience
- **Continual learning benchmarks** for stability-plasticity evaluation
- **Interactive visualization dashboard** for real-time monitoring
- **Full experiment tracking** and reproducibility
- **Ablation study support** for systematic mechanism evaluation

## Installation

### Prerequisites

- Python 3.9+
- pip 21.0+
- CUDA-capable GPU (optional, for accelerated training)

### Install from source

```bash
git clone https://github.com/your-repo/bio-nn.git
cd bio-nn
pip install -e .
```

### With dashboard (optional)

```bash
pip install -e ".[dashboard]"
```

### With development tools

```bash
pip install -e ".[dev]"
```

See [docs/installation.md](docs/installation.md) for detailed instructions including GPU setup and troubleshooting.

## Quick Start

### Run a baseline SNN

```bash
python -m bio_nn.experiments.run --config configs/baseline_snn.yaml
```

### Run BIO-NN (full biological mechanisms)

```bash
python -m bio_nn.experiments.run --config configs/bio_v01.yaml
```

### Run continual learning experiment

```bash
python -m bio_nn.experiments.run --config configs/bio_continual.yaml
```

### Launch the dashboard

```bash
python -m bio_nn.visualization.dashboard.server
```

## Project Structure

```
Bio_NN/
├── README.md
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
├── pyproject.toml
├── configs/
│   ├── baseline_snn.yaml
│   ├── bio_v01.yaml
│   └── bio_continual.yaml
├── bio_nn/
│   ├── __init__.py
│   ├── neurons/
│   │   ├── __init__.py
│   │   ├── lif.py
│   │   └── izhikevich.py
│   ├── plasticity/
│   │   ├── __init__.py
│   │   ├── stdp.py
│   │   └── homeostatic.py
│   ├── structure/
│   │   ├── __init__.py
│   │   └── topology.py
│   ├── encoding/
│   │   ├── __init__.py
│   │   └── rate.py
│   ├── networks/
│   │   ├── __init__.py
│   │   └── snn.py
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── run.py
│   │   └── continual.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── accuracy.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── dashboard/
│   │       ├── __init__.py
│   │       └── server.py
│   └── utils/
│       ├── __init__.py
│       └── seeding.py
├── tests/
│   ├── __init__.py
│   ├── test_neurons.py
│   ├── test_plasticity.py
│   └── test_networks.py
├── docs/
│   ├── installation.md
│   ├── architecture.md
│   ├── extending_bio_nn.md
│   └── reproducibility.md
└── research/
    ├── literature_review.md
    ├── architecture_spec.md
    ├── research_hypotheses.md
    └── experiment_plan.md
```

## Configuration

BIO-NN uses YAML-based configuration. All experiments are defined declaratively — no code changes needed.

### Example: Baseline SNN

```yaml
# configs/baseline_snn.yaml
network:
  neuron_model: lif
  hidden_sizes: [128, 128]
  timestep: 1.0

training:
  optimizer: adam
  learning_rate: 0.001
  epochs: 50
  batch_size: 32

dataset:
  name: mnist
  data_dir: ./data

plasticity: {}  # no plasticity for baseline

structure: {}  # fixed topology
```

### Example: Full BIO-NN

```yaml
# configs/bio_v01.yaml
network:
  neuron_model: lif
  hidden_sizes: [128, 128]
  timestep: 1.0

training:
  optimizer: adam
  learning_rate: 0.001
  epochs: 50
  batch_size: 32

dataset:
  name: mnist
  data_dir: ./data

plasticity:
  stdp:
    enabled: true
    tau_plus: 20.0
    tau_minus: 20.0
    a_plus: 0.01
    a_minus: 0.012
  homeostatic:
    enabled: true
    target_rate: 0.05
    adaptation_rate: 0.01

structure:
  small_world:
    enabled: true
    rewire_prob: 0.1
```

## Experiments

### Running a single experiment

```bash
python -m bio_nn.experiments.run --config configs/bio_v01.yaml
```

### Running an ablation study

```bash
python -m bio_nn.experiments.run --config configs/ablation_stdp.yaml
```

### Running continual learning benchmarks

```bash
python -m bio_nn.experiments.run --config configs/bio_continual.yaml
```

### Results are saved to

```
results/
├── {experiment_name}/
│   ├── metrics.csv
│   ├── config.yaml
│   ├── checkpoints/
│   └── plots/
```

## Documentation

- [Installation Guide](docs/installation.md)
- [Architecture Overview](docs/architecture.md)
- [Extending BIO-NN](docs/extending_bio_nn.md)
- [Reproducibility Guide](docs/reproducibility.md)

## Research

- [Literature Review](research/literature_review.md)
- [Architecture Specification](research/architecture_spec.md)
- [Research Hypotheses](research/research_hypotheses.md)
- [Experiment Plan](research/experiment_plan.md)

## Citation

```bibtex
@software{bio_nn2026,
  title={BIO-NN: Biologically Inspired Neural Network Research Framework},
  year={2026},
  version={0.1.0},
  url={https://github.com/your-repo/bio-nn}
}
```

## License

Research use. See [LICENSE](LICENSE) file.
