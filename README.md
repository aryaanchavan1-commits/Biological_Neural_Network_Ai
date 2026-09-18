# BIO-NN: Biological Superintelligence Research Framework

## Overview

A modular, production-grade research framework for understanding biological superintelligence mechanisms. Built for 2026, BIO-NN provides composable modules spanning large-scale neuron models, hybrid attention architectures, scaling infrastructure, neuromorphic deployment, emergence detection, and safety alignment — enabling systematic study of critical dynamics in biological neural systems.

## Key Features

- **Large-Scale Neuron Models** — Adaptive LIF, AdEx, dual-compartment, resonate-and-fire, spiking brain-inspired neurons
- **Hybrid Attention Mechanisms** — Dendritic Self-Spine Attention (DSSA), spiking attention, bio-attention, memory routing
- **Scaling Infrastructure** — Distributed training, mixed precision, gradient checkpointing, resource profiling
- **Neuromorphic Deployment** — ONNX export, quantization, hardware benchmarking for Loihi/IntelTrueNorth
- **Emergence Detection** — Criticality analysis, complexity metrics, functional zone mapping, phase transition monitoring
- **Safety & Alignment** — Interpretability, controllability, value alignment, continuous monitoring

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

### Install dependencies directly

```bash
pip install torch numpy scipy matplotlib seaborn scikit-learn networkx pyyaml pandas tensorboard psutil
```

### With dashboard (optional)

```bash
pip install -e ".[dashboard]"
```

### With development tools

```bash
pip install -e ".[dev]"
```

## Quick Start

### Run baseline SNN

```bash
python -m bio_nn.experiments.run --config configs/baseline_snn.yaml
```

### Run full BIO-NN

```bash
python -m bio_nn.experiments.run --config configs/bio_v01.yaml
```

### Study critical dynamics at the edge of chaos

```bash
python -m bio_nn.experiments.run --config configs/superintelligence_criticality.yaml
```

### Benchmark neuromorphic deployment

```bash
python -m bio_nn.experiments.run --config configs/superintelligence_neuromorphic.yaml
```

### Run large-scale neuron experiment

```bash
python -m bio_nn.experiments.run --config configs/large_scale_neurons.yaml
```

### Run hybrid attention experiment

```bash
python -m bio_nn.experiments.run --config configs/hybrid_attention.yaml
```

### Launch the dashboard

```bash
python -m bio_nn.visualization.dashboard.server
```

## Module Reference

| Module | Purpose |
|--------|---------|
| `bio_nn.neurons` | Adaptive LIF, AdEx, dual-LIF, resonate-fire, spiking brain neurons |
| `bio_nn.attention` | DSSA, spiking attention, bio-attention, memory routing |
| `bio_nn.scaling` | Distributed training, mixed precision, profiling, checkpointing |
| `bio_nn.neuromorphic` | Export, quantization, benchmarking for neuromorphic hardware |
| `bio_nn.emergence` | Criticality, complexity, functional zones, phase monitoring |
| `bio_nn.safety` | Alignment, controllability, interpretability, monitoring |
| `bio_nn.plasticity` | STDP, homeostatic plasticity, structural plasticity |
| `bio_nn.topology` | Sparse connectivity, small-world networks |
| `bio_nn.dendrites` | Branching dendritic computation |
| `bio_nn.neuromodulation` | Dopamine, serotonin-inspired modulation |

## Example Usage

### Custom neuron configuration

```python
from bio_nn.neurons.adaptive_lif import AdaptiveLIFNeuron

neuron = AdaptiveLIFNeuron(
    tau_mem=20.0,
    tau_adapt=100.0,
    threshold=1.0,
    adapt_rate=0.01,
)
```

### Dendritic self-spine attention

```python
from bio_nn.attention.dssa import DendriticSelfSpineAttention

attn = DendriticSelfSpineAttention(
    d_model=256,
    n_heads=8,
    dendritic_branches=4,
    spine_threshold=0.5,
)
```

### Criticality analysis

```python
from bio_nn.emergence.criticality import CriticalityAnalyzer

analyzer = CriticalityAnalyzer(
    spike_trains=spike_data,
    bin_size=1.0,
    max_lag=50,
)
results = analyzer.compute_branching_ratio()
is_critical = analyzer.is_near_criticality(tolerance=0.05)
```

### Neuromorphic export

```python
from bio_nn.neuromorphic.exporter import NeuromorphicExporter

exporter = NeuromorphicExporter(model)
exporter.export_onnx("model.onnx", input_shape=(1, 784))
exporter.quantize("model_quantized.onnx", bits=8)
```

## Project Structure

```
Bio_NN/
├── README.md
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
├── setup.py
├── requirements.txt
├── configs/
│   ├── baseline_snn.yaml
│   ├── bio_v01.yaml
│   ├── bio_continual.yaml
│   ├── superintelligence_criticality.yaml
│   ├── superintelligence_neuromorphic.yaml
│   ├── large_scale_neurons.yaml
│   └── hybrid_attention.yaml
├── bio_nn/
│   ├── neurons/          # LIF, AdEx, dual-LIF, resonate-fire, spiking brain
│   ├── attention/        # DSSA, spiking attention, bio-attention
│   ├── scaling/          # Distributed, mixed precision, profiling
│   ├── neuromorphic/     # Export, quantize, benchmark
│   ├── emergence/        # Criticality, complexity, functional zones
│   ├── safety/           # Alignment, controllability, interpretability
│   ├── plasticity/       # STDP, homeostatic, structural
│   ├── topology/         # Sparse, small-world connectivity
│   ├── dendrites/        # Branching computation
│   ├── neuromodulation/  # Dopamine/serotonin modulation
│   ├── encoders/         # Rate, temporal encoding
│   ├── decoders/         # Spike decoding
│   ├── memory/           # Working memory, memory routing
│   ├── prediction/       # Predictive coding
│   ├── training/         # Training loops
│   ├── learning/         # Surrogate gradients, learning rules
│   ├── core/             # Base classes
│   ├── config/           # Configuration system
│   ├── evaluation/       # Metrics and evaluation
│   ├── experiments/      # Experiment runners
│   ├── visualization/    # Dashboard, plotting
│   └── utils/            # Seeding, helpers
├── tests/
├── docs/
├── research/
├── experiments/
├── scripts/
├── data/
├── checkpoints/
└── logs/
```

## Research Directions

1. **Critical Dynamics** — Mapping phase transitions between ordered and chaotic regimes in biological-scale networks
2. **Emergent Computation** — How functional zones and memory routing arise from local plasticity rules
3. **Neuromorphic Scaling** — Efficient deployment of biological models on brain-inspired hardware
4. **Safety Alignment** — Ensuring biological superintelligence mechanisms remain interpretable and controllable
5. **Hybrid Architectures** — Combining dendritic attention with spiking dynamics for scalable computation

## Citation

```bibtex
@software{bio_nn2026,
  title={BIO-NN: Biological Superintelligence Research Framework},
  year={2026},
  version={0.2.0},
  url={https://github.com/your-repo/bio-nn}
}
```

## License

Research use. See [LICENSE](LICENSE) file.
