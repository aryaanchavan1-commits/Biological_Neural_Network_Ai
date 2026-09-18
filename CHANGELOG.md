# Changelog

## [0.2.0] - 2026-09-19

### Added
- **Large-scale neuron models**: AdEx, dual-compartment LIF, resonate-and-fire, spiking brain-inspired neurons
- **Hybrid attention mechanisms**: Dendritic Self-Spine Attention (DSSA), spiking attention, bio-attention with neurotransmitter modulation
- **Memory routing**: Gated routing for attention with working memory integration
- **Scaling infrastructure**: Distributed training, mixed precision, gradient checkpointing, resource profiling
- **Neuromorphic deployment**: ONNX export, quantization (per-tensor INT8), hardware benchmarking for Loihi/TrueNorth
- **Emergence detection**: Criticality analysis (branching ratio, avalanche statistics), Lyapunov exponents, entropy metrics, functional zone mapping
- **Safety & alignment**: Interpretability (feature importance, attention visualization), controllability, anomaly/drift detection
- **New configs**:
  - `configs/superintelligence_criticality.yaml` — Critical dynamics and phase transition study
  - `configs/superintelligence_neuromorphic.yaml` — Neuromorphic deployment benchmarks
  - `configs/large_scale_neurons.yaml` — Multi-model neuron comparison
  - `configs/hybrid_attention.yaml` — Hybrid attention with DSSA and spiking attention
- **New dependencies**: scipy, scikit-learn, networkx
- **Updated README**: Positioned as 2026 biological superintelligence research framework

### Changed
- Version bumped from 0.1.0 to 0.2.0
- README rewritten for superintelligence research positioning
- setup.py updated with new dependencies and modules

## [0.1.0] - 2026-09-19

### Added
- Initial project structure
- Literature review
- Architecture specification
- Research hypotheses
- Experiment plan
- Configuration system
- Baseline configurations

### Planned
- LIF neuron implementation
- Basic SNN
- STDP plasticity
- Structural plasticity
- Continual learning
- Visualization dashboard
