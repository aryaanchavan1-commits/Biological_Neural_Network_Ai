# BIO-NN Architecture Specification

**Version:** 1.0.0
**Status:** Draft
**Last Updated:** 2026-09-19

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Module Hierarchy](#2-module-hierarchy)
3. [Abstract Base Classes](#3-abstract-base-classes)
4. [Data Flow](#4-data-flow)
5. [Configuration System](#5-configuration-system)
6. [Biological Plausibility Levels](#6-biological-plausibility-levels)
7. [Experiment Lifecycle](#7-experiment-lifecycle)
8. [Registry and Dynamic Loading](#8-registry-and-dynamic-loading)
9. [Inspection and Observability](#9-inspection-and-observability)
10. [Reproducibility Contract](#10-reproducibility-contract)

---

## 1. Design Principles

### 1.1 Modularity

Every component in BIO-NN is defined by an abstract base class. Concrete implementations are registered and loaded dynamically. No component depends on any other concrete component — only on abstract interfaces. Replacing the neuron model from LIF to Izhikevich requires zero changes outside the neuron module.

### 1.2 Configurability

All experiments are fully described by a YAML configuration file. The config specifies which concrete class to instantiate for each module slot, along with all hyperparameters. No code changes are required to run a new experiment variant. The config schema is validated at load time against a JSON Schema definition.

### 1.3 Reproducibility

Every experiment result is uniquely identified by the tuple `(CODE_COMMIT, CONFIG_HASH, DATASET_HASH, SEED)`. The framework auto-records this tuple, along with hardware info and wall-clock timing, alongside all metrics. Results are never written without this metadata.

### 1.4 Inspectability

Every module exposes inspection hooks. Any tensor — membrane potentials, spike trains, weights, gradients, prediction errors — can be read at any timestep via the module API. A middleware layer captures these snapshots without modifying the computation graph.

### 1.5 Extensibility

Adding a new component means: (1) subclass the relevant abstract base, (2) decorate with `@register("category", "name")`, (3) add a YAML entry. No existing code is modified.

---

## 2. Module Hierarchy

```
bio_nn/
├── __init__.py
├── core/
│   ├── base.py              # Abstract base classes for all components
│   ├── registry.py          # Component registry for dynamic loading
│   └── tensor_ops.py        # Shared tensor operations
├── neurons/
│   ├── base.py              # Abstract NeuronModel
│   ├── lif.py               # Leaky Integrate-and-Fire
│   ├── adaptive_lif.py      # Adaptive LIF
│   ├── izhikevich.py        # Izhikevich model
│   └── config.py            # Neuron configurations
├── synapses/
│   ├── base.py              # Abstract SynapseModel
│   ├── static.py            # Static synapses
│   └── dynamic.py           # Dynamic synapses
├── plasticity/
│   ├── base.py              # Abstract PlasticityRule
│   ├── hebbian.py           # Hebbian learning
│   ├── stdp.py              # STDP
│   ├── reward_stdp.py       # Reward-modulated STDP
│   ├── homeostatic.py       # Homeostatic plasticity
│   └── metaplasticity.py    # Metaplasticity
├── dendrites/
│   ├── base.py              # Abstract DendriticModule
│   ├── compartment.py       # Compartment-like representations
│   └── nonlinear.py         # Dendritic nonlinearities
├── structural/
│   ├── base.py              # Abstract StructuralPlasticity
│   ├── growth.py            # Connection/neuron growth
│   ├── pruning.py           # Connection/neuron pruning
│   └── adaptive_topology.py # Adaptive network topology
├── memory/
│   ├── base.py              # Abstract MemoryModule
│   ├── recurrent.py         # Recurrent state
│   ├── working_memory.py    # Working memory
│   ├── associative.py       # Associative memory
│   └── synaptic.py          # Synaptic memory
├── prediction/
│   ├── base.py              # Abstract PredictionModule
│   ├── predictor.py         # Prediction mechanism
│   └── error.py             # Prediction error computation
├── neuromodulation/
│   ├── base.py              # Abstract Neuromodulator
│   ├── reward.py            # Reward signals
│   └── global.py            # Global modulatory signals
├── encoders/
│   ├── base.py              # Abstract Encoder
│   ├── rate.py              # Rate coding
│   ├── temporal.py          # Temporal coding
│   ├── latency.py           # Latency coding
│   └── population.py        # Population coding
├── decoders/
│   ├── base.py              # Abstract Decoder
│   ├── rate_decoder.py      # Rate-based decoding
│   └── spike_decoder.py     # Spike-based decoding
├── topology/
│   ├── base.py              # Abstract TopologyRule
│   ├── random.py            # Random connectivity
│   ├── small_world.py       # Small-world topology
│   └── scale_free.py        # Scale-free topology
├── learning/
│   ├── base.py              # Abstract LearningRule
│   ├── backprop.py          # Backpropagation (surrogate gradient)
│   ├── local.py             # Local learning rules
│   └── hybrid.py            # Hybrid local/global
├── training/
│   ├── engine.py            # Training loop
│   ├── continual.py         # Continual learning trainer
│   └── profiler.py          # Performance profiling
├── evaluation/
│   ├── metrics.py           # All metrics
│   ├── classification.py    # Classification metrics
│   ├── continual.py         # Continual learning metrics
│   ├── efficiency.py        # Efficiency metrics
│   ├── robustness.py        # Robustness metrics
│   └── calibration.py       # Calibration metrics
├── visualization/
│   ├── static/              # Static plots
│   │   ├── topology.py
│   │   ├── spike_raster.py
│   │   ├── membrane.py
│   │   ├── weights.py
│   │   ├── firing_rates.py
│   │   ├── sparsity.py
│   │   ├── plasticity.py
│   │   ├── structure.py
│   │   ├── continual.py
│   │   └── comparison.py
│   └── dashboard/           # Interactive web dashboard
│       ├── server.py        # FastAPI backend
│       ├── static/          # HTML/CSS/JS
│       └── templates/
├── experiments/
│   ├── manager.py           # Experiment lifecycle
│   ├── tracker.py           # Metric tracking
│   ├── checkpoint.py        # Checkpoint management
│   └── comparison.py        # Experiment comparison
├── config/
│   ├── loader.py            # YAML config loading
│   ├── schema.py            # Config validation
│   └── defaults.py          # Default configurations
└── utils/
    ├── seeds.py             # Reproducibility
    ├── timing.py            # Timing utilities
    ├── memory.py            # Memory tracking
    ├── hardware.py          # Hardware detection
    └── io.py                # File I/O utilities
```

### 2.1 Module Categories

| Category | Slot Name | Base Class | Purpose |
|----------|-----------|------------|---------|
| neurons | `neuron_model` | `NeuronModel` | Membrane dynamics and spike generation |
| synapses | `synapse_model` | `SynapseModel` | Signal transmission between neurons |
| plasticity | `plasticity_rule` | `PlasticityRule` | Synaptic weight modification |
| dendrites | `dendritic_module` | `DendriticModule` | Dendritic computation |
| structural | `structural_plasticity` | `StructuralPlasticity` | Network topology changes |
| memory | `memory_module` | `MemoryModule` | State persistence and retrieval |
| prediction | `prediction_module` | `PredictionModule` | Predictive coding |
| neuromodulation | `neuromodulator` | `Neuromodulator` | Global and local modulation |
| encoders | `encoder` | `Encoder` | Data to spike conversion |
| decoders | `decoder` | `Decoder` | Spike to output conversion |
| topology | `topology_rule` | `TopologyRule` | Initial connectivity pattern |
| learning | `learning_rule` | `LearningRule` | Gradient computation and weight updates |

---

## 3. Abstract Base Classes

### 3.1 NeuronModel

```python
class NeuronModel(ABC):
    """Base class for all neuron models.

    State is an opaque tensor dict. Each concrete model defines
    its own state schema (e.g., membrane potential, threshold, adaptation).
    """

    @abstractmethod
    def initialize_state(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        """Create initial state tensors for a batch.

        Returns:
            dict mapping state name -> tensor of shape (batch_size, n_neurons)
        """
        ...

    @abstractmethod
    def forward(
        self,
        state: dict[str, torch.Tensor],
        input_current: torch.Tensor,
        dt: float,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Single timestep update.

        Args:
            state: Current neuron state dict.
            input_current: Shape (batch_size, n_neurons). Total input current.
            dt: Simulation timestep in ms.

        Returns:
            (new_state, spikes) where spikes is a boolean tensor
            of shape (batch_size, n_neurons).
        """
        ...

    def get_config(self) -> dict:
        """Return constructor arguments for serialization."""
        return {}

    @classmethod
    def from_config(cls, config: dict) -> "NeuronModel":
        """Construct from a config dict."""
        return cls(**config)
```

### 3.2 SynapseModel

```python
class SynapseModel(ABC):
    """Base class for synaptic transmission models.

    Synapses optionally maintain their own state (e.g., short-term
    plasticity variables, gating variables).
    """

    @abstractmethod
    def initialize_state(self, n_neurons: int, device: torch.device) -> dict[str, torch.Tensor]:
        """Create initial synapse state.

        For static synapses this may return an empty dict.
        """
        ...

    @abstractmethod
    def forward(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        state: dict[str, torch.Tensor],
        dt: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute synaptic current from presynaptic spikes.

        Args:
            weights: (n_pre, n_post) weight matrix.
            pre_spikes: (batch_size, n_pre) boolean spike tensor.
            state: Synapse state dict.
            dt: Simulation timestep.

        Returns:
            (post_current, new_state) where post_current is
            (batch_size, n_post).
        """
        ...
```

### 3.3 PlasticityRule

```python
class PlasticityRule(ABC):
    """Base class for synaptic plasticity rules.

    Rules observe pre/post spike activity and compute weight deltas.
    """

    @abstractmethod
    def initialize_state(self, n_neurons: int, device: torch.device) -> dict[str, torch.Tensor]:
        """State for trace variables, eligibility traces, etc."""
        ...

    @abstractmethod
    def compute_delta(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        state: dict[str, torch.Tensor],
        dt: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute weight change delta.

        Args:
            weights: (n_pre, n_post) current weights.
            pre_spikes: (batch_size, n_pre) boolean.
            post_spikes: (batch_size, n_post) boolean.
            state: Rule state dict.
            dt: Simulation timestep.

        Returns:
            (delta, new_state) where delta is (n_pre, n_post).
            The delta is averaged across the batch internally by the caller.
        """
        ...

    @abstractmethod
    def apply(
        self,
        weights: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Apply delta to weights with constraints (e.g., clipping, bounds)."""
        ...
```

### 3.4 StructuralPlasticity

```python
class StructuralPlasticity(ABC):
    """Base class for structural plasticity (growth and pruning)."""

    @abstractmethod
    def evaluate(
        self,
        adjacency: torch.Tensor,
        weights: torch.Tensor,
        activity: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate and apply structural changes.

        Args:
            adjacency: (n_neurons, n_neurons) binary connectivity matrix.
            weights: (n_neurons, n_neurons) weight matrix.
            activity: (batch_size, n_neurons) recent activity.
            step: Current training step.

        Returns:
            (new_adjacency, new_weights) — potentially modified.
        """
        ...
```

### 3.5 DendriticModule

```python
class DendriticModule(ABC):
    """Base class for dendritic computation."""

    @abstractmethod
    def initialize_state(self, n_neurons: int, n_dendrites: int, device: torch.device) -> dict[str, torch.Tensor]:
        """Dendritic state initialization."""
        ...

    @abstractmethod
    def forward(
        self,
        state: dict[str, torch.Tensor],
        input_current: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Process input through dendritic compartments.

        Args:
            state: Dendritic state.
            input_current: (batch_size, n_neurons) or (batch_size, n_neurons, n_dendrites).
            dt: Simulation timestep.

        Returns:
            (somatic_current, new_state) where somatic_current is
            (batch_size, n_neurons) — the output after dendritic integration.
        """
        ...
```

### 3.6 MemoryModule

```python
class MemoryModule(ABC):
    """Base class for memory mechanisms.

    Memory modules can store and retrieve information across timesteps.
    """

    @abstractmethod
    def initialize_state(self, n_neurons: int, device: torch.device) -> dict[str, torch.Tensor]:
        """Initial memory state."""
        ...

    @abstractmethod
    def read(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """Read from memory.

        Returns:
            Tensor of shape (batch_size, n_neurons) representing
            the memory readout contribution.
        """
        ...

    @abstractmethod
    def write(
        self,
        state: dict[str, torch.Tensor],
        input_data: torch.Tensor,
        spikes: torch.Tensor,
        dt: float,
    ) -> dict[str, torch.Tensor]:
        """Write to memory.

        Args:
            state: Current memory state.
            input_data: (batch_size, n_neurons) external input.
            spikes: (batch_size, n_neurons) current spike pattern.
            dt: Simulation timestep.

        Returns:
            Updated memory state.
        """
        ...
```

### 3.7 PredictionModule

```python
class PredictionModule(ABC):
    """Base class for predictive coding modules."""

    @abstractmethod
    def initialize_state(self, n_neurons: int, device: torch.device) -> dict[str, torch.Tensor]:
        """Initial prediction state."""
        ...

    @abstractmethod
    def predict(
        self,
        state: dict[str, torch.Tensor],
        spikes: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Generate prediction of next state.

        Returns:
            (prediction, new_state) where prediction is
            (batch_size, n_neurons).
        """
        ...

    @abstractmethod
    def compute_error(
        self,
        prediction: torch.Tensor,
        actual: torch.Tensor,
    ) -> torch.Tensor:
        """Compute prediction error.

        Returns:
            (batch_size, n_neurons) prediction error signal.
        """
        ...
```

### 3.8 Neuromodulator

```python
class Neuromodulator(ABC):
    """Base class for neuromodulatory signals."""

    @abstractmethod
    def initialize_state(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Initial modulator concentration/state."""
        ...

    @abstractmethod
    def compute_signal(
        self,
        state: dict[str, torch.Tensor],
        reward: torch.Tensor | None,
        global_activity: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute modulatory signal.

        Args:
            state: Modulator state.
            reward: Optional scalar reward signal (batch_size, 1) or None.
            global_activity: (batch_size, n_neurons) network activity.
            step: Current timestep.

        Returns:
            (signal, new_state) where signal is (batch_size, n_neurons)
            or (batch_size, 1) depending on the modulator type.
        """
        ...
```

### 3.9 Encoder

```python
class Encoder(ABC):
    """Base class for data-to-spike encoding."""

    @abstractmethod
    def encode(
        self,
        data: torch.Tensor,
        n_steps: int,
        dt: float,
    ) -> torch.Tensor:
        """Encode input data into spike trains.

        Args:
            data: (batch_size, input_dim) raw input.
            n_steps: Number of simulation timesteps.
            dt: Simulation timestep.

        Returns:
            (n_steps, batch_size, input_dim) boolean spike tensor.
        """
        ...
```

### 3.10 Decoder

```python
class Decoder(ABC):
    """Base class for spike-to-output decoding."""

    @abstractmethod
    def decode(
        self,
        spikes: torch.Tensor,
        n_steps: int,
    ) -> torch.Tensor:
        """Decode spike trains into output.

        Args:
            spikes: (n_steps, batch_size, n_neurons) spike tensor.
            n_steps: Number of timesteps used.

        Returns:
            (batch_size, output_dim) decoded output.
        """
        ...
```

### 3.11 TopologyRule

```python
class TopologyRule(ABC):
    """Base class for network topology generation."""

    @abstractmethod
    def generate(
        self,
        n_neurons: int,
        seed: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate an adjacency matrix.

        Args:
            n_neurons: Number of neurons.
            seed: Random seed for reproducibility.

        Returns:
            (n_neurons, n_neurons) binary adjacency matrix.
        """
        ...
```

### 3.12 LearningRule

```python
class LearningRule(ABC):
    """Base class for learning rules (gradient-based or local)."""

    @abstractmethod
    def compute_gradients(
        self,
        loss: torch.Tensor,
        model: "BIONNModel",
    ) -> dict[str, torch.Tensor]:
        """Compute gradients for all parameters.

        For surrogate gradient methods, this wraps autograd with
        a surrogate spike derivative.

        Returns:
            dict mapping parameter name -> gradient tensor.
        """
        ...

    @abstractmethod
    def update_weights(
        self,
        model: "BIONNModel",
        gradients: dict[str, torch.Tensor],
        lr: float,
    ) -> None:
        """Apply weight updates in-place."""
        ...
```

### 3.13 Metric

```python
class Metric(ABC):
    """Base class for evaluation metrics."""

    @abstractmethod
    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> float | dict[str, float]:
        """Compute metric value.

        Returns:
            Scalar metric value or dict of named sub-metrics.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable metric name."""
        ...

    @property
    @abstractmethod
    def direction(self) -> str:
        """'higher' or 'lower' — which direction is better."""
        ...
```

### 3.14 Visualizer

```python
class Visualizer(ABC):
    """Base class for visualization components."""

    @abstractmethod
    def render(
        self,
        data: dict[str, Any],
        save_path: str | None = None,
    ) -> matplotlib.Figure | None:
        """Render visualization from data dict.

        Args:
            data: Dict containing tensors, metrics, configs, etc.
            save_path: Optional path to save the figure.

        Returns:
            matplotlib Figure, or None if saved to file.
        """
        ...
```

---

## 4. Data Flow

### 4.1 Single Forward Pass

```
raw_input
    │
    ▼
┌──────────┐
│  Encoder  │  raw_input → spike_trains
└────┬─────┘
     │ spike_trains: (T, B, D_in)
     ▼
┌──────────────────────────────────────────────────────────────────┐
│                        NETWORK STEP LOOP                         │
│                                                                  │
│  For each timestep t = 0..T-1:                                  │
│    ┌──────────────┐                                              │
│    │  Dendrites   │  input → somatic_current                    │
│    └──────┬───────┘                                              │
│           ▼                                                      │
│    ┌──────────────┐    ┌──────────────┐                         │
│    │   Synapses   │◄───│ Pre-synaptic │                         │
│    └──────┬───────┘    │    spikes    │                         │
│           │             └──────────────┘                         │
│           ▼                                                      │
│    ┌──────────────┐    ┌──────────────┐                         │
│    │   Neurons    │◄───│   External   │                         │
│    └──────┬───────┘    │   current    │                         │
│           │             └──────────────┘                         │
│           │ spikes                                               │
│           ├──────────────────────────► Memory (write)            │
│           │                                                      │
│           ├──────────────────────────► Prediction (predict+error)│
│           │                                                      │
│           ▼                                                      │
│    ┌──────────────┐    ┌──────────────┐                         │
│    │  Plasticity  │◄───│  Neuromod    │                         │
│    └──────┬───────┘    └──────────────┘                         │
│           │ weight_deltas                                        │
│           ▼                                                      │
│    ┌──────────────┐                                              │
│    │   Structural │                                              │
│    │  (periodic)  │                                              │
│    └──────────────┘                                              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
     │
     │ accumulated spikes / states
     ▼
┌──────────┐
│  Decoder  │  spike_trains → output
└────┬─────┘
     │
     ▼
  output: (B, D_out)
```

### 4.2 Data Tensors Through the Pipeline

| Stage | Tensor | Shape | Type |
|-------|--------|-------|------|
| Encoder output | `spike_trains` | `(T, B, D_in)` | `bool` |
| Synapse output | `post_current` | `(B, N)` | `float32` |
| Neuron output | `spikes` | `(B, N)` | `bool` |
| Neuron output | `membrane_potential` | `(B, N)` | `float32` |
| Plasticity output | `weight_delta` | `(N, N)` | `float32` |
| Structural output | `adjacency` | `(N, N)` | `bool` |
| Memory read | `memory_output` | `(B, N)` | `float32` |
| Prediction output | `prediction` | `(B, N)` | `float32` |
| Prediction error | `pred_error` | `(B, N)` | `float32` |
| Neuromod signal | `modulatory_signal` | `(B, N)` or `(B, 1)` | `float32` |
| Decoder output | `output` | `(B, D_out)` | `float32` |

### 4.3 Inspection Hooks

Every module emits inspection data via the `InspectionBus`:

```python
class InspectionBus:
    """Central bus for collecting inspection data from all modules."""

    def record(self, module_name: str, tensor_name: str, tensor: torch.Tensor, step: int) -> None:
        """Record a tensor snapshot."""
        ...

    def get(self, module_name: str, tensor_name: str, step_range: tuple[int, int] | None = None) -> torch.Tensor:
        """Retrieve recorded tensors."""
        ...

    def get_all(self, step: int) -> dict[str, dict[str, torch.Tensor]]:
        """Get all recorded tensors for a given step."""
        ...
```

Modules call `bus.record(...)` during `forward()`. The bus stores data in a ring buffer (configurable max steps) to control memory usage.

---

## 5. Configuration System

### 5.1 YAML Config Structure

```yaml
experiment:
  name: "lif_stdp_mnist"
  seed: 42
  device: "cuda"

dataset:
  name: "mnist"
  path: "./data/mnist"
  batch_size: 64
  num_workers: 4

network:
  topology:
    class: "bio_nn.topology.random.RandomTopology"
    params:
      connectivity: 0.1
      seed: 42

  layers:
    - name: "excitatory"
      n_neurons: 800
      neuron:
        class: "bio_nn.neurons.lif.LIFNeuron"
        params:
          tau_mem: 20.0
          v_rest: -65.0
          v_thresh: -50.0
          v_reset: -65.0
          tau_refrac: 2.0
      synapse:
        class: "bio_nn.synapses.static.StaticSynapse"
        params: {}
      plasticity:
        class: "bio_nn.plasticity.stdp.STDPRule"
        params:
          tau_plus: 20.0
          tau_minus: 20.0
          a_plus: 0.01
          a_minus: 0.012
          w_max: 0.05
      dendrites:
        class: "bio_nn.dendrites.nonlinear.NonlinearDendrite"
        params:
          n_dendrites: 5
          nonlinearity: "relu"

    - name: "inhibitory"
      n_neurons: 200
      neuron:
        class: "bio_nn.neurons.lif.LIFNeuron"
        params:
          tau_mem: 10.0
          v_rest: -65.0
          v_thresh: -55.0
          v_reset: -65.0
      synapse:
        class: "bio_nn.synapses.dynamic.DynamicSynapse"
        params:
          tau_d: 200.0
          tau_f: 50.0
          U: 0.2
      plasticity:
        class: "bio_nn.plasticity.homeostatic.HomeostaticPlasticity"
        params:
          target_rate: 0.05
          timescale: 1000

  memory:
    class: "bio_nn.memory.recurrent.RecurrentMemory"
    params:
      memory_size: 100

  prediction:
    class: "bio_nn.prediction.predictor.TBPredictor"
    params:
      lookahead: 1

  neuromodulation:
    class: "bio_nn.neuromodulation.reward.RewardModulator"
    params:
      tau_da: 200.0
      baseline: 0.0

encoder:
  class: "bio_nn.encoders.rate.RateEncoder"
  params:
    n_steps: 15
    dt: 1.0
    max_rate: 200.0

decoder:
  class: "bio_nn.deoders.rate_decoder.RateDecoder"
  params:
    reduction: "mean"

learning:
  class: "bio_nn.learning.local.LocalLearning"
  params:
    optimizer: "sgd"
    lr: 0.01

training:
  epochs: 50
  trainer:
    class: "bio_nn.training.engine.TrainingEngine"
    params:
      loss: "cross_entropy"
      grad_clip: 1.0

structural:
  enabled: true
  class: "bio_nn.structural.pruning.ActivityPruning"
  params:
    threshold: 0.01
    interval: 1000

evaluation:
  metrics:
    - class: "bio_nn.evaluation.classification.Accuracy"
    - class: "bio_nn.evaluation.classification.CrossEntropy"
    - class: "bio_nn.evaluation.efficiency.SparsityMetric"
    - class: "bio_nn.evaluation.efficiency.FLOPsMetric"
    - class: "bio_nn.evaluation.robustness.NoiseRobustness"
      params:
        noise_levels: [0.1, 0.2, 0.3]

visualization:
  enabled: true
  static:
    - class: "bio_nn.visualization.static.spike_raster.SpikeRasterPlot"
    - class: "bio_nn.visualization.static.membrane.MembranePotentialPlot"
    - class: "bio_nn.visualization.static.weights.WeightHeatmap"
  dashboard:
    enabled: true
    port: 8080
```

### 5.2 Config Loading Flow

```
YAML file
    │
    ▼
┌──────────────┐
│ yaml.load()  │
└──────┬───────┘
       │ raw dict
       ▼
┌──────────────────┐
│ ConfigValidator  │  validates against JSON Schema
└──────┬───────────┘
       │ validated dict
       ▼
┌──────────────────┐
│ ConfigResolver   │  resolves class references, merges defaults
└──────┬───────────┘
       │ resolved config
       ▼
┌──────────────────┐
│ ExperimentManager│  creates components, wires them together
└──────────────────┘
```

### 5.3 Config Schema Validation

Each config section has a JSON Schema. The validator checks:

- All `class` values are valid dotted Python paths
- All referenced classes exist in the registry
- All required parameters are present
- All parameter types match (int, float, str, list)
- All parameter values are within valid ranges

```python
class ConfigValidator:
    def __init__(self, schema_path: str = "config/schema.json"):
        self.schema = self._load_schema(schema_path)

    def validate(self, config: dict) -> ValidationResult:
        """Validate config against schema.

        Returns:
            ValidationResult with errors list and warnings list.
        """
        ...

    def validate_component(self, category: str, config: dict) -> list[str]:
        """Validate a single component config section."""
        ...
```

### 5.4 Default Configs

The `defaults.py` module provides sensible defaults for every component. Users only need to specify overrides:

```python
DEFAULTS = {
    "experiment": {"seed": 42, "device": "auto"},
    "network": {
        "topology": {"class": "bio_nn.topology.random.RandomTopology", "params": {"connectivity": 0.1}},
    },
    "encoder": {"class": "bio_nn.encoders.rate.RateEncoder", "params": {"n_steps": 15, "dt": 1.0}},
    "decoder": {"class": "bio_nn.decoders.rate_decoder.RateDecoder", "params": {"reduction": "mean"}},
    "training": {"epochs": 50, "trainer": {"class": "bio_nn.training.engine.TrainingEngine"}},
}
```

---

## 6. Biological Plausibility Levels

Every component in BIO-NN is annotated with a plausibility level. This allows experiments to systematically trade off biological realism against computational tractability.

### 6.1 Level Definitions

| Level | Name | Description | Typical Use |
|-------|------|-------------|-------------|
| 0 | **Abstract** | No biological grounding. Abstract computation in SNN clothing. | Baseline / engineering SNN |
| 1 | **Inspired** | Loosely inspired by biology. Key mechanisms captured abstractly. | Rapid prototyping |
| 2 | **Constrained** | Follows known biological constraints (time constants, connectivity patterns). | Mechanism testing |
| 3 | **Detailed** | Implements specific biophysical mechanisms with realistic parameters. | Validation against data |
| 4 | **Biophysical** | Multi-compartment models with ion channels, detailed morphology. | Neuroscience modeling |

### 6.2 Component Classification

| Component | Level 0 | Level 1 | Level 2 | Level 3 | Level 4 |
|-----------|---------|---------|---------|---------|---------|
| **Neuron** | Threshold unit | LIF | Adaptive LIF | Izhikevich | Hodgkin-Huxley |
| **Synapse** | Linear weight | Static weight | Short-term dynamics | STP + NMDA | Multi-receptor |
| **Plasticity** | Gradient descent | Hebbian | STDP | R-STDP + calcium | Full calcium dynamics |
| **Dendrite** | Linear sum | ReLU pooling | Compartment-like | Multi-compartment | Morphological |
| **Topology** | All-to-all | Random | Small-world | Layered + motifs | Connectome-based |
| **Neuromod** | None | Scalar reward | Dopamine signal | Multi-amine | Full neurochemistry |
| **Encoding** | One-hot | Rate | Temporal | Latency + phase | Population + burst |
| **Structural** | Fixed | Pruning | Growth + pruning | Critical period | Developmental |

### 6.3 Plausibility Annotation

Each component class exposes a class-level attribute:

```python
class LIFNeuron(NeuronModel):
    plausibility_level: ClassVar[int] = 2
    plausibility_tags: ClassVar[list[str]] = ["leaky_integrate_and_fire", "refractory_period"]
```

The framework can filter components by level, enabling experiments like:

```yaml
experiment:
  max_plausibility_level: 2  # Only use components at level 0-2
```

---

## 7. Experiment Lifecycle

### 7.1 Phases

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────────┐     ┌──────────────┐     ┌─────────┐
│  CONFIG  │────▶│  BUILD  │────▶│  TRAIN  │────▶│  EVALUATE   │────▶│ VISUALIZE    │────▶│ REPORT  │
└─────────┘     └─────────┘     └─────────┘     └─────────────┘     └──────────────┘     └─────────┘
    │               │               │                │                     │                  │
    ▼               ▼               ▼                ▼                     ▼                  ▼
  Load YAML    Instantiate     Run epochs       Compute metrics      Generate plots     Save summary
  Validate     components      Checkpoint        Statistical tests   Dashboard update   Compare experiments
  Hash config  Wire topology   Log metrics       Ablation runs       Record figures     Write report
```

### 7.2 ExperimentManager

```python
class ExperimentManager:
    """Manages the full lifecycle of a BIO-NN experiment."""

    def __init__(self, config_path: str):
        """Load config from YAML path."""
        ...

    def build(self) -> "BIONNModel":
        """Phase 2: Instantiate all components from config."""
        ...

    def train(self, model: "BIONNModel") -> TrainingLog:
        """Phase 3: Run training loop."""
        ...

    def evaluate(self, model: "BIONNModel") -> EvaluationReport:
        """Phase 4: Compute all configured metrics."""
        ...

    def visualize(self, model: "BIONNModel", results: dict) -> list[str]:
        """Phase 5: Generate all configured visualizations."""
        ...

    def report(self, results: dict) -> str:
        """Phase 6: Generate and save experiment report."""
        ...

    def run(self) -> ExperimentResult:
        """Execute all phases end-to-end."""
        ...
```

### 7.3 ExperimentResult

```python
@dataclass
class ExperimentResult:
    experiment_id: str          # UUID
    code_commit: str            # Git SHA
    config_hash: str            # SHA-256 of YAML content
    dataset_hash: str           # SHA-256 of dataset
    seed: int
    hardware: HardwareInfo      # GPU, CPU, RAM
    wall_clock: float           # Total seconds
    metrics: dict[str, float]   # Final metric values
    metric_history: dict[str, list[float]]  # Per-epoch values
    checkpoints: list[str]      # Paths to saved checkpoints
    visualizations: list[str]   # Paths to saved figures
    config: dict                # Full resolved config
```

### 7.4 Checkpoint System

```python
class CheckpointManager:
    def save(self, model: "BIONNModel", epoch: int, metrics: dict) -> str:
        """Save model state, optimizer state, and metadata.

        Returns:
            Path to checkpoint file.
        """
        ...

    def load(self, checkpoint_path: str) -> tuple["BIONNModel", int, dict]:
        """Restore model and training state.

        Returns:
            (model, start_epoch, metrics)
        """
        ...

    def list_checkpoints(self, experiment_id: str) -> list[str]:
        """List all checkpoints for an experiment."""
        ...
```

### 7.5 Experiment Comparison

```python
class ExperimentComparator:
    def compare(self, experiment_ids: list[str]) -> ComparisonReport:
        """Compare multiple experiments.

        Returns:
            ComparisonReport with side-by-side metrics,
            statistical significance tests, and config diffs.
        """
        ...

    def plot_comparison(self, experiment_ids: list[str], metric: str) -> matplotlib.Figure:
        """Plot metric curves for multiple experiments."""
        ...
```

---

## 8. Registry and Dynamic Loading

### 8.1 Registry Design

```python
_registry: dict[str, dict[str, type]] = {}

def register(category: str, name: str):
    """Decorator to register a component class.

    Usage:
        @register("neurons", "lif")
        class LIFNeuron(NeuronModel):
            ...
    """
    def decorator(cls):
        if category not in _registry:
            _registry[category] = {}
        _registry[category][name] = cls
        cls._registry_name = name
        cls._registry_category = category
        return cls
    return decorator


def get_class(category: str, name: str) -> type:
    """Retrieve a registered class by category and name."""
    return _registry[category][name]


def instantiate(category: str, name: str, params: dict) -> Any:
    """Instantiate a registered class with given params."""
    cls = get_class(category, name)
    return cls.from_config(params) if hasattr(cls, 'from_config') else cls(**params)


def list_components(category: str | None = None) -> dict:
    """List all registered components."""
    if category:
        return {name: cls.__doc__ for name, cls in _registry.get(category, {}).items()}
    return {cat: list(comps.keys()) for cat, comps in _registry.items()}
```

### 8.2 Dynamic Loading in Config

```yaml
neuron:
  class: "bio_nn.neurons.izhikevich.IzhikevichNeuron"
  params:
    a: 0.02
    b: 0.2
    c: -65.0
    d: 8.0
```

The config loader:

1. Splits `"bio_nn.neurons.izhikevich.IzhikevichNeuron"` into module path + class name
2. Dynamically imports the module
3. Looks up the class
4. Instantiates with `params`

Alternatively, using the registry shorthand:

```yaml
neuron:
  name: "izhikevich"  # registered name
  params:
    a: 0.02
    ...
```

### 8.3 Discovery

All modules under `bio_nn/` are auto-discovered at import time via `__init__.py` files that import all submodules. The registry is populated during import. Manual discovery is also supported:

```python
from bio_nn.core.registry import discover_components
discover_components("bio_nn/")  # Scans and imports all .py files
```

---

## 9. Inspection and Observability

### 9.1 TimestepRecorder

```python
class TimestepRecorder:
    """Records module states at each timestep for later analysis."""

    def __init__(self, bus: InspectionBus, config: dict):
        """Configure which tensors to record and at what frequency."""
        self.bus = bus
        self.record_every = config.get("record_every", 1)
        self.max_steps = config.get("max_steps", 10000)
        self.tensors_to_record = config.get("tensors", [])

    def step_hook(self, module_name: str, step: int, tensors: dict[str, torch.Tensor]):
        """Called at each timestep. Records selected tensors."""
        if step % self.record_every != 0:
            return
        for name, tensor in tensors.items():
            if name in self.tensors_to_record:
                self.bus.record(module_name, name, tensor.detach().cpu(), step)
```

### 9.2 Available Inspection Points

| Module | Tensors Available | Description |
|--------|-------------------|-------------|
| Neuron | `v`, `spikes`, `refrac_counter`, `adaptation` | Membrane voltage, spikes, refractory state |
| Synapse | `current`, `short_term_u`, `short_term_x` | Synaptic current, STP variables |
| Plasticity | `delta`, `traces_pre`, `traces_post`, `eligibility` | Weight changes, spike traces |
| Dendrite | `dendritic_potential`, `somatic_current` | Dendritic integration |
| Memory | `memory_state`, `read_output`, `write_gate` | Memory contents |
| Prediction | `prediction`, `prediction_error`, `beta` | Predictive signals |
| Neuromod | `concentration`, `modulatory_signal` | Neuromodulator levels |
| Structural | `new_edges`, `pruned_edges`, `n_connections` | Topology changes |

### 9.3 Real-Time Dashboard

The dashboard server exposes WebSocket connections for live monitoring:

```python
# FastAPI backend
@app.websocket("/ws/metrics")
async def metrics_stream(websocket: WebSocket):
    """Stream training metrics in real-time."""
    ...

@app.get("/api/tensors/{module}/{name}")
async def get_tensor(module: str, name: str, step: int = -1):
    """Get recorded tensor at a specific timestep."""
    ...

@app.get("/api/config")
async def get_config():
    """Get the current experiment configuration."""
    ...

@app.get("/api/topology")
async def get_topology():
    """Get current network adjacency matrix."""
    ...
```

---

## 10. Reproducibility Contract

### 10.1 Seed Management

```python
class SeedManager:
    """Ensures reproducibility via deterministic seeding."""

    def __init__(self, master_seed: int):
        self.master_seed = master_seed

    def seed_all(self):
        """Seed Python random, NumPy, PyTorch CPU, PyTorch CUDA."""
        random.seed(self.master_seed)
        np.random.seed(self.master_seed)
        torch.manual_seed(self.master_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.master_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def derive_seed(self, component_name: str) -> int:
        """Derive a deterministic sub-seed for a component.

        Ensures different components get different seeds
        while remaining reproducible from the master seed.
        """
        return self.master_seed + hash(component_name) % (2**31)
```

### 10.2 Artifact Hashing

```python
class ArtifactHasher:
    """Compute deterministic hashes for reproducibility tracking."""

    @staticmethod
    def hash_config(config_path: str) -> str:
        """SHA-256 hash of YAML file contents."""
        ...

    @staticmethod
    def hash_dataset(dataset_path: str) -> str:
        """SHA-256 hash of dataset file(s)."""
        ...

    @staticmethod
    def hash_code(repo_path: str) -> str:
        """Git HEAD commit SHA."""
        ...

    @staticmethod
    def fingerprint(config_path: str, dataset_path: str, repo_path: str, seed: int) -> str:
        """Compute full experiment fingerprint.

        Returns:
            Compact string: "commit[:8]-config[:8]-dataset[:8]-seed"
        """
        ...
```

### 10.3 Hardware Recording

```python
@dataclass
class HardwareInfo:
    gpu_name: str | None
    gpu_memory: int | None
    cpu_name: str
    ram_gb: float
    cuda_version: str | None
    pytorch_version: str
    python_version: str

    @classmethod
    def detect(cls) -> "HardwareInfo":
        """Auto-detect current hardware."""
        ...

    def to_dict(self) -> dict:
        return asdict(self)
```

### 10.4 Result Storage Layout

```
results/
├── {experiment_id}/
│   ├── metadata.json          # fingerprint, hardware, wall_clock
│   ├── config.yaml            # resolved config used
│   ├── metrics.json           # final metrics
│   ├── metrics_history.csv    # per-epoch metrics
│   ├── checkpoints/
│   │   ├── epoch_001.pt
│   │   ├── epoch_010.pt
│   │   └── best.pt
│   ├── visualizations/
│   │   ├── spike_raster.png
│   │   ├── membrane_potential.png
│   │   └── weight_heatmap.png
│   ├── tensors/               # recorded inspection data
│   │   ├── neurons/exc/v.pt
│   │   └── neurons/exc/spikes.pt
│   └── report.md              # auto-generated experiment report
```

### 10.5 metadata.json Schema

```json
{
  "experiment_id": "uuid",
  "experiment_name": "lif_stdp_mnist",
  "code_commit": "a1b2c3d4e5f6",
  "config_hash": "sha256:abcdef...",
  "dataset_hash": "sha256:123456...",
  "seed": 42,
  "hardware": {
    "gpu_name": "NVIDIA RTX 4090",
    "gpu_memory": 24576,
    "cpu_name": "AMD Ryzen 9 7950X",
    "ram_gb": 64.0,
    "cuda_version": "12.1",
    "pytorch_version": "2.4.0",
    "python_version": "3.12.0"
  },
  "wall_clock_seconds": 3420.5,
  "plausibility_levels": [2, 2, 1],
  "created_at": "2026-09-19T14:30:00Z",
  "bio_nn_version": "1.0.0"
}
```

---

## Appendix A: Module Interface Summary

| Module Slot | `initialize_state()` | `forward()` / Core Method | State Dict Keys |
|-------------|----------------------|---------------------------|-----------------|
| `NeuronModel` | `(B, N) → dict` | `forward(state, I, dt) → (state, spikes)` | `v`, `refrac`, `adaptation` |
| `SynapseModel` | `(N, N) → dict` | `forward(W, pre_spikes, state, dt) → (I, state)` | `x`, `u` (STP) |
| `PlasticityRule` | `(N, N) → dict` | `compute_delta(W, pre, post, state, dt) → (ΔW, state)` | `trace_pre`, `trace_post`, `elig` |
| `StructuralPlasticity` | N/A | `evaluate(adj, W, activity, step) → (adj', W')` | N/A |
| `DendriticModule` | `(N, D) → dict` | `forward(state, I, dt) → (I_soma, state)` | `dend_v` |
| `MemoryModule` | `(N) → dict` | `read(state) → I_mem`; `write(state, I, spikes, dt) → state` | `content`, `gate` |
| `PredictionModule` | `(N) → dict` | `predict(state, spikes) → (pred, state)`; `error(pred, actual) → err` | `pred_state`, `beta` |
| `Neuromodulator` | `() → dict` | `compute_signal(state, reward, activity, step) → (signal, state)` | `concentration` |
| `Encoder` | N/A | `encode(data, T, dt) → spikes (T, B, D)` | N/A |
| `Decoder` | N/A | `decode(spikes, T) → output (B, D_out)` | N/A |
| `TopologyRule` | N/A | `generate(N, seed) → adj (N, N)` | N/A |
| `LearningRule` | N/A | `compute_gradients(loss, model) → grads`; `update_weights(model, grads, lr)` | N/A |

---

## Appendix B: Error Handling

```python
class BIONNError(Exception):
    """Base exception for BIO-NN framework."""
    pass

class ConfigError(BIONNError):
    """Configuration validation failed."""
    pass

class ComponentError(BIONNError):
    """Component initialization or execution failed."""
    pass

class InspectionError(BIONNError):
    """Inspection data retrieval failed."""
    pass

class ReproducibilityError(BIONNError):
    """Reproducibility check failed (hash mismatch, seed conflict)."""
    pass
```

---

## Appendix C: Dependency Matrix

| Component | PyTorch | NumPy | Matplotlib | FastAPI | PyYAML | Optional |
|-----------|---------|-------|------------|---------|--------|----------|
| core/ | ✓ | | | | | |
| neurons/ | ✓ | | | | | |
| synapses/ | ✓ | | | | | |
| plasticity/ | ✓ | | | | | |
| dendrites/ | ✓ | | | | | |
| structural/ | ✓ | | | | | |
| memory/ | ✓ | | | | | |
| prediction/ | ✓ | | | | | |
| neuromodulation/ | ✓ | | | | | |
| encoders/ | ✓ | | | | | |
| decoders/ | ✓ | | | | | |
| topology/ | ✓ | ✓ | | | | |
| learning/ | ✓ | | | | | |
| training/ | ✓ | | | | | |
| evaluation/ | ✓ | ✓ | | | | |
| visualization/static/ | | ✓ | ✓ | | | |
| visualization/dashboard/ | | | | ✓ | | websockets |
| experiments/ | | | | | ✓ | |
| config/ | | | | | ✓ | jsonschema |
| utils/ | ✓ | ✓ | | | | psutil |

---

*End of architecture specification.*
