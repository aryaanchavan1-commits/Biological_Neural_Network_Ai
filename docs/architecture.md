# Architecture Overview

BIO-NN follows a modular, component-based architecture where each biological mechanism is an independent, swappable module.

## Design Principles

1. **Separation of concerns** — neurons, plasticity, structure, and encoding are independent subsystems
2. **Configuration-driven** — all behavior defined in YAML, no code changes for experiments
3. **Composability** — mechanisms can be combined freely via configuration
4. **Reproducibility** — full experiment state is captured and restorable

## Core Components

```
┌─────────────────────────────────────────────────┐
│                  Experiment Runner               │
├──────────┬──────────┬──────────┬────────────────┤
│ Neurons  │Plasticity│ Structure│   Encoding     │
│          │          │          │                │
│ • LIF    │ • STDP   │ • Small  │ • Rate         │
│ • Izhik. │ • Homeo. │   World  │ • Temporal     │
│          │          │ • Scale  │ • Population   │
├──────────┴──────────┴──────────┴────────────────┤
│              Spiking Network                     │
├─────────────────────────────────────────────────┤
│              Metrics & Visualization             │
└─────────────────────────────────────────────────┘
```

## Module Interfaces

Each module type follows a consistent interface:

- **Neurons**: `forward(input, state) -> (output, new_state)`
- **Plasticity**: `update(pre_spikes, post_spikes, weights) -> new_weights`
- **Structure**: `rewire(adjacency, activity) -> new_adjacency`
- **Encoding**: `encode(data) -> spikes`

## Data Flow

```
Input Data
    │
    ▼
Encoding ──► Spike Train
                │
                ▼
        Spiking Network
        ┌───────┴───────┐
        │               │
   Forward Pass    Plasticity Update
        │               │
        ▼               ▼
   Output         Weight Update
        │               │
        ▼               ▼
   Metrics      Structure Rewire
```

## Detailed Specification

For the full architecture specification including class diagrams, configuration schema, and API contracts, see [research/architecture_spec.md](../research/architecture_spec.md).
