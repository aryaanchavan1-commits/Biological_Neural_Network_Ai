# Extending BIO-NN

This guide covers adding new components to the framework.

## Adding a New Neuron Model

1. Create a new file in `bio_nn/neurons/`:

```python
# bio_nn/neurons/my_neuron.py
import torch
import torch.nn as nn


class MyNeuron(nn.Module):
    def __init__(self, input_size, output_size, **params):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.threshold = params.get('threshold', 1.0)
        self.reset_potential = params.get('reset_potential', 0.0)
        # Initialize state
        self.register_buffer('membrane_potential', torch.zeros(output_size))

    def forward(self, input_spikes, state=None):
        # Update membrane potential
        self.membrane_potential += input_spikes
        # Check threshold
        spikes = (self.membrane_potential >= self.threshold).float()
        # Reset
        self.membrane_potential = torch.where(
            spikes.bool(),
            torch.full_like(self.membrane_potential, self.reset_potential),
            self.membrane_potential
        )
        return spikes, self.membrane_potential

    def reset_state(self):
        self.membrane_potential.zero_()
```

2. Register it in `bio_nn/neurons/__init__.py`:

```python
from .my_neuron import MyNeuron

NEURON_REGISTRY = {
    'lif': LIFNeuron,
    'izhikevich': IzhikevichNeuron,
    'my_neuron': MyNeuron,
}
```

3. Reference it in your YAML config:

```yaml
network:
  neuron_model: my_neuron
  neuron_params:
    threshold: 1.5
    reset_potential: -0.5
```

## Adding a New Plasticity Rule

1. Create a new file in `bio_nn/plasticity/`:

```python
# bio_nn/plasticity/my_plasticity.py
import torch


class MyPlasticity:
    def __init__(self, **params):
        self.learning_rate = params.get('learning_rate', 0.01)

    def update(self, pre_spikes, post_spikes, weights):
        # Your plasticity rule here
        delta = self.compute_weight_change(pre_spikes, post_spikes)
        weights = weights + self.learning_rate * delta
        return weights

    def compute_weight_change(self, pre_spikes, post_spikes):
        raise NotImplementedError
```

2. Register it in `bio_nn/plasticity/__init__.py`

3. Reference it in YAML:

```yaml
plasticity:
  my_plasticity:
    enabled: true
    learning_rate: 0.01
```

## Adding a New Topology Rule

1. Create a new file in `bio_nn/structure/`:

```python
# bio_nn/structure/my_topology.py
import torch


class MyTopology:
    def __init__(self, **params):
        self.param = params.get('param', 0.1)

    def rewire(self, adjacency, activity):
        # Your topology rule here
        new_adjacency = adjacency.clone()
        # Modify connections based on activity
        return new_adjacency
```

2. Register it in `bio_nn/structure/__init__.py`

## Adding a New Dataset

1. Create a new file in `bio_nn/datasets/`:

```python
# bio_nn/datasets/my_dataset.py
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # Load your data
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
```

2. Register it in `bio_nn/datasets/__init__.py`

## Adding a New Metric

1. Create a new file in `bio_nn/metrics/`:

```python
# bio_nn/metrics/my_metric.py
class MyMetric:
    def __init__(self):
        self.values = []

    def update(self, predictions, targets):
        # Compute your metric
        value = self.compute(predictions, targets)
        self.values.append(value)

    def compute(self, predictions, targets):
        raise NotImplementedError

    def get(self):
        return self.values
```

2. Register it in `bio_nn/metrics/__init__.py`

## Adding a New Visualization

1. Create a new file in `bio_nn/visualization/`:

```python
# bio_nn/visualization/my_viz.py
def plot_my_metric(results, save_path=None):
    """Plot your custom visualization."""
    import matplotlib.pyplot as plt
    # Create your plot
    plt.figure()
    # ... plotting code ...
    if save_path:
        plt.savefig(save_path)
    plt.close()
```

2. Call it from the experiment runner or dashboard.

## Testing Your Extensions

Run the test suite to ensure nothing is broken:

```bash
pytest tests/
```

Add tests for your new component:

```python
# tests/test_my_neuron.py
from bio_nn.neurons.my_neuron import MyNeuron

def test_my_neuron_forward():
    neuron = MyNeuron(10, 5)
    spikes, state = neuron(torch.ones(10))
    assert spikes.shape == (5,)
```
