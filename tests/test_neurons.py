import pytest
import torch
from bio_nn.neurons import create_neuron, NEURON_REGISTRY

class TestLIFNeuron:
    def test_creation(self):
        neuron = create_neuron("lif", 128, {"tau_mem": 20.0, "threshold": 1.0})
        assert neuron is not None
    
    def test_forward_shape(self):
        neuron = create_neuron("lif", 128, {"tau_mem": 20.0, "threshold": 1.0})
        x = torch.randn(8, 128)
        state = neuron.get_initial_state(batch_size=8)
        spikes, v, new_state = neuron(x, state)
        assert spikes.shape == (8, 128)
        assert v.shape == (8, 128)
    
    def test_spike_rate(self):
        neuron = create_neuron("lif", 128, {"tau_mem": 20.0, "threshold": 0.5})
        state = neuron.get_initial_state(batch_size=32)
        total_spikes = 0
        for _ in range(100):
            x = torch.ones(32, 128) * 2.0
            spikes, v, state = neuron(x, state)
            total_spikes += spikes.sum().item()
        rate = total_spikes / (100 * 32 * 128)
        assert 0.0 < rate < 1.0

class TestAdaptiveLIF:
    def test_creation(self):
        neuron = create_neuron("adaptive_lif", 128, {"tau_mem": 20.0, "tau_adapt": 100.0, "threshold": 1.0})
        assert neuron is not None
    
    def test_forward(self):
        neuron = create_neuron("adaptive_lif", 64, {"tau_mem": 20.0, "tau_adapt": 100.0, "threshold": 1.0})
        state = neuron.get_initial_state(batch_size=4)
        x = torch.randn(4, 64)
        spikes, v, state = neuron(x, state)
        assert spikes.shape == (4, 64)

class TestIzhikevich:
    def test_creation(self):
        neuron = create_neuron("izhikevich", 128, {"preset": "regular_spiking"})
        assert neuron is not None
    
    def test_forward(self):
        neuron = create_neuron("izhikevich", 64, {"preset": "regular_spiking"})
        state = neuron.get_initial_state(batch_size=4)
        x = torch.randn(4, 64) * 5
        spikes, v, state = neuron(x, state)
        assert spikes.shape == (4, 64)

def test_neuron_registry():
    assert "lif" in NEURON_REGISTRY
    assert "adaptive_lif" in NEURON_REGISTRY
    assert "izhikevich" in NEURON_REGISTRY

def test_unknown_neuron_raises():
    with pytest.raises(ValueError):
        create_neuron("unknown", 128, {})
