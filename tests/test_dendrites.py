import torch
from bio_nn.dendrites import get_dendrite

class TestCompartmentDendrite:
    def test_forward(self):
        d = get_dendrite("compartment", {"n_branches": 4, "combine": "multiply", "in_features": 128, "out_features": 64})
        x = torch.randn(8, 128)
        out = d.forward(x)
        assert out.shape == (8, 64)

class TestNonlinearDendrite:
    def test_forward(self):
        d = get_dendrite("nonlinear", {"n_branches": 4, "gate_type": "relu", "in_features": 128, "out_features": 64})
        x = torch.randn(8, 128)
        out = d.forward(x)
        assert out.shape == (8, 64)
