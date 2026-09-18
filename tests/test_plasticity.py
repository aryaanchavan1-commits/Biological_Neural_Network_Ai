import torch
from bio_nn.plasticity import create_plasticity

class TestHebbian:
    def test_update(self):
        rule = create_plasticity("hebbian", {"learning_rate": 0.01})
        w = torch.randn(64, 32) * 0.1
        pre = torch.randn(8, 64).sigmoid()
        post = torch.randn(8, 32).sigmoid()
        w_new = rule.update(w, pre, post)
        assert w_new.shape == w.shape

class TestSTDP:
    def test_update(self):
        rule = create_plasticity("stdp", {"learning_rate": 0.01, "tau_pre": 20.0, "tau_post": 20.0})
        w = torch.randn(64, 32) * 0.1
        pre = (torch.randn(8, 64) > 0.5).float()
        post = (torch.randn(8, 32) > 0.5).float()
        w_new = rule.update(w, pre, post)
        assert w_new.shape == w.shape

class TestHomeostatic:
    def test_update(self):
        rule = create_plasticity("homeostatic", {"target_rate": 0.05, "rate_window": 10})
        w = torch.ones(64, 32) * 0.1
        pre = (torch.randn(8, 64) > 0.9).float()
        post = (torch.randn(8, 32) > 0.9).float()
        w_new = rule.update(w, pre, post)
        assert w_new.shape == w.shape
