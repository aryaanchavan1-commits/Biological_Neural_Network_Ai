import torch
from bio_nn.structural import create_structural

class TestGrowth:
    def test_growth(self):
        rule = create_structural("growth", {"growth_rate": 0.1, "activity_threshold": 0.1})
        w = torch.randn(64, 32) * 0.1
        activity = {"pre_rates": torch.rand(64), "post_rates": torch.rand(32)}
        w_new = rule.modify(w, activity=activity, epoch=0)
        assert w_new.shape == w.shape

class TestPruning:
    def test_pruning(self):
        rule = create_structural("pruning", {"pruning_rate": 0.1, "weight_threshold": 0.01})
        w = torch.randn(64, 32) * 0.1
        n_before = (w.abs() > 0.01).sum().item()
        w_new = rule.modify(w, epoch=0)
        n_after = (w_new.abs() > 0.01).sum().item()
        assert n_after <= n_before
