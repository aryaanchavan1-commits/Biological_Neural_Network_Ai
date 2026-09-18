import torch
import numpy as np
from bio_nn.evaluation.metrics import accuracy, f1_score, forgetting, average_accuracy

class TestMetrics:
    def test_accuracy(self):
        pred = torch.tensor([0, 1, 2, 0, 1])
        target = torch.tensor([0, 1, 1, 0, 2])
        acc = accuracy(pred, target)
        assert 0.0 <= acc <= 1.0
    
    def test_forgetting(self):
        per_task = [[90, 85, 80], [80, 75, 70]]
        f = forgetting(per_task)
        assert isinstance(f, float)
    
    def test_average_accuracy(self):
        per_task = [[90, 85, 80], [80, 75, 70]]
        avg = average_accuracy(per_task)
        assert isinstance(avg, float)
