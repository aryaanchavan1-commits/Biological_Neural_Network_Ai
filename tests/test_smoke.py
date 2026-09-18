import torch
import yaml
from bio_nn.core.model_builder import build_model

def test_build_baseline_snn():
    """Test that baseline SNN config builds and runs."""
    with open("D:/Arynoxtech/Bio_NN/configs/baseline_snn.yaml") as f:
        config = yaml.safe_load(f)
    model = build_model(config)
    assert model is not None
    # Run one forward pass
    data = torch.randn(4, 784)
    output = model(data)
    assert output.shape[0] == 4

def test_build_bio_v01():
    """Test that full BIO-NN config builds."""
    with open("D:/Arynoxtech/Bio_NN/configs/bio_v01.yaml") as f:
        config = yaml.safe_load(f)
    model = build_model(config)
    assert model is not None
    data = torch.randn(2, 784)
    output = model(data)
    assert output.shape[0] == 2
