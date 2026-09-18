import os, tempfile, yaml
from bio_nn.config.loader import load_config, Config

class TestConfig:
    def test_load_yaml(self):
        config = load_config("D:/Arynoxtech/Bio_NN/configs/baseline_snn.yaml")
        assert config is not None
        assert "experiment" in config
    
    def test_config_access(self):
        config = load_config("D:/Arynoxtech/Bio_NN/configs/baseline_snn.yaml")
        assert config.experiment.name == "baseline_snn"
