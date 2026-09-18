import torch
from bio_nn.synapses import create_synapse

class TestStaticSynapse:
    def test_forward(self):
        syn = create_synapse("static", 128, 64, {"weight_init": {"method": "xavier"}})
        pre_spikes = torch.randn(8, 128).sigmoid()
        out = syn(pre_spikes)
        assert out.shape == (8, 64)

class TestDynamicSynapse:
    def test_traces(self):
        syn = create_synapse("dynamic", 128, 64, {"tau_pre": 20.0, "tau_post": 20.0})
        pre = torch.randn(8, 128).sigmoid()
        out = syn(pre)
        assert out.shape == (8, 64)
        stats = syn.get_statistics()
        assert "pre_trace_norm" in stats
