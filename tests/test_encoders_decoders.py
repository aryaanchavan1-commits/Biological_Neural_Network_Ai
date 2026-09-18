import torch
from bio_nn.encoders import create_encoder
from bio_nn.decoders import create_decoder

class TestRateEncoder:
    def test_encode(self):
        enc = create_encoder("rate", {"max_rate": 100.0})
        data = torch.rand(8, 784)
        spikes = enc.encode(data, time_steps=25)
        assert spikes.shape == (25, 8, 784)

class TestTemporalEncoder:
    def test_encode(self):
        enc = create_encoder("temporal", {})
        data = torch.rand(8, 784)
        spikes = enc.encode(data, time_steps=25)
        assert spikes.shape == (25, 8, 784)

class TestRateDecoder:
    def test_decode(self):
        dec = create_decoder("rate", {})
        spikes = torch.randn(25, 8, 64).sigmoid()
        out = dec.decode(spikes)
        assert out.shape == (8, 64)

class TestSpikeDecoder:
    def test_decode(self):
        dec = create_decoder("spike", {"mode": "last"})
        spikes = (torch.randn(25, 8, 64) > 0).float()
        out = dec.decode(spikes)
        assert out.shape == (8, 64)
