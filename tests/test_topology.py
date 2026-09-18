from bio_nn.topology import build_topology

class TestRandomTopology:
    def test_generate(self):
        topo = build_topology({"type": "random", "sparsity": 0.5})
        mask = topo.generate(128, 64)
        assert mask.shape == (128, 64)
        assert mask.min() >= 0
        assert mask.max() <= 1

class TestSmallWorld:
    def test_generate(self):
        topo = build_topology({"type": "small_world", "k": 4, "beta": 0.3})
        mask = topo.generate(128, 128)
        assert mask.shape == (128, 128)
