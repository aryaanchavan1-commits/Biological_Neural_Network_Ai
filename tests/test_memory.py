import torch
from bio_nn.memory import get_memory

class TestRecurrentMemory:
    def test_read_write(self):
        mem = get_memory("recurrent", {"hidden_size": 128})
        mem.write(torch.randn(8, 64))
        out = mem.read()
        assert out.shape == (8, 128)

class TestWorkingMemory:
    def test_buffer(self):
        mem = get_memory("working", {"buffer_size": 10, "input_size": 64})
        for _ in range(15):
            mem.write(torch.randn(8, 64))
        out = mem.read()
        assert out is not None
