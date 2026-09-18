# Installation Guide

## Requirements

- **Python**: 3.9 or higher
- **pip**: 21.0 or higher
- **OS**: Linux, macOS, or Windows
- **GPU** (optional): CUDA-capable GPU with CUDA 11.7+ for accelerated training

## Virtual Environment Setup

Always use a virtual environment to avoid dependency conflicts.

### Using venv

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### Using conda

```bash
conda create -n bio-nn python=3.10
conda activate bio-nn
```

## Basic Installation

```bash
git clone https://github.com/your-repo/bio-nn.git
cd bio-nn
pip install -e .
```

This installs the core framework with all required dependencies.

## PyTorch Installation

### CPU only

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### CUDA 11.8

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### CUDA 12.1

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify installation:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

## snnTorch Installation

snnTorch is installed automatically with the core package. To install separately:

```bash
pip install snntorch
```

Verify:

```bash
python -c "import snntorch; print(f'snnTorch {snntorch.__version__}')"
```

## Optional Dependencies

### Dashboard

```bash
pip install -e ".[dashboard]"
```

Includes: Plotly, Dash

### Development tools

```bash
pip install -e ".[dev]"
```

Includes: pytest, black, flake8, mypy

## Troubleshooting

### CUDA not found

1. Ensure CUDA toolkit is installed: `nvcc --version`
2. Ensure nvidia-smi shows your GPU: `nvidia-smi`
3. Install matching PyTorch CUDA version (see PyTorch section above)

### snnTorch import errors

```bash
pip install --force-reinstall snntorch
```

### Permission errors on Windows

Run terminal as administrator, or use:

```bash
pip install -e . --user
```

### Package version conflicts

```bash
pip install --upgrade pip
pip install --no-cache-dir -e .
```

## Verify Installation

```bash
python -c "
from bio_nn.neurons.lif import LIFNeuron
from bio_nn.plasticity.stdp import STDP
from bio_nn.networks.snn import SpikingNetwork
print('BIO-NN installed successfully')
"
```
