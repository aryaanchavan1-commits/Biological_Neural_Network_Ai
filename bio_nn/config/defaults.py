"""Default configuration values for the BIO-NN framework."""

from __future__ import annotations

from typing import Any, Dict

DEFAULTS: Dict[str, Any] = {
    "experiment": {
        "name": "unnamed",
        "seed": 42,
        "epochs": 100,
        "log_interval": 10,
        "checkpoint_interval": 25,
    },
    "model": {
        "type": "lif_neuron",
        "n_neurons": 128,
        "n_layers": 3,
        "dt": 1.0,
        "tau_mem": 20.0,
        "tau_syn": 5.0,
        "threshold": 1.0,
        "dropout": 0.0,
    },
    "data": {
        "dataset": "mnist",
        "batch_size": 64,
        "num_workers": 0,
        "pin_memory": True,
        "shuffle": True,
    },
    "training": {
        "optimizer": "adam",
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "lr_scheduler": "cosine",
        "lr_min": 1e-6,
    },
    "synapse": {
        "type": "static_synapse",
        "w_max": 1.0,
        "w_min": 0.0,
        "init_std": 0.01,
    },
    "plasticity": {
        "type": "stdp",
        "a_plus": 0.01,
        "a_minus": 0.012,
        "tau_plus": 20.0,
        "tau_minus": 20.0,
    },
    "topology": {
        "type": "random",
        "p_connect": 0.1,
        "lateral_inhibition": False,
    },
    "encoder": {
        "type": "rate",
        "max_rate": 100.0,
        "duration": 0.05,
    },
    "decoder": {
        "type": "spike_count",
    },
    "device": "auto",
    "output_dir": "outputs",
}
