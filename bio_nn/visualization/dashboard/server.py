"""
BIO-NN 3D Visualization Dashboard
FastAPI backend serving the interactive Three.js neural network visualizer.
"""
import json
import os
import glob
import math
import random
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="BIO-NN 3D Dashboard", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
EXPERIMENTS_DIR = Path(os.environ.get("BIONN_DATA_DIR", BASE_DIR.parent.parent / "experiments"))
TEMPLATE_DIR = BASE_DIR / "templates"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

EXPERIMENT_DB = {}


def _scan_experiments():
    global EXPERIMENT_DB
    if not EXPERIMENTS_DIR.exists():
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    for p in EXPERIMENTS_DIR.iterdir():
        if p.is_dir():
            meta_file = p / "metadata.json"
            results_file = p / "results.json"
            config_file = p / "config.yaml"
            meta = {}
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text())
                except Exception:
                    meta = {}
            results = {}
            if results_file.exists():
                try:
                    results = json.loads(results_file.read_text())
                except Exception:
                    results = {}
            EXPERIMENT_DB[p.name] = {
                "id": p.name,
                "path": str(p),
                "metadata": meta,
                "results": results,
                "has_config": config_file.exists(),
            }


_scan_experiments()


def _generate_demo_data():
    """Generate synthetic demo data for visualization when no real experiments exist."""
    n_neurons = [784, 256, 128, 10]
    layers = []
    layer_names = ["Input (784)", "Hidden 1 (256)", "Hidden 2 (128)", "Output (10)"]
    for i, (n, name) in enumerate(zip(n_neurons, layer_names)):
        neurons = []
        for j in range(min(n, 50)):
            neurons.append({
                "id": f"L{i}_N{j}",
                "layer": i,
                "index": j,
                "firing_rate": round(random.uniform(0.01, 0.3), 4),
                "membrane_potential": round(random.uniform(-70, -50), 2),
                "threshold": -55.0,
                "spike_count": random.randint(0, 100),
            })
        layers.append({"name": name, "count": n, "neurons": neurons})

    connections = []
    for i in range(len(n_neurons) - 1):
        for j in range(min(n_neurons[i], 30)):
            for k in range(min(n_neurons[i + 1], 20)):
                w = round(random.uniform(-0.1, 0.5), 4)
                if abs(w) > 0.05:
                    connections.append({
                        "from_layer": i,
                        "from_neuron": j,
                        "to_layer": i + 1,
                        "to_neuron": k,
                        "weight": w,
                        "active": abs(w) > 0.1,
                    })

    spike_trains = []
    for t in range(50):
        step = []
        for i, n in enumerate(n_neurons):
            for j in range(min(n, 30)):
                if random.random() < 0.05:
                    step.append({"layer": i, "neuron": j, "time": t})
        spike_trains.append(step)

    epochs = list(range(1, 21))
    training = {
        "loss": [round(2.5 * math.exp(-0.15 * e) + random.uniform(-0.1, 0.1), 4) for e in epochs],
        "accuracy": [round(min(0.95, 0.1 + 0.85 * (1 - math.exp(-0.2 * e))) + random.uniform(-0.02, 0.02), 4) for e in epochs],
        "val_loss": [round(2.6 * math.exp(-0.13 * e) + random.uniform(-0.15, 0.15), 4) for e in epochs],
        "val_accuracy": [round(min(0.93, 0.08 + 0.85 * (1 - math.exp(-0.18 * e))) + random.uniform(-0.03, 0.03), 4) for e in epochs],
    }

    sparsity_over_time = [round(random.uniform(0.3, 0.7), 4) for _ in range(20)]
    weight_history = [
        {"mean": round(random.uniform(0.1, 0.3), 4), "std": round(random.uniform(0.05, 0.2), 4)}
        for _ in range(20)
    ]

    return {
        "layers": layers,
        "connections": connections,
        "spike_trains": spike_trains,
        "training": training,
        "sparsity_over_time": sparsity_over_time,
        "weight_history": weight_history,
        "config": {
            "neuron_model": "adaptive_lif",
            "tau_mem": 20.0,
            "threshold": 1.0,
            "encoding": "rate",
            "time_steps": 25,
            "plasticity": True,
            "structural_plasticity": True,
            "sparse_connectivity": True,
        },
        "metrics": {
            "accuracy": 0.94,
            "f1": 0.93,
            "spike_rate": 0.07,
            "sparsity": 0.52,
            "params": 236810,
            "flops_millions": 47.3,
        },
    }


@app.get("/", response_class=HTMLResponse)
async def overview(request: Request):
    _scan_experiments()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "experiments": EXPERIMENT_DB,
        "demo": _generate_demo_data(),
    })


@app.get("/network", response_class=HTMLResponse)
async def network_3d(request: Request, exp_id: Optional[str] = None):
    data = _generate_demo_data()
    return templates.TemplateResponse("network3d.html", {
        "request": request,
        "data": json.dumps(data),
        "exp_id": exp_id or "demo",
    })


@app.get("/spikes", response_class=HTMLResponse)
async def spikes_view(request: Request, exp_id: Optional[str] = None):
    data = _generate_demo_data()
    return templates.TemplateResponse("spikes.html", {
        "request": request,
        "data": json.dumps(data),
        "exp_id": exp_id or "demo",
    })


@app.get("/neurons", response_class=HTMLResponse)
async def neurons_view(request: Request, exp_id: Optional[str] = None):
    data = _generate_demo_data()
    return templates.TemplateResponse("neurons.html", {
        "request": request,
        "data": json.dumps(data),
        "exp_id": exp_id or "demo",
    })


@app.get("/synapses", response_class=HTMLResponse)
async def synapses_view(request: Request, exp_id: Optional[str] = None):
    data = _generate_demo_data()
    return templates.TemplateResponse("synapses.html", {
        "request": request,
        "data": json.dumps(data),
        "exp_id": exp_id or "demo",
    })


@app.get("/plasticity", response_class=HTMLResponse)
async def plasticity_view(request: Request, exp_id: Optional[str] = None):
    data = _generate_demo_data()
    return templates.TemplateResponse("plasticity.html", {
        "request": request,
        "data": json.dumps(data),
        "exp_id": exp_id or "demo",
    })


@app.get("/structure", response_class=HTMLResponse)
async def structure_view(request: Request, exp_id: Optional[str] = None):
    data = _generate_demo_data()
    return templates.TemplateResponse("structure.html", {
        "request": request,
        "data": json.dumps(data),
        "exp_id": exp_id or "demo",
    })


@app.get("/experiments", response_class=HTMLResponse)
async def experiments_view(request: Request):
    _scan_experiments()
    return templates.TemplateResponse("experiments.html", {
        "request": request,
        "experiments": EXPERIMENT_DB,
    })


@app.get("/compare", response_class=HTMLResponse)
async def compare_view(request: Request):
    _scan_experiments()
    return templates.TemplateResponse("compare.html", {
        "request": request,
        "experiments": EXPERIMENT_DB,
    })


@app.get("/continual", response_class=HTMLResponse)
async def continual_view(request: Request):
    return templates.TemplateResponse("continual.html", {
        "request": request,
        "data": json.dumps(_generate_demo_data()),
    })


@app.get("/robustness", response_class=HTMLResponse)
async def robustness_view(request: Request):
    return templates.TemplateResponse("robustness.html", {
        "request": request,
        "data": json.dumps(_generate_demo_data()),
    })


@app.get("/config", response_class=HTMLResponse)
async def config_view(request: Request):
    return templates.TemplateResponse("config.html", {
        "request": request,
        "data": json.dumps(_generate_demo_data()["config"]),
    })


@app.get("/about", response_class=HTMLResponse)
async def about_view(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.get("/api/data")
async def api_data():
    return JSONResponse(_generate_demo_data())


@app.get("/api/network")
async def api_network():
    d = _generate_demo_data()
    return JSONResponse({"layers": d["layers"], "connections": d["connections"]})


@app.get("/api/spikes")
async def api_spikes():
    d = _generate_demo_data()
    return JSONResponse({"spike_trains": d["spike_trains"]})


@app.get("/api/training")
async def api_training():
    d = _generate_demo_data()
    return JSONResponse(d["training"])


@app.get("/api/metrics")
async def api_metrics():
    d = _generate_demo_data()
    return JSONResponse(d["metrics"])


@app.get("/api/experiments")
async def api_experiments():
    _scan_experiments()
    return JSONResponse(list(EXPERIMENT_DB.keys()))


@app.get("/api/experiment/{exp_id}")
async def api_experiment(exp_id: str):
    _scan_experiments()
    if exp_id in EXPERIMENT_DB:
        return JSONResponse(EXPERIMENT_DB[exp_id])
    return JSONResponse({"error": "not found"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  BIO-NN 3D Visualization Dashboard")
    print("  http://localhost:8000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
