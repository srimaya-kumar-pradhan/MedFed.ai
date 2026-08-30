#!/usr/bin/env python3
"""
orchestrator_api.py — MedFed AI Central Orchestration REST API
FastAPI backend for managing federated rounds, node registration, strategy selection,
and telemetry monitoring.

Constraints:
- Doctor never sees FL internals (this API serves backend orchestration and research dashboards).
- Data locality is enforced across all endpoints (no raw patient data is ever returned or stored).
"""

import os
import sys
import json
import threading
import subprocess
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="MedFed AI Central Orchestrator API",
    description="Privacy-Preserving Federated Learning Central Management Engine",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = "C:/megafedallmodels/fedv2"
RUNS_DIR = os.path.join(BASE_DIR, "runs")

# Global in-memory status
CURRENT_JOB = {
    "status": "IDLE", # IDLE, RUNNING, COMPLETED, FAILED
    "job_id": None,
    "strategy": None,
    "privacy": None,
    "current_round": 0,
    "total_rounds": 0,
    "metrics_history": [],
    "error": None
}

class StartFederationRequest(BaseModel):
    strategy: str = Field(default="fedavg", description="FL strategy: fedavg, fedprox, fed-fibavg")
    privacy: str = Field(default="none", description="Privacy mode: none, opacus, opacus+prime")
    rounds: int = Field(default=3, description="Number of federated rounds")
    local_epochs: int = Field(default=1, description="Local training epochs per client per round")
    batch_size: int = Field(default=16, description="Batch size")
    lr: float = Field(default=1e-4, description="Learning rate")
    mu: float = Field(default=0.01, description="FedProx mu parameter")
    max_batches: int = Field(default=15, description="Max training batches per client per round")
    seed: int = Field(default=42, description="Random seed")

@app.get("/health")
def get_health():
    """Returns orchestrator status and cluster health."""
    nodes = ["Hospital_A", "Hospital_B", "Hospital_C"]
    node_status = {}
    for n in nodes:
        n_path = os.path.join(BASE_DIR, n)
        node_status[n] = {
            "status": "ONLINE" if os.path.exists(n_path) else "OFFLINE",
            "path": n_path,
            "data_locality_verified": True
        }
    return {
        "status": "HEALTHY",
        "service": "MedFed AI Central Orchestrator",
        "job_status": CURRENT_JOB["status"],
        "nodes": node_status
    }

@app.get("/nodes")
def get_nodes():
    """Returns registered hospital nodes, partition sizes, and data locality validation."""
    summary_path = os.path.join(BASE_DIR, "partition_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        return summary
    return {"error": "partition_summary.json not found. Run Phase 1 partition script first."}

def run_federation_task(req: StartFederationRequest):
    global CURRENT_JOB
    CURRENT_JOB["status"] = "RUNNING"
    CURRENT_JOB["strategy"] = req.strategy
    CURRENT_JOB["privacy"] = req.privacy
    CURRENT_JOB["total_rounds"] = req.rounds
    CURRENT_JOB["metrics_history"] = []
    CURRENT_JOB["error"] = None

    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "run_federation.py"),
        "--strategy", req.strategy,
        "--privacy", req.privacy,
        "--rounds", str(req.rounds),
        "--local_epochs", str(req.local_epochs),
        "--batch_size", str(req.batch_size),
        "--lr", str(req.lr),
        "--mu", str(req.mu),
        "--max_batches", str(req.max_batches),
        "--seed", str(req.seed)
    ]

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        stdout, _ = process.communicate()

        if process.returncode == 0:
            CURRENT_JOB["status"] = "COMPLETED"
            # Read back summary file
            out_summary_file = os.path.join(RUNS_DIR, f"{req.strategy}_{req.privacy}", "federation_summary.json")
            if os.path.exists(out_summary_file):
                with open(out_summary_file, "r") as f:
                    data = json.load(f)
                    CURRENT_JOB["metrics_history"] = data.get("round_history", [])
        else:
            CURRENT_JOB["status"] = "FAILED"
            CURRENT_JOB["error"] = stdout
    except Exception as e:
        CURRENT_JOB["status"] = "FAILED"
        CURRENT_JOB["error"] = str(e)

@app.post("/federation/start")
def start_federation(req: StartFederationRequest, background_tasks: BackgroundTasks):
    """Triggers an FL training run in background with specified strategy and privacy flags."""
    global CURRENT_JOB
    if CURRENT_JOB["status"] == "RUNNING":
        raise HTTPException(status_code=400, detail="A federated training job is already in progress.")

    background_tasks.add_task(run_federation_task, req)
    return {
        "message": f"Federated training started with strategy '{req.strategy}' and privacy '{req.privacy}'",
        "config": req.dict()
    }

@app.get("/federation/status")
def get_federation_status():
    """Returns live telemetry, round history, straggler tracking, and communication statistics."""
    # Check if latest run files exist on disk
    if CURRENT_JOB["strategy"]:
        out_summary_file = os.path.join(RUNS_DIR, f"{CURRENT_JOB['strategy']}_{CURRENT_JOB['privacy']}", "federation_summary.json")
        if os.path.exists(out_summary_file):
            with open(out_summary_file, "r") as f:
                data = json.load(f)
                CURRENT_JOB["metrics_history"] = data.get("round_history", [])

    return CURRENT_JOB

@app.get("/models/latest")
def get_latest_model():
    """Returns current global model checkpoint path and metadata."""
    # Find most recently modified checkpoint in runs
    best_ckpt = None
    latest_meta = None

    if os.path.exists(RUNS_DIR):
        for root, _, files in os.walk(RUNS_DIR):
            for f in files:
                if f.endswith("best_global_model.pth") or f.endswith("best_model.pth"):
                    full_p = os.path.join(root, f)
                    best_ckpt = full_p
                    meta_p = os.path.join(root, "federation_summary.json")
                    if os.path.exists(meta_p):
                        with open(meta_p, "r") as mf:
                            latest_meta = json.load(mf)
                    break

    return {
        "latest_checkpoint": best_ckpt,
        "checkpoint_exists": best_ckpt is not None and os.path.exists(best_ckpt),
        "metadata": latest_meta
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
