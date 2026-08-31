"""
Lightweight async training job manager. No Celery/Redis — we just spawn
a Python subprocess and write structured status to a JSON file that the
API reads back. This is deliberately small for an MVP.

CRITICAL: training is NEVER started automatically. The API endpoint requires
an explicit user action with confirmation, and the job is launched in a
subprocess. The FastAPI event loop is never blocked.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.core import config

# Where training jobs persist their status
JOBS_DIR = config.PROJECT_ROOT / "models" / "metadata" / "training_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


# In-memory snapshot of all jobs (re-read on demand from disk too)
_JOBS_LOCK = threading.RLock()
_JOBS: Dict[str, Dict[str, Any]] = {}


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _persist(job: Dict[str, Any]) -> None:
    _job_path(job["id"]).write_text(json.dumps(job, indent=2))


def list_jobs() -> List[Dict[str, Any]]:
    """Return every persisted job, newest first."""
    jobs = []
    for p in sorted(JOBS_DIR.glob("*.json"), reverse=True):
        try:
            jobs.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return jobs


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    p = _job_path(job_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def create_job(
    strategy: str,
    privacy: str,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    mu: float,
    max_batches: int,
    seed: int,
    hospital_nodes: List[str],
    requested_by: str,
) -> Dict[str, Any]:
    """Allocate a new job record. Does NOT start training — that's start_job()."""
    job_id = f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    job = {
        "id": job_id,
        "status": "pending",  # pending | running | completed | failed | stopped
        "created_at": _now(),
        "started_at": None,
        "ended_at": None,
        "strategy": strategy,
        "privacy": privacy,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "mu": mu,
        "max_batches": max_batches,
        "seed": seed,
        "hospital_nodes": hospital_nodes,
        "requested_by": requested_by,
        "current_round": 0,
        "progress_pct": 0.0,
        "per_round": [],
        "result": None,
        "error": None,
        "log_tail": [],
    }
    with _JOBS_LOCK:
        _JOBS[job_id] = job
        _persist(job)
    return job


def start_job(job_id: str) -> Dict[str, Any]:
    """Spawn the training subprocess. Idempotent: re-starts are not allowed."""
    with _JOBS_LOCK:
        job = get_job(job_id)
        if not job:
            raise KeyError(job_id)
        if job["status"] != "pending":
            raise RuntimeError(f"Job {job_id} is already in status '{job['status']}'")
        job["status"] = "running"
        job["started_at"] = _now()
        _persist(job)

    # Build subprocess command. We invoke the existing run_federation.py
    # and forward the same args. Output is streamed to a per-job log file.
    log_path = JOBS_DIR / f"{job_id}.log"
    output_dir = config.RUNS_DIR / f"{job['strategy']}_{job['privacy']}_{job_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(config.PROJECT_ROOT / "run_federation.py"),
        "--strategy", job["strategy"],
        "--privacy", job["privacy"],
        "--rounds", str(job["rounds"]),
        "--local_epochs", str(job["local_epochs"]),
        "--batch_size", str(job["batch_size"]),
        "--lr", str(job["lr"]),
        "--mu", str(job["mu"]),
        "--max_batches", str(job["max_batches"]),
        "--seed", str(job["seed"]),
        "--output_dir", str(output_dir),
    ]

    log_fh = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(config.PROJECT_ROOT),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    with _JOBS_LOCK:
        job["pid"] = proc.pid
        job["log_path"] = str(log_path)
        job["output_dir"] = str(output_dir)
        _persist(job)

    # Background monitor thread — does not block the API.
    threading.Thread(
        target=_monitor_subprocess,
        args=(job_id, proc, log_path, output_dir),
        daemon=True,
    ).start()
    return job


def stop_job(job_id: str) -> Dict[str, Any]:
    """Send SIGTERM to a running job."""
    with _JOBS_LOCK:
        job = get_job(job_id)
        if not job:
            raise KeyError(job_id)
        pid = job.get("pid")
    if pid:
        try:
            import signal
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
    return get_job(job_id) or {}


def _monitor_subprocess(job_id: str, proc: subprocess.Popen, log_path: Path, output_dir: Path) -> None:
    """Poll the subprocess, harvest log tail, harvest round-by-round metrics."""
    while proc.poll() is None:
        _harvest(job_id, log_path, output_dir)
        time.sleep(2.0)
    rc = proc.returncode
    _harvest(job_id, log_path, output_dir)
    with _JOBS_LOCK:
        job = _JOBS.get(job_id) or get_job(job_id)
        if not job:
            return
        job["ended_at"] = _now()
        if rc == 0:
            job["status"] = "completed"
            summary_path = output_dir / "federation_summary.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text())
                    job["result"] = {
                        "summary_path": str(summary_path),
                        "best_global_macro_f1": summary.get("best_global_macro_f1"),
                        "final_global_macro_f1": summary.get("final_global_macro_f1"),
                        "final_global_roc_auc": summary.get("final_global_roc_auc"),
                        "total_wall_clock_sec": summary.get("total_wall_clock_sec"),
                    }
                except (OSError, json.JSONDecodeError):
                    pass
        else:
            job["status"] = "failed"
            job["error"] = f"Training subprocess exited with code {rc}"
        _persist(job)


def _harvest(job_id: str, log_path: Path, output_dir: Path) -> None:
    """Update progress and round metrics from running summary."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id) or get_job(job_id)
        if not job:
            return
    # Read log tail
    try:
        log_text = log_path.read_text(errors="replace")
        tail = log_text.splitlines()[-30:]
    except OSError:
        tail = []
    # Read federation_summary if it exists (gets updated after each round)
    summary_path = output_dir / "federation_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            history = summary.get("round_history", [])
            with _JOBS_LOCK:
                job["current_round"] = len(history)
                job["progress_pct"] = round(100.0 * len(history) / max(1, job["rounds"]), 1)
                job["per_round"] = [
                    {
                        "round": r.get("round"),
                        "global_macro_f1": r.get("global_macro_f1"),
                        "global_roc_auc": r.get("global_roc_auc"),
                        "round_duration_sec": r.get("round_duration_sec"),
                        "straggler_node": r.get("straggler_node"),
                        "client_f1s": r.get("client_f1s", {}),
                    }
                    for r in history
                ]
        except (OSError, json.JSONDecodeError):
            pass
    with _JOBS_LOCK:
        job["log_tail"] = tail
        _persist(job)
