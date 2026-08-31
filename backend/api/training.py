"""
Training routes. ALL routes require explicit user action and authentication.
The async job manager never starts training on its own.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.core.deps import get_current_user
from backend.jobs import training_jobs
from backend.services import auth_service


router = APIRouter(prefix="/api/training", tags=["training"])


class StartTrainingRequest(BaseModel):
    strategy: str = Field(..., description="fedavg | fedprox | fed-fibavg")
    privacy: str = Field("none", description="none | opacus | opacus+prime")
    rounds: int = Field(3, ge=1, le=50)
    local_epochs: int = Field(1, ge=1, le=20)
    batch_size: int = Field(16, ge=1, le=128)
    lr: float = Field(1e-4, gt=0)
    mu: float = Field(0.01, ge=0)
    max_batches: int = Field(15, ge=1, le=500)
    seed: int = Field(42, ge=0)
    hospital_nodes: List[str] = Field(default_factory=lambda: ["Hospital_A", "Hospital_B", "Hospital_C"])
    confirm: bool = Field(..., description="User must set this to True to confirm GPU/time cost.")


@router.post("/start")
def start_training(
    req: StartTrainingRequest,
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    if not auth_service.has_permission(user, "start_training"):
        raise HTTPException(status_code=403, detail="Role not authorized to start training")
    if not req.confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. Re-submit with confirm=true to acknowledge GPU/time cost.",
        )
    if req.strategy not in ("fedavg", "fedprox", "fed-fibavg"):
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {req.strategy}")

    job = training_jobs.create_job(
        strategy=req.strategy,
        privacy=req.privacy,
        rounds=req.rounds,
        local_epochs=req.local_epochs,
        batch_size=req.batch_size,
        lr=req.lr,
        mu=req.mu,
        max_batches=req.max_batches,
        seed=req.seed,
        hospital_nodes=req.hospital_nodes,
        requested_by=user["username"],
    )
    training_jobs.start_job(job["id"])
    return job


@router.post("/stop")
def stop_training(
    job_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    if not auth_service.has_permission(user, "stop_training"):
        raise HTTPException(status_code=403, detail="Role not authorized to stop training")
    return training_jobs.stop_job(job_id)


@router.get("/status")
def training_status(
    job_id: Optional[str] = None,
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    if not auth_service.has_permission(user, "view_training_runs"):
        raise HTTPException(status_code=403, detail="Role not authorized to view training")
    if job_id:
        job = training_jobs.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job
    return {"jobs": training_jobs.list_jobs()}


@router.get("/runs")
def list_runs(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if not auth_service.has_permission(user, "view_training_runs"):
        raise HTTPException(status_code=403, detail="Role not authorized")
    return {"runs": training_jobs.list_jobs()}


@router.get("/runs/{run_id}")
def get_run(run_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if not auth_service.has_permission(user, "view_training_runs"):
        raise HTTPException(status_code=403, detail="Role not authorized")
    job = training_jobs.get_job(run_id)
    if not job:
        raise HTTPException(status_code=404, detail="Run not found")
    return job
