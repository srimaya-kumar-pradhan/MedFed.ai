"""
MedFed AI — FastAPI backend entry point.

CRITICAL startup contract:
  1. FastAPI process starts.
  2. The persistent current model is loaded exactly once, in memory.
  3. Model is set to eval mode.
  4. /api/health returns 200; /api/model/status reflects the loaded model.
  5. NO training, NO aggregation, NO dataset preprocessing occurs.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Make project root importable so we can reuse model.py / gradcam.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.core import config
from backend.api import auth as auth_routes
from backend.api import health as health_routes
from backend.api import predict as predict_routes
from backend.api import training as training_routes
from backend.api import models_registry as models_routes
from backend.api import nodes as nodes_routes
from backend.services.model_service import model_holder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("medfed")


app = FastAPI(
    title="MedFed AI",
    description="Privacy-Preserving Federated Learning Platform for Medical Diagnostics",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────────────────────────────────────
# Wire up API routes
# ────────────────────────────────────────────────────────────────────────────
app.include_router(health_routes.router)
app.include_router(auth_routes.router)
app.include_router(predict_routes.router)
app.include_router(training_routes.router)
app.include_router(models_routes.router)
app.include_router(nodes_routes.router)


# ────────────────────────────────────────────────────────────────────────────
# Optional static hosting for the React build
# ────────────────────────────────────────────────────────────────────────────
_FRONTEND_DIST = PROJECT_ROOT / "frontend" / "medfed-ui" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(_FRONTEND_DIST / "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    def serve_index_root() -> Any:
        return FileResponse(str(_FRONTEND_DIST / "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str) -> Any:
        # Don't shadow API routes
        if full_path.startswith("api/"):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        candidate = _FRONTEND_DIST / full_path
        if candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(_FRONTEND_DIST / "index.html"))


# ────────────────────────────────────────────────────────────────────────────
# Startup: load model exactly once. NO training.
# ────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def on_startup() -> None:
    logger.info("=" * 60)
    logger.info("MedFed AI backend starting")
    logger.info("=" * 60)
    logger.info("Loading persistent current model (no training)...")
    model_holder.initialize()
    status = model_holder.status()
    if status["available"]:
        logger.info(
            "Model ready: %s | device=%s | loaded_for=%ds",
            status["registry"].get("current_version"),
            status["device"],
            status["loaded_for_seconds"],
        )
    else:
        logger.warning(
            "Model is not available. Inference endpoints will return 503. "
            "Reason: %s",
            status["error"],
        )
    logger.info("API ready. Frontend can connect now.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
