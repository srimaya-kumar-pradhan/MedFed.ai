"""
Prediction routes: /api/predict, /api/explain
Read-only against the in-memory model. No training side effects.
"""
from __future__ import annotations

import io
from typing import Annotated, Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image

from backend.core.deps import get_current_user
from backend.services import prediction_service
from backend.services.model_service import model_holder


router = APIRouter(prefix="/api", tags=["prediction"])


def _load_image(upload: UploadFile) -> Image.Image:
    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")
    try:
        data = upload.file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")
    if image.size[0] < 64 or image.size[1] < 64:
        raise HTTPException(status_code=400, detail="Image is too small (min 64x64).")
    return image


@router.post("/predict")
def predict(
    file: Annotated[UploadFile, File(...)],
    top_k: int = 5,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Run AI-assisted prediction. Doctor/researcher scope only."""
    if not model_holder.status()["available"]:
        raise HTTPException(
            status_code=503,
            detail="Model unavailable. Please contact the system administrator.",
        )
    image = _load_image(file)
    result = prediction_service.predict(image, top_k=top_k)
    result["disclaimer"] = (
        "This is an AI-assisted prediction, not a diagnosis. "
        "Clinical decision remains with the qualified healthcare professional."
    )
    result["study_type"] = "chest_xray"
    return result


@router.post("/explain")
def explain(
    file: Annotated[UploadFile, File(...)],
    target_class: Annotated[Optional[str], Form()] = None,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Run prediction + Grad-CAM visualization. Doctor/researcher scope only.

    If `target_class` is omitted, defaults to the highest-probability class.
    The response always includes `explained_class` so the caller knows
    which class the heatmap corresponds to.
    """
    if not model_holder.status()["available"]:
        raise HTTPException(
            status_code=503,
            detail="Model unavailable. Please contact the system administrator.",
        )
    image = _load_image(file)
    result = prediction_service.explain(image, target_class=target_class)
    # `disclaimer` is already populated by prediction_service.explain().
    return result
