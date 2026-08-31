"""Backend core configuration and paths."""
import os
from pathlib import Path

# Project root is the parent of the backend/ directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
ML_CORE_DIR = PROJECT_ROOT  # model.py, gradcam.py live at root

# Persistent model directories
MODELS_GLOBAL_DIR = PROJECT_ROOT / "models" / "global"
MODELS_CURRENT_DIR = MODELS_GLOBAL_DIR / "current"
MODELS_METADATA_DIR = PROJECT_ROOT / "models" / "metadata"
MODELS_CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints"

REGISTRY_PATH = MODELS_METADATA_DIR / "model_registry.json"

# Hospital node partitions
HOSPITAL_DIRS = {
    "Hospital_A": PROJECT_ROOT / "Hospital_A",
    "Hospital_B": PROJECT_ROOT / "Hospital_B",
    "Hospital_C": PROJECT_ROOT / "Hospital_C",
}

# Run output (training summaries)
RUNS_DIR = PROJECT_ROOT / "runs"

# CORS
CORS_ORIGINS = [
    "http://localhost:5173",   # Vite dev server
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Image preprocessing defaults
INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Pathology labels (multi-label)
PATHOLOGY_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]
NUM_CLASSES = len(PATHOLOGY_LABELS)
