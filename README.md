# MedFed AI

> **Privacy-Preserving Federated Learning Platform for Medical Diagnostics**
> Prototype validation domain: NIH Chest X-ray (14-class multi-label). Architecture generalizes to Brain MRI without rework.

---

## Overview

MedFed AI is a privacy-first federated learning platform that enables hospitals to collaboratively train medical imaging AI models **without ever sharing raw patient data**. Each hospital node trains locally on its own data; only model parameters cross the network. The platform is built for clinical use with mandatory explainability (Grad-CAM), strict data locality, and a doctor-facing React portal that hides all federated learning internals.

This is a prototype intended for technical evaluation and demonstration. It is not a clinically validated medical device. Clinical decision remains with the qualified healthcare professional.

---

## Architecture

```
React (Vite + TypeScript + Tailwind)
        │   REST + JWT
        ▼
FastAPI (backend/app.py)
        │
        ├── services/model_service.py    (loads model once, caches in memory)
        ├── services/registry_service.py (versioning, deploy, registry JSON)
        ├── services/prediction_service.py
        ├── services/auth_service.py     (JWT, RBAC, demo user DB)
        ├── jobs/training_jobs.py        (async subprocess job manager)
        │
        ▼
Persistent model storage
        models/global/current/model.pth
        models/global/v1/model.pth
        models/global/v2/model.pth
        models/metadata/model_registry.json
        models/metadata/training_jobs/<job_id>.json
```

**Critical startup contract**: backend boot loads the persisted model once into memory, sets it to eval mode, and starts the API. Training is **never** triggered by startup.

**Training flow**: research user explicitly calls `POST /api/training/start` (gated by a `confirm: true` payload). A background subprocess runs `run_federation.py`; its `federation_summary.json` is harvested to a per-job status file. The new model is registered as a new version; deployment is human-approved via `POST /api/models/{version}/deploy`.

---

## Repository layout

```
fedv2/
├── backend/                    # FastAPI application
│   ├── app.py                  # Entry point
│   ├── core/                   # config, deps
│   ├── api/                    # route modules
│   │   ├── auth.py
│   │   ├── health.py
│   │   ├── predict.py
│   │   ├── training.py
│   │   ├── models_registry.py
│   │   └── nodes.py
│   ├── services/               # business logic
│   │   ├── model_service.py
│   │   ├── prediction_service.py
│   │   ├── registry_service.py
│   │   └── auth_service.py
│   └── jobs/                   # async training job manager
│       └── training_jobs.py
│
├── frontend/
│   └── medfed-ui/              # Vite + React + TypeScript
│       ├── src/
│       │   ├── api/client.ts          # REST client
│       │   ├── auth/AuthContext.tsx
│       │   ├── components/            # AppShell, Icon
│       │   ├── pages/                 # Login, Dashboard, Analyze, etc.
│       │   └── App.tsx
│       └── tailwind.config.js
│
├── models/                     # Persistent model storage
│   ├── global/
│   │   ├── current/model.pth
│   │   ├── v1/model.pth
│   │   └── v2/model.pth
│   ├── metadata/
│   │   ├── model_registry.json
│   │   └── training_jobs/<job_id>.json
│   └── checkpoints/
│
├── Hospital_A, Hospital_B, Hospital_C/   # Local federated node datasets
├── runs/                                # Raw training summaries
│
├── model.py, gradcam.py, losses.py, fl_client.py, fl_server.py,
├── run_federation.py, evaluate.py, sample_dataset.py, partition_nodes.py,
├── prime_dp.py, fed_fibavg.py, preprocess.py, train_local.py
│
└── requirements.txt
```

The ML core (`model.py`, `gradcam.py`, `losses.py`, `run_federation.py`, etc.) is preserved unchanged from the previous build. The new backend `services/` thin-wraps these.

---

## Quick start

### 1. Backend (FastAPI)

```bash
cd fedv2
python -m pip install -r requirements.txt
python backend/app.py
# uvicorn running on http://localhost:8000
```

The first call logs:

```
MedFed AI backend starting
Loading persistent current model (no training)...
Model ready: global_v1 | device=cpu | loaded_for=0s
API ready. Frontend can connect now.
```

### 2. Frontend (React)

```bash
cd fedv2/frontend/medfed-ui
npm install
npm run dev
# Vite running on http://localhost:5173
```

### 3. Demo accounts

| Email | Password | Role |
|-------|----------|------|
| `dr.sharma@hospitala.com` | `demo123` | Doctor (Hospital A) |
| `dr.lee@hospitalc.com` | `demo123` | Doctor (Hospital C) |
| `researcher@institution1.com` | `research123` | Researcher (Hospital B) |
| `admin@hospitala.com` | `admin123` | Institution Admin |
| `platform@medfed.ai` | `platform123` | Platform Admin |

### 4. Run a federated training round

Sign in as a researcher or admin → **Federated Training** → configure run → click **Start Training (confirm)** → review warning → **Start Training**.

A new job appears under "Live status" with progress and per-round metrics. On completion, a new version is registered under **Model Registry**. An institution admin can click **Deploy** (with confirm) to switch the live inference model to the new version.

---

## API surface

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/api/health` | public | Liveness |
| POST | `/api/auth/login` | public | Returns JWT |
| GET | `/api/auth/me` | bearer | Current user |
| GET | `/api/model/status` | bearer | Loaded model + registry |
| GET | `/api/model/current` | bearer | Current model record |
| POST | `/api/predict` | bearer | Run inference |
| POST | `/api/explain` | bearer | Inference + Grad-CAM |
| GET | `/api/training/runs` | researcher+ | List runs |
| GET | `/api/training/runs/{id}` | researcher+ | Run details |
| POST | `/api/training/start` | researcher+ | Start federated run |
| POST | `/api/training/stop` | researcher+ | Stop run |
| GET | `/api/models` | researcher+ | Registry |
| GET | `/api/models/{version}` | researcher+ | Version details |
| POST | `/api/models/{version}/deploy` | admin+ | Deploy version |
| GET | `/api/nodes` | researcher+ | Hospital nodes |
| GET | `/api/nodes/{node_id}` | researcher+ | Node details |

---

## How model persistence works

1. A new model is produced by the federated training subprocess into `runs/<strategy>_<privacy>_<job_id>/global_model_round_*.pth` and `best_global_model.pth`.
2. On completion, the registry discovers the new checkpoint and copies it into `models/global/<new_version>/model.pth` (e.g. `v2/model.pth`). All metrics are read directly from the checkpoint's stored metadata — no fabricated values.
3. `current/model.pth` is updated only on explicit **Deploy** action.
4. On backend boot, `model_service._ModelHolder` looks for `models/global/current/model.pth` → `models/global/v1/model.pth` and loads exactly one. If none exists, inference endpoints return 503 with the message *"Model unavailable. Please contact the system administrator."* — never silently trains a replacement.

---

## How training is triggered

Only via `POST /api/training/start` with `{confirm: true}`. The orchestrator:

1. Creates a job record under `models/metadata/training_jobs/<id>.json`.
2. Spawns the training subprocess in the background (Python `subprocess.Popen`).
3. A daemon thread polls the subprocess, harvests per-round metrics from the live `federation_summary.json`, and updates the job file.
4. On success, the new checkpoint is registered as a new version. A user with `deploy_model` permission can then promote it to `current/`.

If the user clicks "Start Training" without setting `confirm: true`, the API returns 400. The UI enforces a two-step confirmation gate.

---

## How inference works

`POST /api/predict` and `POST /api/explain`:

1. FastAPI receives the uploaded image, validates MIME type and minimum size.
2. The already-loaded model runs sigmoid inference on the preprocessed tensor.
3. Top-k predictions and the full 14-class distribution are returned.
4. `/api/explain` additionally runs Grad-CAM and returns the overlay as base64 PNG.

No GPU is required. No file persistence. No new training. The model has its `requires_grad` flag toggled correctly per request to keep the default path locked to inference.

---

## Design system

- **Palette**: near-black ink (`#0A0A0A`–`#1A1A1A`), white paper (`#FFFFFF`/`#FAFAFA`), grays for borders and secondary text. A single clinical teal (`#0F4C5C`) is the only accent, used for primary actions.
- **Typography**: Inter, single family, weight-driven hierarchy.
- **No emojis** anywhere. Status communicated by small text badges (`Completed`, `Training`, `Failed`, etc.) with monochrome-friendly dots.
- **Borders** are 1px solid; no glow or heavy shadows.
- **Spacing** is on an 8px grid.
- **WCAG AA** contrast throughout.

---

## Prototype limitations

- Demo JWT is HMAC-signed with a hard-coded fallback secret. **Replace** `MEDFED_JWT_SECRET` for any real deployment.
- Demo user database is in-memory; no password reset, no MFA, no SSO.
- `run_federation.py` is invoked as a subprocess; job management is deliberately lightweight (no Celery/Redis/Kubernetes).
- Model is a prototype DenseNet121 fine-tuned on a small subset of NIH Chest X-ray — not FDA-cleared, not CE-marked, not clinically validated.
- Data partitions `Hospital_A/B/C` are local filesystem paths in this prototype; in production, each hospital would run its own `fl_client.py` on its own infrastructure.

---

## Brain MRI roadmap (future)

The architecture is parameterized for a config-only swap:

- Replace `DEFAULT_CHEST_XRAY_CLASSES` in `model.py`.
- Update `losses.py` to use `CrossEntropyLoss` (Brain MRI is multi-class, not multi-label).
- Extend `preprocess.py` to support DICOM via `pydicom`.
- Add `"Brain MRI"` to the Study Type dropdown in `AnalyzePage.tsx`.

See `domain_transfer_readiness.md` for the full compatibility note.
