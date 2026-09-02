# MedFed AI

> **Privacy-Preserving Federated Learning Platform for Medical Diagnostics (MEDfed.ai)**
> Initial validation domain: NIH Chest X-ray (14-class multi-label). Architecture generalizes to Brain MRI without rework.

---

## Overview

**MedFed AI** is a privacy-first federated learning platform that enables hospitals to collaboratively train medical imaging AI models **without ever sharing raw patient data**. Each hospital node trains locally on its own data; only model parameters cross the network. The platform is built for clinical use with mandatory explainability (Grad-CAM), strict data locality, and a doctor-facing portal that hides all federated learning internals.

### Key Innovations

- **Fed-FibAvg**: Fibonacci-tier weighted federated aggregation that down-weights straggling nodes without discarding their unique non-IID data
- **Prime-DP Masking**: Additive prime-seeded obfuscation layered atop Opacus's compute-verified differential privacy baseline
- **Data Locality by Construction**: Raw images never leave hospital infrastructure — only model parameters cross the network
- **Mandatory Explainability**: Every prediction ships with confidence score + Grad-CAM visual explanation
- **Doctor-First UX**: Clinical Portal intentionally hides federated learning math from radiologists

---

## Project Structure

```
fedv2/
├── Phase 1: Data Pipeline
│   ├── sample_dataset.py        # 15,600 images sampled from 12 folders
│   ├── partition_nodes.py       # Non-IID partition into Hospital_A/B/C
│   └── preprocess.py            # Per-node validation + EDA reports
│
├── Phase 2: Local Training
│   ├── model.py                 # DenseNet121 (multi-label)
│   ├── losses.py                # Focal Loss + FedProx proximal term
│   ├── train_local.py           # Single-node training loop
│   └── gradcam.py               # Grad-CAM visual explanations
│
├── Phase 3: Federated Orchestration (Baseline FedAvg)
│   ├── fl_client.py             # Flower NumPyClient (data locality)
│   ├── fl_server.py             # Central FedAvg strategy
│   ├── run_federation.py        # Multi-node FL simulation runner
│   └── orchestrator_api.py      # FastAPI REST API
│
├── Phase 4: Innovation Injection
│   ├── fed_fibavg.py            # Fibonacci-tier weighted aggregation
│   └── prime_dp.py              # Prime-DP masking + Opacus baseline
│
├── Phase 5: Comparative Evaluation
│   └── evaluate.py              # 4-arm comparison + dark-theme charts
│
├── Phase 6-7: Clinical Portal + Research/Admin Dashboards
│   └── clinical_portal/
│       ├── clinical_portal.py   # Streamlit doctor-facing app
│       └── clinical_auth.py     # JWT + RBAC
│
├── Phase 8: Domain Transfer Readiness
│   └── domain_transfer_readiness.md
│
├── Hospital_A/                  # Local partitioned dataset
├── Hospital_B/
├── Hospital_C/
├── runs/                        # FL run artifacts (checkpoints, logs)
└── evaluation_results/          # Comparison charts and tables
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- CUDA optional (CPU build tested and supported)
- NIH Chest X-ray dataset (12 image folders + `Data_Entry_2017.csv`)

### Installation
```bash
git clone https://github.com/srimaya-kumar-pradhan/MedFed.ai.git
cd MedFed.ai

# CPU build
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Or GPU build (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Other dependencies
pip install flwr opacus reportlab python-jose streamlit scikit-learn matplotlib seaborn
```

### Run the Pipeline

**Phase 1 — Data Pipeline & Non-IID Partitioning**
```bash
python sample_dataset.py --seed 42
python partition_nodes.py --num_nodes 3 --alpha 0.5 --seed 42
python preprocess.py --node_dir ./Hospital_A --node_name "Hospital A"
python preprocess.py --node_dir ./Hospital_B --node_name "Hospital B"
python preprocess.py --node_dir ./Hospital_C --node_name "Hospital C"
```

**Phase 2 — Local Training Baseline**
```bash
python train_local.py --node_dir ./Hospital_A --epochs 3 --mu 0.0 --seed 42
```

**Phase 3-4 — Federated Training**
```bash
# Baseline FedAvg
python run_federation.py --strategy fedavg --privacy none --rounds 3

# FedProx (proximal regularization)
python run_federation.py --strategy fedprox --privacy none --rounds 3 --mu 0.01

# Fed-FibAvg (Fibonacci-tier weighted aggregation)
python run_federation.py --strategy fed-fibavg --privacy none --rounds 3

# Fed-FibAvg with Prime-DP masking
python run_federation.py --strategy fed-fibavg --privacy "opacus+prime" --rounds 3
```

**Phase 5 — Comparative Evaluation**
```bash
python evaluate.py --epochs 3 --max_batches 10 --seed 42
```

**Phase 6-7 — Clinical Portal (Streamlit)**
```bash
streamlit run clinical_portal/clinical_portal.py
```

Open `http://localhost:8501` in your browser.

**Demo Credentials** (in `clinical_auth.py`):
| Email | Password | Role |
|-------|----------|------|
| dr.sharma@hospitalA.com | demo123 | Clinician (Hospital A) |
| dr.patel@hospitalA.com | admin123 | Admin (Hospital A) |
| researcher@institution1.com | research123 | Researcher (Hospital B) |
| dr.lee@hospitalC.com | demo123 | Clinician (Hospital C) |

---

## Global Constraints (Enforced)

- **Data Locality**: Raw images (PNG/DICOM) never leave their assigned hospital node directory. Only model parameters cross the network.
- **No Black-Box Output**: Every prediction shown to a clinical user ships with confidence + Grad-CAM explanation.
- **Doctor Never Sees FL Internals**: Federated learning, Fed-FibAvg, DP, and aggregation math are backend infrastructure — never surfaced in the Clinical Portal UI.
- **Study-Type Honesty**: All UI copy says "Chest X-ray" — never claims a different study type the current model doesn't support.
- **Reproducibility**: Every script accepts `--seed` and logs it.
- **Dark-Theme Defaults**: All visualizations use `#2c3e50` background family, `#7f8c8d` secondary, dropped top/right spines, percentage annotations on bars.

---

## Phase 5 Results — Comparative Evaluation

| Arm | Macro F1 | ROC-AUC | Time (s) | Notes |
|-----|----------|---------|----------|-------|
| **FedProx** | **0.0765** | 0.5419 | **130.7** | Best F1 + Fastest |
| Centralized | 0.0569 | 0.4785 | 173.3 | Single-node baseline |
| Fed-FibAvg | 0.0247 | 0.5080 | 642.6 | Fibonacci tier weighting |
| FedAvg | 0.0246 | 0.5108 | 595.6 | Standard FedAvg baseline |

> **Note**: These are proof-of-concept numbers on a small subset (max_batches=10-15 per round for responsive evaluation). For production-grade metrics, increase `--max_batches` to full epoch and `--rounds` to 20+.

---

## Architecture Highlights

### Fed-FibAvg Aggregation (Mathematical Formulation)

1. **Composite Node Fitness Score**: $S_i = \frac{Q_i}{\tau_i + \epsilon}$ where $Q_i$ is data quality and $\tau_i$ is local latency
2. **Client Tier Allocation**: Sort nodes by $S_i$ ascending, assign Fibonacci multipliers $\beta_i \in \{1, 2, 3, 5, 8\}$
3. **Normalized Aggregation Weights**: $w_i = \frac{\beta_i \cdot n_i}{\sum_j \beta_j \cdot n_j}$
4. **Global Update**: $\theta_{global}^{(r+1)} = \sum_i w_i \cdot \theta_i^{(r)}$

See `fed_fibavg.py` docstring for the complete mathematical specification.

### Prime-DP Masking

- **Opacus DP base**: Gaussian mechanism with calibrated $(\epsilon, \delta)$ for formal privacy guarantees
- **Prime-number obfuscation layer**: Additive deterministic masks seeded by safe primes
- **Architectural note**: Prime masking is **additive obfuscation**, not a formal DP mechanism. Formal $(\epsilon, \delta)$ claims come from Opacus only.

---

## Domain Transfer to Brain MRI

The architecture is parameterized for a config-only swap to Brain MRI tumor classification. See [`domain_transfer_readiness.md`](domain_transfer_readiness.md) for the full compatibility note. Key change: replace `DEFAULT_CHEST_XRAY_CLASSES` constant and update the loss function from multi-label Focal Loss to multi-class CrossEntropy.

---

## License

This project is a research prototype. Not yet approved for clinical use. AI-generated assistance — final clinical interpretation remains with the qualified healthcare professional.

---

## Authors

**Team Chanakya** — MedFed AI Build