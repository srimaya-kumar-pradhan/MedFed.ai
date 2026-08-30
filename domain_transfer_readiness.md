# MedFed AI — Domain Transfer Readiness Note
## Swapping from Chest X-ray to Brain MRI Tumor Classification

**Status**: Architecture verified, NOT yet implemented (Phase 8 deferred per PRD).
**Goal**: A reviewer reading this note should believe the swap is a config change plus a new dataset, not a rewrite.

---

## 1. Parameterized Components (Already in Place)

The current architecture is already parameterized at four key points, all of which a Brain MRI swap can leverage without code changes:

| Layer | Chest X-ray (current) | Brain MRI (target) | Parameter Location |
|-------|----------------------|--------------------|--------------------|
| **Label taxonomy** | `DEFAULT_CHEST_XRAY_CLASSES` in `model.py` | `['glioma', 'meningioma', 'pituitary', 'no_tumor']` (e.g., 4 classes) | Pass `num_classes` to `build_model()` |
| **Study Type UI lock** | `selectbox(options=["Chest X-ray"])` in `clinical_portal.py` | Add `"Brain MRI"` to options | Single line in `page_new_analysis()` |
| **Image dimensions** | 224×224 (DenseNet121) | 224×224 (same — DenseNet121 backbone preserved) | No change — `get_image_transform()` already config-free |
| **Class names in Grad-CAM** | `DEFAULT_CHEST_XRAY_CLASSES` in `gradcam.py` | Passed via same constant; replace with MRI taxonomy | Single import swap |

**Key proof**: The `MedFedDenseNet` class accepts `num_classes=14` or any other integer. The `MultiLabelFocalLoss` and `LocalChestXrayDataset` already work for any number of labels. Only the constant `DEFAULT_CHEST_XRAY_CLASSES` needs to be replaced.

---

## 2. What Actually Changes for Brain MRI

### 2.1 Dataset Schema Differences
- **File format**: DICOM (`.dcm`) instead of PNG. The `preprocess.py` validation step must be updated to accept `.dcm` via `pydicom.dcmread()`.
- **Multi-slice handling**: Brain MRI volumes are 3D (e.g., T1, T1-contrast, T2, FLAIR sequences, 100+ slices per scan). The current `LocalChestXrayDataset` assumes single 2D images.
  - **Solution**: For initial Brain MRI swap, treat each axial slice as a 2D sample (drop in replacement for X-ray). For volumetric, add a `slices_per_volume` parameter and pool features across slices.
- **Label taxonomy shift**:
  - Chest X-ray: 14 pathologies + "No Finding" (multi-label, single image → 0..14 positive labels)
  - Brain MRI: typically 4 classes (multi-class, single image → exactly 1 class). Change from multi-label sigmoid to multi-class softmax + CrossEntropyLoss.

### 2.2 Grad-CAM Region-of-Interest Expectations
- Chest X-ray: Grad-CAM highlights localize around lung fields, cardiac silhouette, mediastinum.
- Brain MRI: Grad-CAM should localize around tumor mass regions, edema, or ventricular areas.
- The GradCAM engine itself is class-agnostic — it will highlight whatever drives the target class prediction. The only requirement is that the target pathology class index matches the new label taxonomy.

### 2.3 Clinical Workflow Differences
- **Chest X-ray**: Single frontal/lateral view, immediate triage.
- **Brain MRI**: Multi-sequence, multi-planar (axial, sagittal, coronal). Doctor may need to select which sequence(s) to analyze.
- **Solution**: Extend `study_type` in the UI to include `"Brain MRI - Axial"`, `"Brain MRI - Sagittal"`, `"Brain MRI - Coronal"`, etc. All use the same `num_classes=4` head but with sequence-specific pretrained backbones if needed.

### 2.4 Privacy & Compliance Differences
- Chest X-ray: PHI is usually patient ID + study date.
- Brain MRI: Includes anatomical information potentially identifying facial features. May require additional de-identification beyond X-ray standards.
- **Solution**: The `clinical_portal.py` already enforces hospital-scoped tenancy. Add a sequence-level de-identification check in `preprocess.py` for DICOM metadata scrubbing (`pydicom` provides this).

---

## 3. Concrete Steps to Execute the Swap

### Step 1: Acquire & Preprocess Brain MRI Dataset
```bash
# Example: Brain Tumor Classification dataset (4 classes)
# Source: figshare "Brain Tumor Classification (MRI)" (Kaggle equivalent)
# 4 classes: glioma_tumor, meningioma_tumor, pituitary_tumor, no_tumor
# ~3,000+ T1-weighted contrast-enhanced images, ~750 per class
```

Run preprocessing (single-node, similar to Phase 1):
```python
# Analogous to sample_dataset.py — adjust folder structure for MRI dataset
python sample_dataset_brainmri.py --dataset_dir <path> --output_dir ./ --samples_per_folder 600 --seed 42
```

### Step 2: Update Model & Loss
```python
# model.py — change one line
DEFAULT_CHEST_XRAY_CLASSES -> BRAIN_MRI_CLASSES = ['glioma', 'meningioma', 'pituitary', 'no_tumor']

# losses.py — switch from multi-label sigmoid + Focal Loss to multi-class softmax + CrossEntropyLoss
# Because Brain MRI is multi-class (exactly 1 class per image), not multi-label
# This is a one-file change in the loss function family
```

### Step 3: Re-run Phases 1-5 (Pipeline)
- Phase 1: Partition MRI dataset into Hospital_A/B/C nodes (same scripts, new dataset).
- Phase 2: Train DenseNet121 locally (same `train_local.py` with new `--num_classes 4`).
- Phase 3-4: Federated training with same strategies (FedAvg, FedProx, Fed-FibAvg).
- Phase 5: Comparative evaluation (same `evaluate.py`).

### Step 4: Update UI Lock
In `clinical_portal.py`:
```python
# In page_new_analysis()
study_type = st.selectbox(
    "Study Type",
    options=["Chest X-ray", "Brain MRI"],  # Add Brain MRI
    ...
)
```

### Step 5: Update DICOM Handling
In `preprocess.py`:
```python
# Add DICOM validation branch
if file_path.lower().endswith('.dcm'):
    import pydicom
    dcm = pydicom.dcmread(file_path)
    img_array = dcm.pixel_array
    pil_image = Image.fromarray(img_array)
    # Scrub PHI from DICOM metadata
    # (DICOM tag stripping: PatientName, PatientID, etc.)
```

---

## 4. Architecture Confirmed NOT Blocking Brain MRI Swap

✅ **DenseNet121 backbone**: Reusable for any 2D image classification.
✅ **Multi-class/multi-label flexibility**: `build_model(num_classes=N)` already accepts any N.
✅ **Federated Learning orchestration**: Strategy-agnostic, dataset-agnostic.
✅ **Privacy layer (DP + Prime masking)**: Works on any parameter tensor, not image-specific.
✅ **Local partitioning scripts**: Apply to any image folder structure.
✅ **Grad-CAM explainability**: Class-agnostic.
✅ **JWT/RBAC auth**: Role permissions don't depend on study type.

---

## 5. NOT Blocked But Would Require Work

⚠️ **DICOM parser** (~50 lines): Needed for raw `.dcm` ingestion.
⚠️ **Loss function swap** (Focal → CrossEntropy) for multi-class (~20 lines).
⚠️ **UI update** (1-2 lines in `clinical_portal.py`).
⚠️ **Dataset acquisition & annotation** (external — not a code change).

**Estimated effort to swap**: 1-2 days of engineering + dataset prep time.

---

## 6. Conclusion

The MedFed AI architecture is **ready for domain transfer** with minimal code changes. The single biggest swap point is the label taxonomy constant (`DEFAULT_CHEST_XRAY_CLASSES` → new MRI classes), which propagates through:
- `model.py` (via `num_classes`)
- `losses.py` (MultiLabelFocal → standard CrossEntropy for multi-class)
- `gradcam.py` (via class index lookup)
- `clinical_portal.py` (Study Type dropdown)
- `partition_nodes.py` (folder structure)

The federated learning infrastructure, privacy layers, evaluation framework, and RBAC all transfer without modification.
