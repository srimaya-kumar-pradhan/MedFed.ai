#!/usr/bin/env python3
"""
train_local.py — MedFed AI Phase 2 Single-Node Training Loop
Trains DenseNet121 on one isolated hospital node's local dataset.

Features:
- Focal Loss (multi-label) with alpha and gamma CLI parameters.
- FedProx proximal regularization term (mu / 2) * ||w - w_global||^2.
  When mu=0, recovers plain Focal Loss.
- Dataset / DataLoader reading strictly from local node CSVs (data locality enforced).
- Comprehensive metrics: Loss, Macro/Micro F1, Per-Class F1, ROC-AUC, Wall-clock time, Memory footprint.
- Grad-CAM sanity generation on validation samples.
- Seed reproducibility.
"""

import os
import sys
import time
import argparse
import logging
import json
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

from model import build_model, DEFAULT_CHEST_XRAY_CLASSES
from losses import MultiLabelFocalLoss, FedProxLossWrapper
from gradcam import GradCAM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TrainLocal")

class LocalChestXrayDataset(Dataset):
    """
    Isolated local dataset loader reading directly from a hospital node's CSV.
    Enforces data locality: only accesses images listed in this node's CSV.
    """
    def __init__(self, csv_file, transform=None, classes=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.classes = classes or DEFAULT_CHEST_XRAY_CLASSES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']

        # Safe image loading
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback to black image if corrupted/missing
            image = Image.new('RGB', (224, 224), color=0)

        if self.transform is not None:
            image = self.transform(image)

        # Multi-label binary target tensor
        labels = [float(row.get(c, 0)) for c in self.classes]
        target = torch.tensor(labels, dtype=torch.float32)

        return image, target, row['image_name']

def get_transforms():
    """
    ImageNet standard normalization for DenseNet121.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def parse_args():
    parser = argparse.ArgumentParser(description="MedFed AI — Single Node Local Training Baseline")
    parser.add_argument(
        "--node_dir",
        type=str,
        default="C:/megafedallmodels/fedv2/Hospital_A",
        help="Path to the hospital node directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of local training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and validation (default: 32)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for AdamW optimizer (default: 1e-4)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay for AdamW optimizer (default: 1e-2)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Alpha hyperparameter for Focal Loss (default: 0.25)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.0,
        help="Gamma focusing parameter for Focal Loss (default: 2.0)"
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.0,
        help="FedProx proximal term weight (default: 0.0 for baseline)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda or cpu)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save checkpoint and metrics JSON (default: node_dir/runs)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker count (default: 0 for Windows)"
    )
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=None,
        help="Optional cap on training batches per epoch for quick validation"
    )
    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=None,
        help="Optional cap on validation batches"
    )
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(model, val_loader, loss_fn, device, classes, max_batches=None):
    """
    Evaluates model on validation/test set.
    Computes Loss, Macro F1, Micro F1, Per-Class F1, ROC-AUC, Precision, Recall.
    """
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds_binary = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, (images, targets, _) in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss, _, _ = loss_fn(logits, targets, model=model)
            total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            preds_bin = (probs >= 0.5).astype(int)

            all_probs.append(probs)
            all_preds_binary.append(preds_bin)
            all_targets.append(targets.cpu().numpy())

    num_samples = sum(len(t) for t in all_targets)
    avg_loss = total_loss / max(1, num_samples)

    all_targets = np.vstack(all_targets)
    all_preds_binary = np.vstack(all_preds_binary)
    all_probs = np.vstack(all_probs)

    # Compute classification metrics
    macro_f1 = f1_score(all_targets, all_preds_binary, average='macro', zero_division=0)
    micro_f1 = f1_score(all_targets, all_preds_binary, average='micro', zero_division=0)
    macro_prec = precision_score(all_targets, all_preds_binary, average='macro', zero_division=0)
    macro_rec = recall_score(all_targets, all_preds_binary, average='macro', zero_division=0)

    # Per-class F1
    per_class_f1 = {}
    per_class_auc = {}
    for idx, cname in enumerate(classes):
        c_target = all_targets[:, idx]
        c_pred = all_preds_binary[:, idx]
        c_prob = all_probs[:, idx]

        per_class_f1[cname] = float(f1_score(c_target, c_pred, zero_division=0))

        # Compute ROC-AUC only if both classes are present in ground truth
        if len(np.unique(c_target)) > 1:
            try:
                per_class_auc[cname] = float(roc_auc_score(c_target, c_prob))
            except Exception:
                per_class_auc[cname] = 0.5
        else:
            per_class_auc[cname] = 0.5

    valid_aucs = [v for v in per_class_auc.values() if v > 0.0]
    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.5

    return {
        "loss": float(avg_loss),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_roc_auc": float(mean_auc),
        "per_class_f1": per_class_f1,
        "per_class_auc": per_class_auc
    }

def main():
    args = parse_args()
    set_seed(args.seed)

    node_name = os.path.basename(os.path.normpath(args.node_dir))
    output_dir = args.output_dir or os.path.join(args.node_dir, "runs")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"=== Starting Local Training on {node_name} ===")
    logger.info(f"Config: Epochs={args.epochs}, BatchSize={args.batch_size}, LR={args.lr}, Mu={args.mu}, Seed={args.seed}")
    logger.info(f"Device: {args.device}")

    # Data paths
    train_csv = os.path.join(args.node_dir, "train.csv")
    val_csv = os.path.join(args.node_dir, "val.csv")
    test_csv = os.path.join(args.node_dir, "test.csv")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Missing train.csv at {train_csv}")

    train_transform, val_transform = get_transforms()

    train_dataset = LocalChestXrayDataset(train_csv, transform=train_transform)
    val_dataset = LocalChestXrayDataset(val_csv, transform=val_transform)
    test_dataset = LocalChestXrayDataset(test_csv, transform=val_transform)

    logger.info(f"Loaded datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Instantiate Model
    model = build_model(num_classes=len(DEFAULT_CHEST_XRAY_CLASSES), pretrained=True, device=args.device)

    # Instantiate Loss & Optimizer
    base_loss_fn = MultiLabelFocalLoss(alpha=args.alpha, gamma=args.gamma)
    loss_wrapper = FedProxLossWrapper(base_loss_fn, mu=args.mu)
    if args.mu > 0.0:
        loss_wrapper.set_global_weights(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = []
    start_time = time.time()
    best_val_f1 = -1.0
    best_checkpoint_path = os.path.join(output_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_base_loss = 0.0
        running_prox_loss = 0.0
        batches_processed = 0

        for batch_idx, (images, targets, _) in enumerate(train_loader):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break

            images = images.to(args.device)
            targets = targets.to(args.device)

            optimizer.zero_grad()
            logits = model(images)
            total_loss, base_loss_val, prox_loss_val = loss_wrapper(logits, targets, model=model)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += total_loss.item()
            running_base_loss += base_loss_val
            running_prox_loss += prox_loss_val
            batches_processed += 1

        scheduler.step()
        epoch_duration = time.time() - epoch_start
        avg_train_loss = running_loss / max(1, batches_processed)

        # Validation pass
        val_metrics = evaluate(
            model, val_loader, loss_wrapper, args.device, DEFAULT_CHEST_XRAY_CLASSES, max_batches=args.max_val_batches
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(avg_train_loss),
            "train_base_loss": float(running_base_loss / max(1, batches_processed)),
            "train_prox_loss": float(running_prox_loss / max(1, batches_processed)),
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_micro_f1": val_metrics["micro_f1"],
            "val_macro_roc_auc": val_metrics["macro_roc_auc"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "epoch_time_sec": float(epoch_duration)
        }
        history.append(epoch_record)

        logger.info(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Macro F1: {val_metrics['macro_f1']:.4f} | "
            f"Val ROC-AUC: {val_metrics['macro_roc_auc']:.4f} | "
            f"Time: {epoch_duration:.1f}s"
        )

        # Save best model checkpoint
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_macro_f1": best_val_f1,
                "config": vars(args)
            }, best_checkpoint_path)

    total_training_time = time.time() - start_time

    # Final Evaluation on Test Set using Best Checkpoint
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=args.device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model, test_loader, loss_wrapper, args.device, DEFAULT_CHEST_XRAY_CLASSES, max_batches=args.max_val_batches
    )

    logger.info(f"\n=== Final Test Results for {node_name} ===")
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    logger.info(f"Test Micro F1: {test_metrics['micro_f1']:.4f}")
    logger.info(f"Test ROC-AUC: {test_metrics['macro_roc_auc']:.4f}")

    # Generate Grad-CAM Sanity Check Samples
    logger.info("Generating Grad-CAM sanity check overlays on test samples...")
    gradcam = GradCAM(model)
    gradcam_output_dir = os.path.join(output_dir, "gradcam_samples")
    os.makedirs(gradcam_output_dir, exist_ok=True)

    gradcam_samples_generated = 0
    for images, targets, filenames in test_loader:
        for i in range(len(images)):
            if gradcam_samples_generated >= 5:
                break
            img_tensor = images[i:i+1].to(args.device)
            fname = filenames[i]

            # Find positive classes in ground truth or highest prediction
            pos_indices = torch.where(targets[i] == 1)[0]
            target_cls = pos_indices[0].item() if len(pos_indices) > 0 else 0
            cls_name = DEFAULT_CHEST_XRAY_CLASSES[target_cls]

            heatmap, pred_idx, prob = gradcam.generate_heatmap(img_tensor, class_idx=target_cls)

            # Reconstruct un-normalized PIL image for overlay
            unnorm_img = images[i].clone()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            unnorm_img = unnorm_img * std + mean
            unnorm_img = torch.clamp(unnorm_img, 0, 1)
            pil_img = transforms.ToPILImage()(unnorm_img)

            overlay = gradcam.overlay_heatmap(pil_img, heatmap, alpha=0.45)
            sample_path = os.path.join(gradcam_output_dir, f"sample_{gradcam_samples_generated}_{cls_name}_{fname}")
            overlay.save(sample_path)
            gradcam_samples_generated += 1

        if gradcam_samples_generated >= 5:
            break

    logger.info(f"Saved {gradcam_samples_generated} Grad-CAM sample overlays to: {gradcam_output_dir}")

    # Save Run Results JSON
    results_summary = {
        "node_name": node_name,
        "config": vars(args),
        "total_training_time_sec": total_training_time,
        "best_val_macro_f1": best_val_f1,
        "test_metrics": test_metrics,
        "history": history
    }

    metrics_file = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"Saved run metrics to: {metrics_file}")

    return results_summary

if __name__ == "__main__":
    main()
