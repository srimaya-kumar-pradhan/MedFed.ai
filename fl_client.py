#!/usr/bin/env python3
"""
fl_client.py — MedFed AI Flower Federated Learning Client
Implements NumPyClient for isolated hospital nodes.

Constraints:
- Data Locality: Client accesses only its local hospital node directory.
- Model parameters are the only information serialized and transmitted across the network.
- Telemetry: Returns training duration, communication payload size, and local evaluation metrics.
"""

import os
import sys
import time
import argparse
import logging
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import flwr as fl

from model import build_model, DEFAULT_CHEST_XRAY_CLASSES
from losses import MultiLabelFocalLoss, FedProxLossWrapper
from train_local import LocalChestXrayDataset, get_transforms, evaluate, set_seed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def get_model_parameters(model):
    """Extract model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_model_parameters(model, parameters):
    """Load NumPy parameters into PyTorch model state_dict."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def compute_payload_size_mb(parameters):
    """Calculates total payload size in Megabytes for communication accounting."""
    total_bytes = sum(p.nbytes for p in parameters)
    return total_bytes / (1024 * 1024)

class HospitalNumPyClient(fl.client.NumPyClient):
    """
    Flower NumPyClient representing a single healthcare institution.
    """
    def __init__(self, node_dir, node_name=None, device="cpu", lr=1e-4,
                 batch_size=16, local_epochs=1, mu=0.0, max_batches=None, seed=42):
        self.node_dir = os.path.abspath(node_dir)
        self.node_name = node_name or os.path.basename(self.node_dir)
        self.logger = logging.getLogger(f"Client-{self.node_name}")
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.mu = mu
        self.max_batches = max_batches
        self.seed = seed

        set_seed(self.seed)

        # 1. Initialize local model
        self.model = build_model(
            num_classes=len(DEFAULT_CHEST_XRAY_CLASSES),
            pretrained=True,
            device=self.device
        )

        # 2. Setup Loss & FedProx Wrapper
        self.base_loss_fn = MultiLabelFocalLoss(alpha=0.25, gamma=2.0)
        self.loss_wrapper = FedProxLossWrapper(self.base_loss_fn, mu=self.mu)

        # 3. Load strictly local datasets (Data Locality Hard Constraint)
        train_csv = os.path.join(self.node_dir, "train.csv")
        val_csv = os.path.join(self.node_dir, "val.csv")

        if not os.path.exists(train_csv) or not os.path.exists(val_csv):
            raise FileNotFoundError(f"Local CSVs not found in {self.node_dir}")

        train_transform, val_transform = get_transforms()
        self.train_dataset = LocalChestXrayDataset(train_csv, transform=train_transform)
        self.val_dataset = LocalChestXrayDataset(val_csv, transform=val_transform)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

        self.logger.info(
            f"Initialized {self.node_name} | Local Train Samples: {len(self.train_dataset):,} | "
            f"Val Samples: {len(self.val_dataset):,} | Device: {self.device}"
        )

    def get_parameters(self, config=None):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        """
        Local training round triggered by the central orchestrator.
        """
        # Step 1: Update model with global parameters
        set_model_parameters(self.model, parameters)

        # Update FedProx global weights snapshot if mu > 0
        if self.mu > 0.0:
            self.loss_wrapper.set_global_weights(self.model)

        # Parse round configs
        server_round = config.get("server_round", 1)
        epochs = config.get("local_epochs", self.local_epochs)
        lr = config.get("lr", self.lr)

        self.logger.info(f"[Round {server_round}] Starting local training ({epochs} epoch(s))...")
        start_time = time.time()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        self.model.train()

        running_loss = 0.0
        batches_done = 0

        for epoch in range(epochs):
            for batch_idx, (images, targets, _) in enumerate(self.train_loader):
                if self.max_batches and batch_idx >= self.max_batches:
                    break

                images = images.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                logits = self.model(images)
                loss, _, _ = self.loss_wrapper(logits, targets, model=self.model)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item()
                batches_done += 1

        duration = time.time() - start_time
        avg_train_loss = running_loss / max(1, batches_done)
        updated_params = get_model_parameters(self.model)
        payload_size_mb = compute_payload_size_mb(updated_params)

        # Evaluate on local validation set
        val_metrics = evaluate(
            self.model,
            self.val_loader,
            self.loss_wrapper,
            self.device,
            DEFAULT_CHEST_XRAY_CLASSES,
            max_batches=10
        )

        metrics = {
            "node_name": self.node_name,
            "train_loss": float(avg_train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_roc_auc": float(val_metrics["macro_roc_auc"]),
            "training_time_sec": float(duration),
            "payload_mb": float(payload_size_mb),
            "num_samples": len(self.train_dataset)
        }

        self.logger.info(
            f"[Round {server_round}] Completed in {duration:.2f}s | "
            f"Train Loss: {avg_train_loss:.4f} | Val F1: {val_metrics['macro_f1']:.4f} | "
            f"Payload: {payload_size_mb:.2f} MB"
        )

        return updated_params, len(self.train_dataset), metrics

    def evaluate(self, parameters, config):
        """
        Evaluate global model parameters on local validation set.
        """
        set_model_parameters(self.model, parameters)
        val_metrics = evaluate(
            self.model,
            self.val_loader,
            self.loss_wrapper,
            self.device,
            DEFAULT_CHEST_XRAY_CLASSES,
            max_batches=self.max_batches
        )

        loss = float(val_metrics["loss"])
        num_examples = len(self.val_dataset)

        metrics = {
            "node_name": self.node_name,
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_micro_f1": float(val_metrics["micro_f1"]),
            "val_roc_auc": float(val_metrics["macro_roc_auc"]),
            "val_precision": float(val_metrics["macro_precision"]),
            "val_recall": float(val_metrics["macro_recall"])
        }

        return loss, num_examples, metrics

def start_client(server_address, node_dir, node_name=None, device="cpu", mu=0.0):
    """
    Connects client to running Flower server.
    """
    client = HospitalNumPyClient(node_dir=node_dir, node_name=node_name, device=device, mu=mu)
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedFed AI — Flower Hospital Client")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--node_dir", type=str, required=True)
    parser.add_argument("--node_name", type=str, default=None)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    start_client(
        server_address=args.server_address,
        node_dir=args.node_dir,
        node_name=args.node_name,
        device=args.device,
        mu=args.mu
    )
