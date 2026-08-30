#!/usr/bin/env python3
"""
run_federation.py — MedFed AI Complete Federated Simulation Runner
Coordinates federated learning rounds across hospital nodes (Hospital_A, Hospital_B, Hospital_C).
Supports:
- Strategies: --strategy={fedavg, fedprox, fed-fibavg}
- Privacy: --privacy={none, opacus, opacus+prime}
- Telemetry: tracks global F1, ROC-AUC, per-client metrics, communication volume, stragglers.
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
import torch

from model import build_model, DEFAULT_CHEST_XRAY_CLASSES
from fl_client import HospitalNumPyClient, get_model_parameters, set_model_parameters, compute_payload_size_mb
from losses import MultiLabelFocalLoss, FedProxLossWrapper
from train_local import evaluate, set_seed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [FedRunner] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FedRunner")

def parse_args():
    parser = argparse.ArgumentParser(description="MedFed AI — Federated Simulation Runner")
    parser.add_argument(
        "--strategy",
        type=str,
        default="fedavg",
        choices=["fedavg", "fedprox", "fed-fibavg"],
        help="FL aggregation strategy (default: fedavg)"
    )
    parser.add_argument(
        "--privacy",
        type=str,
        default="none",
        choices=["none", "opacus", "opacus+prime"],
        help="Privacy mechanism layer (default: none)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of federated aggregation rounds (default: 3)"
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=1,
        help="Number of local training epochs per client per round (default: 1)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Local batch size (default: 16)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Local learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.01,
        help="FedProx mu parameter (used when strategy=fedprox)"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="C:/megafedallmodels/fedv2",
        help="Base directory containing Hospital_A, Hospital_B, Hospital_C"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for logs and checkpoints"
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=15,
        help="Batch cap per client epoch for responsive simulation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device (default: cpu)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir or os.path.join(args.base_dir, "runs", f"{args.strategy}_{args.privacy}")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"=== Starting MedFed AI Federated Orchestration ===")
    logger.info(f"Strategy: {args.strategy.upper()} | Privacy: {args.privacy.upper()} | Rounds: {args.rounds} | Seed: {args.seed}")

    # Discover Hospital Nodes
    node_names = ["Hospital_A", "Hospital_B", "Hospital_C"]
    node_dirs = [os.path.join(args.base_dir, n) for n in node_names]
    for nd in node_dirs:
        if not os.path.exists(nd):
            raise FileNotFoundError(f"Hospital node directory not found at: {nd}")

    # Set FedProx mu based on strategy
    effective_mu = args.mu if args.strategy == "fedprox" else 0.0

    # 1. Initialize Hospital Clients (Enforces data locality)
    clients = []
    for n_name, n_dir in zip(node_names, node_dirs):
        client = HospitalNumPyClient(
            node_dir=n_dir,
            node_name=n_name,
            device=args.device,
            lr=args.lr,
            batch_size=args.batch_size,
            local_epochs=args.local_epochs,
            mu=effective_mu,
            max_batches=args.max_batches,
            seed=args.seed
        )
        clients.append(client)

    # 2. Initialize Central Global Model
    global_model = build_model(
        num_classes=len(DEFAULT_CHEST_XRAY_CLASSES),
        pretrained=True,
        device=args.device
    )
    global_params = get_model_parameters(global_model)

    # 3. Setup Innovation Strategies (Fed-FibAvg & Prime DP if requested)
    fed_fib_aggregator = None
    prime_dp_engine = None

    if args.strategy == "fed-fibavg":
        from fed_fibavg import FedFibAvgEngine
        fed_fib_aggregator = FedFibAvgEngine(num_clients=len(clients))

    if "prime" in args.privacy:
        from prime_dp import PrimeDPEngine
        prime_dp_engine = PrimeDPEngine(seed=args.seed)

    round_history = []
    total_training_start = time.time()
    best_global_f1 = -1.0
    cumulative_comm_mb = 0.0

    # 4. Federated Training Loop
    for r in range(1, args.rounds + 1):
        round_start = time.time()
        logger.info(f"\n==================== [FL Round {r}/{args.rounds}] ({args.strategy.upper()}) ====================")

        client_updates = []
        client_sample_counts = []
        client_metrics_list = []
        client_latencies = {}

        # Round broadcast payload size
        broadcast_mb = compute_payload_size_mb(global_params)
        cumulative_comm_mb += broadcast_mb * len(clients)

        # Train on each hospital node
        for client in clients:
            # Client fit with local training
            updated_params, num_samples, metrics = client.fit(
                global_params,
                config={"server_round": r, "local_epochs": args.local_epochs, "lr": args.lr}
            )

            # Apply Privacy Layer if enabled
            if prime_dp_engine is not None:
                updated_params = prime_dp_engine.apply_prime_mask(updated_params, client.node_name, round_num=r)

            client_updates.append(updated_params)
            client_sample_counts.append(num_samples)
            client_metrics_list.append(metrics)
            client_latencies[client.node_name] = metrics["training_time_sec"]
            cumulative_comm_mb += metrics["payload_mb"]

        # Identify Straggler Node
        straggler_node = max(client_latencies.items(), key=lambda x: x[1])[0]
        straggler_time = client_latencies[straggler_node]

        # 5. Model Aggregation
        if args.strategy == "fed-fibavg" and fed_fib_aggregator is not None:
            # Fed-FibAvg Fibonacci-weighted tier aggregation
            aggregated_params = fed_fib_aggregator.aggregate(
                client_updates=client_updates,
                client_samples=client_sample_counts,
                client_latencies=[m["training_time_sec"] for m in client_metrics_list],
                server_round=r
            )
        else:
            # Standard FedAvg / FedProx weighted average by sample count
            total_samples = sum(client_sample_counts)
            aggregated_params = []
            for param_idx in range(len(client_updates[0])):
                param_shape = client_updates[0][param_idx].shape
                weighted_sum = np.zeros(param_shape, dtype=np.float32)
                for client_idx in range(len(clients)):
                    w = client_sample_counts[client_idx] / total_samples
                    weighted_sum += w * client_updates[client_idx][param_idx]
                aggregated_params.append(weighted_sum)

        # If Prime DP was applied, de-mask aggregated parameters
        if prime_dp_engine is not None:
            aggregated_params = prime_dp_engine.remove_prime_mask(aggregated_params, round_num=r)

        global_params = aggregated_params
        set_model_parameters(global_model, global_params)

        # 6. Global Model Evaluation across all nodes' validation sets
        eval_losses = []
        eval_f1s = []
        eval_aucs = []
        total_eval_samples = 0

        for client in clients:
            loss, num_examples, m = client.evaluate(global_params, config={})
            eval_losses.append(loss * num_examples)
            eval_f1s.append(m["val_macro_f1"] * num_examples)
            eval_aucs.append(m["val_roc_auc"] * num_examples)
            total_eval_samples += num_examples

        global_loss = sum(eval_losses) / max(1, total_eval_samples)
        global_macro_f1 = sum(eval_f1s) / max(1, total_eval_samples)
        global_roc_auc = sum(eval_aucs) / max(1, total_eval_samples)

        round_duration = time.time() - round_start

        round_record = {
            "round": r,
            "strategy": args.strategy,
            "privacy": args.privacy,
            "global_loss": float(global_loss),
            "global_macro_f1": float(global_macro_f1),
            "global_roc_auc": float(global_roc_auc),
            "round_duration_sec": float(round_duration),
            "cumulative_comm_mb": float(cumulative_comm_mb),
            "straggler_node": straggler_node,
            "straggler_latency_sec": float(straggler_time),
            "client_f1s": {m["node_name"]: m["val_macro_f1"] for m in client_metrics_list},
            "client_latencies": client_latencies
        }
        round_history.append(round_record)

        logger.info(f"--- [Round {r} Evaluation Results] ---")
        logger.info(f"Global Macro F1: {global_macro_f1:.4f} | Global ROC-AUC: {global_roc_auc:.4f} | Global Loss: {global_loss:.4f}")
        logger.info(f"Straggler Node: {straggler_node} ({straggler_time:.2f}s) | Cumulative Comms: {cumulative_comm_mb:.2f} MB")

        # Save Checkpoint
        ckpt_path = os.path.join(output_dir, f"global_model_round_{r}.pth")
        torch.save({
            "round": r,
            "strategy": args.strategy,
            "privacy": args.privacy,
            "global_macro_f1": global_macro_f1,
            "global_roc_auc": global_roc_auc,
            "model_state_dict": global_model.state_dict()
        }, ckpt_path)

        if global_macro_f1 > best_global_f1:
            best_global_f1 = global_macro_f1
            best_ckpt = os.path.join(output_dir, "best_global_model.pth")
            torch.save({
                "round": r,
                "strategy": args.strategy,
                "privacy": args.privacy,
                "global_macro_f1": best_global_f1,
                "global_roc_auc": global_roc_auc,
                "model_state_dict": global_model.state_dict()
            }, best_ckpt)

    total_wall_clock = time.time() - total_training_start

    # Final Summary Artifacts
    summary = {
        "strategy": args.strategy,
        "privacy": args.privacy,
        "num_rounds": args.rounds,
        "total_wall_clock_sec": float(total_wall_clock),
        "cumulative_comm_mb": float(cumulative_comm_mb),
        "best_global_macro_f1": float(best_global_f1),
        "final_global_macro_f1": float(global_macro_f1),
        "final_global_roc_auc": float(global_roc_auc),
        "round_history": round_history
    }

    summary_file = os.path.join(output_dir, "federation_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n=== Federated Run Complete! ===")
    logger.info(f"Best Global Macro F1: {best_global_f1:.4f} | Total Time: {total_wall_clock:.1f}s | Summary: {summary_file}")
    return summary

if __name__ == "__main__":
    main()
