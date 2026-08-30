#!/usr/bin/env python3
"""
fl_server.py — MedFed AI Central Federated Learning Orchestrator & Server
Manages Flower aggregation strategies, telemetry tracking, checkpointing, and security stubs.

Global Constraints:
- Baseline: Standard unmodified flwr FedAvg strategy as working reference.
- Telemetry: Global F1/AUC, per-client local F1, comms payload size (MB), wall-clock time, straggler tracking.
- Security: mTLS configuration placeholder clearly documented.
"""

import os
import sys
import time
import argparse
import logging
import json
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)
from flwr.server.strategy import FedAvg

from model import build_model, DEFAULT_CHEST_XRAY_CLASSES
from fl_client import get_model_parameters, set_model_parameters

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [FL-Server] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FL-Server")

# ==============================================================================
# SECURE TRANSPORT STUB (mTLS Placeholder)
# ==============================================================================
def get_server_certificates(ca_cert_path=None, server_cert_path=None, server_key_path=None):
    """
    # TODO: production mTLS
    # In production healthcare deployment, Mutual TLS (mTLS) with institutional
    # certificates (X.509) signed by the central MedFed Root CA must be enforced
    # to authenticate both server and participating hospital nodes.
    """
    if ca_cert_path and server_cert_path and server_key_path:
        with open(ca_cert_path, "rb") as f:
            ca_cert = f.read()
        with open(server_cert_path, "rb") as f:
            server_cert = f.read()
        with open(server_key_path, "rb") as f:
            server_key = f.read()
        return (ca_cert, server_cert, server_key)
    return None

class MedFedAvgStrategy(FedAvg):
    """
    Customized FedAvg Strategy for MedFed AI baseline evaluation.
    Tracks round-by-round global F1, ROC-AUC, straggler nodes, and comms volume.
    """
    def __init__(
        self,
        output_dir="C:/megafedallmodels/fedv2/runs/fedavg",
        min_fit_clients=3,
        min_available_clients=3,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        initial_parameters=None,
        **kwargs
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            **kwargs
        )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.round_history = []
        self.round_start_time = None
        self.strategy_name = "FedAvg"

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """Pass round number and hyperparams to clients."""
        self.round_start_time = time.time()
        logger.info(f"\n==================== [FL Round {server_round}] Starting {self.strategy_name} Aggregation ====================")
        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        for client, fit_arg in fit_ins:
            fit_arg.config["server_round"] = server_round
        return fit_ins

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregates parameters using FedAvg and logs stragglers, comms, and latency.
        """
        round_duration = time.time() - self.round_start_time if self.round_start_time else 0.0

        if not results:
            logger.warning(f"[Round {server_round}] No client results received!")
            return None, {}

        # Aggregate parameters via standard FedAvg
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        # Parse telemetry metrics from all clients
        client_latencies = {}
        client_losses = {}
        client_f1s = {}
        total_payload_mb = 0.0

        for client, fit_res in results:
            m = fit_res.metrics
            c_name = m.get("node_name", client.cid)
            lat = float(m.get("training_time_sec", 0.0))
            loss = float(m.get("train_loss", 0.0))
            f1 = float(m.get("val_macro_f1", 0.0))
            payload = float(m.get("payload_mb", 0.0))

            client_latencies[c_name] = lat
            client_losses[c_name] = loss
            client_f1s[c_name] = f1
            total_payload_mb += payload

        # Identify Straggler Node (Slowest Reporting Client)
        straggler_node = max(client_latencies.items(), key=lambda x: x[1])[0] if client_latencies else "None"
        straggler_time = client_latencies.get(straggler_node, 0.0)

        # Communication volume includes downstream (broadcast) + upstream (upload)
        total_comm_mb = total_payload_mb * 2.0

        round_record = {
            "round": server_round,
            "strategy": self.strategy_name,
            "round_duration_sec": float(round_duration),
            "total_comm_mb": float(total_comm_mb),
            "num_clients": len(results),
            "client_latencies": client_latencies,
            "client_losses": client_losses,
            "client_f1s": client_f1s,
            "straggler_node": straggler_node,
            "straggler_latency_sec": float(straggler_time)
        }

        logger.info(f"--- [Round {server_round} Summary] ---")
        logger.info(f"Wall-Clock Duration: {round_duration:.2f}s | Comm Volume: {total_comm_mb:.2f} MB")
        logger.info(f"Straggler Node: {straggler_node} ({straggler_time:.2f}s latency)")
        for c, f1 in client_f1s.items():
            logger.info(f"  Node [{c}]: Train Loss={client_losses[c]:.4f} | Local Val F1={f1:.4f}")

        # Save checkpoint of global model
        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            model = build_model(num_classes=len(DEFAULT_CHEST_XRAY_CLASSES), pretrained=False)
            set_model_parameters(model, ndarrays)

            ckpt_path = os.path.join(self.output_dir, f"global_model_round_{server_round}.pth")
            import torch
            torch.save({
                "round": server_round,
                "model_state_dict": model.state_dict(),
                "metrics": round_record
            }, ckpt_path)

            # Update latest symlink/file
            latest_path = os.path.join(self.output_dir, "global_model_latest.pth")
            torch.save({
                "round": server_round,
                "model_state_dict": model.state_dict(),
                "metrics": round_record
            }, latest_path)

        self.round_history.append(round_record)

        # Save cumulative history JSON
        history_file = os.path.join(self.output_dir, "server_history.json")
        with open(history_file, "w") as f:
            json.dump(self.round_history, f, indent=2)

        return aggregated_parameters, round_record

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregates local validation evaluations into global Macro F1 and ROC-AUC.
        """
        if not results:
            return None, {}

        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        # Weighted average of F1 and ROC-AUC across clients by sample size
        total_examples = sum(eval_res.num_examples for _, eval_res in results)
        weighted_f1 = sum(
            eval_res.num_examples * float(eval_res.metrics.get("val_macro_f1", 0.0))
            for _, eval_res in results
        ) / max(1, total_examples)

        weighted_auc = sum(
            eval_res.num_examples * float(eval_res.metrics.get("val_roc_auc", 0.5))
            for _, eval_res in results
        ) / max(1, total_examples)

        eval_summary = {
            "global_macro_f1": float(weighted_f1),
            "global_roc_auc": float(weighted_auc),
            "global_loss": float(loss_aggregated) if loss_aggregated else 0.0
        }

        logger.info(
            f"[Round {server_round} Global Eval] Macro F1: {weighted_f1:.4f} | "
            f"ROC-AUC: {weighted_auc:.4f} | Loss: {loss_aggregated:.4f}"
        )

        if self.round_history and self.round_history[-1]["round"] == server_round:
            self.round_history[-1].update(eval_summary)
            history_file = os.path.join(self.output_dir, "server_history.json")
            with open(history_file, "w") as f:
                json.dump(self.round_history, f, indent=2)

        return loss_aggregated, eval_summary

def run_server(
    server_address="127.0.0.1:8080",
    num_rounds=3,
    output_dir="C:/megafedallmodels/fedv2/runs/fedavg",
    min_clients=3
):
    """
    Initializes global model weights and launches the Flower server.
    """
    init_model = build_model(num_classes=len(DEFAULT_CHEST_XRAY_CLASSES), pretrained=True)
    init_params = ndarrays_to_parameters(get_model_parameters(init_model))

    strategy = MedFedAvgStrategy(
        output_dir=output_dir,
        min_fit_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=init_params
    )

    logger.info(f"Starting MedFed AI Flower Server on {server_address} for {num_rounds} rounds...")
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedFed AI — Central FL Server")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--min_clients", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="C:/megafedallmodels/fedv2/runs/fedavg")
    args = parser.parse_args()

    run_server(
        server_address=args.server_address,
        num_rounds=args.num_rounds,
        output_dir=args.output_dir,
        min_clients=args.min_clients
    )
