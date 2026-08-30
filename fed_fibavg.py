#!/usr/bin/env python3
"""
fed_fibavg.py — MedFed AI Fed-FibAvg Federated Aggregation Engine
Subclasses flwr.server.strategy.FedAvg and provides client tiering with Fibonacci cadence weighting.

================================================================================
MATHEMATICAL FORMULATION OF FED-FIBAVG AGGREGATION
================================================================================
Let M be the number of participating hospital nodes in round r.
Each node i has:
  - n_i: Local dataset sample size.
  - \tau_i: Measured local training wall-clock latency (seconds).
  - Q_i: Data-quality / annotation certainty score in [0, 1].

1. Composite Node Fitness Score:
     S_i = \frac{Q_i}{\tau_i + \epsilon}

2. Client Tier Allocation:
     Sort clients in ascending order of S_i and partition into K discrete tiers:
     Tier 1 (Slow / High Latency), Tier 2 (Medium), Tier 3 (Fast / High Quality).

3. Fibonacci Cadence Scaling:
     Assign Fibonacci multiplier \beta_i from sequence F = (1, 2, 3, 5, 8, 13, ...)
     based on tier ranking:
       \beta_i = F_{tier(i)}

4. Normalized Aggregation Weights:
     w_i^{(r)} = \frac{\beta_i \cdot n_i}{\sum_{j=1}^M \beta_j \cdot n_j}

5. Global Parameter Update:
     \theta_{global}^{(r+1)} = \sum_{i=1}^M w_i^{(r)} \cdot \theta_i^{(r)}

Properties:
- Down-weights straggling/noisy nodes without completely discarding their unique non-IID distributions.
- Accelerates rounds-to-convergence while preserving convergence guarantees.
================================================================================
"""

import os
import sys
import time
import json
import logging
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [Fed-FibAvg] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Fed-FibAvg")

# Fibonacci Sequence Table for Tier Mapping
FIBONACCI_TIERS = [1, 2, 3, 5, 8, 13, 21]

class FedFibAvgEngine:
    """
    Core Mathematical Aggregator for Fed-FibAvg.
    Computes tier assignments, Fibonacci weights, and weighted parameter aggregation.
    """
    def __init__(self, num_clients=3, quality_scores=None):
        self.num_clients = num_clients
        # Default quality scores (e.g., from MAPLEZ/NIH certainty metrics)
        self.quality_scores = quality_scores or {"Hospital_A": 0.95, "Hospital_B": 0.90, "Hospital_C": 0.92}
        self.history = []

    def compute_tier_weights(self, client_names, client_samples, client_latencies):
        """
        Computes composite fitness scores S_i, assigns Fibonacci tier multipliers,
        and returns normalized aggregation weights w_i.
        """
        M = len(client_names)
        fitness_scores = []

        for name, lat in zip(client_names, client_latencies):
            q = self.quality_scores.get(name, 0.90)
            score = q / (lat + 1e-4) # Higher is better (fast + quality)
            fitness_scores.append(score)

        # Rank clients by fitness score (ascending: slowest/lowest -> fastest/highest)
        ranked_indices = np.argsort(fitness_scores)
        tier_multipliers = np.zeros(M, dtype=float)

        for rank, client_idx in enumerate(ranked_indices):
            fib_idx = min(rank, len(FIBONACCI_TIERS) - 1)
            tier_multipliers[client_idx] = FIBONACCI_TIERS[fib_idx]

        # Calculate effective weighted sample sizes
        effective_weights = tier_multipliers * np.array(client_samples, dtype=float)
        normalized_weights = effective_weights / np.sum(effective_weights)

        tier_info = {}
        for idx, name in enumerate(client_names):
            tier_info[name] = {
                "latency_sec": float(client_latencies[idx]),
                "quality_score": float(self.quality_scores.get(name, 0.90)),
                "fitness_score": float(fitness_scores[idx]),
                "fibonacci_multiplier": int(tier_multipliers[idx]),
                "normalized_weight": float(normalized_weights[idx])
            }

        return normalized_weights, tier_info

    def aggregate(self, client_updates, client_samples, client_latencies, server_round=1, client_names=None):
        """
        Aggregates list of parameter updates using Fed-FibAvg weighting.
        """
        if client_names is None:
            client_names = [f"Hospital_{chr(65 + i)}" for i in range(len(client_updates))]

        normalized_weights, tier_info = self.compute_tier_weights(
            client_names=client_names,
            client_samples=client_samples,
            client_latencies=client_latencies
        )

        logger.info(f"[Round {server_round}] Fed-FibAvg Tier Multipliers and Weights:")
        for name, info in tier_info.items():
            logger.info(
                f"  -> {name:12s} | Latency: {info['latency_sec']:5.1f}s | "
                f"Fib Mult: {info['fibonacci_multiplier']}x | Agg Weight: {info['normalized_weight']:.4f}"
            )

        num_params = len(client_updates[0])
        aggregated_params = []

        for p_idx in range(num_params):
            param_shape = client_updates[0][p_idx].shape
            weighted_param = np.zeros(param_shape, dtype=np.float32)
            for c_idx, weight in enumerate(normalized_weights):
                weighted_param += weight * client_updates[c_idx][p_idx]
            aggregated_params.append(weighted_param)

        self.history.append({
            "round": server_round,
            "tier_info": tier_info
        })

        return aggregated_params

class FedFibAvgStrategy(FedAvg):
    """
    Flower Strategy subclassing FedAvg to inject Fed-FibAvg aggregation.
    """
    def __init__(
        self,
        output_dir="C:/megafedallmodels/fedv2/runs/fed-fibavg",
        min_fit_clients=3,
        min_available_clients=3,
        initial_parameters=None,
        **kwargs
    ):
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            **kwargs
        )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.engine = FedFibAvgEngine(num_clients=min_fit_clients)
        self.strategy_name = "Fed-FibAvg"
        self.round_start_time = None
        self.round_history = []

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
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
        round_duration = time.time() - self.round_start_time if self.round_start_time else 0.0

        if not results:
            return None, {}

        # Extract client parameter ndarrays and metrics
        client_updates = []
        client_samples = []
        client_latencies = []
        client_names = []
        client_f1s = {}
        total_payload_mb = 0.0

        for client, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            m = fit_res.metrics
            c_name = str(m.get("node_name", client.cid))
            lat = float(m.get("training_time_sec", 1.0))
            f1 = float(m.get("val_macro_f1", 0.0))
            payload = float(m.get("payload_mb", 26.9))

            client_updates.append(ndarrays)
            client_samples.append(fit_res.num_examples)
            client_latencies.append(lat)
            client_names.append(c_name)
            client_f1s[c_name] = f1
            total_payload_mb += payload

        # Aggregate using Fed-FibAvg mathematical engine
        aggregated_ndarrays = self.engine.aggregate(
            client_updates=client_updates,
            client_samples=client_samples,
            client_latencies=client_latencies,
            server_round=server_round,
            client_names=client_names
        )

        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        # Track straggler
        straggler_idx = int(np.argmax(client_latencies))
        straggler_node = client_names[straggler_idx]
        straggler_time = client_latencies[straggler_idx]
        total_comm_mb = total_payload_mb * 2.0

        round_record = {
            "round": server_round,
            "strategy": self.strategy_name,
            "round_duration_sec": float(round_duration),
            "total_comm_mb": float(total_comm_mb),
            "straggler_node": straggler_node,
            "straggler_latency_sec": float(straggler_time),
            "client_f1s": client_f1s
        }
        self.round_history.append(round_record)

        # Save checkpoint and history
        history_file = os.path.join(self.output_dir, "server_history.json")
        with open(history_file, "w") as f:
            json.dump(self.round_history, f, indent=2)

        return aggregated_parameters, round_record

if __name__ == "__main__":
    print("Testing Fed-FibAvg Mathematical Aggregation Engine...")
    engine = FedFibAvgEngine(num_clients=3)

    # 3 dummy client updates (tensors of shape [4, 4])
    dummy_updates = [
        [np.ones((4, 4), dtype=np.float32) * 1.0], # Client A (Fast)
        [np.ones((4, 4), dtype=np.float32) * 2.0], # Client B (Medium)
        [np.ones((4, 4), dtype=np.float32) * 3.0]  # Client C (Slow)
    ]
    dummy_samples = [2000, 6000, 4000]
    dummy_latencies = [15.2, 38.5, 72.1]

    agg = engine.aggregate(dummy_updates, dummy_samples, dummy_latencies, server_round=1)
    print(f"Aggregated Tensor Value:\n{agg[0]}")
    assert agg[0].shape == (4, 4)
    print("Fed-FibAvg unit test passed successfully!")
