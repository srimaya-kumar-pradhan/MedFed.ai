#!/usr/bin/env python3
"""
partition_nodes.py — MedFed AI Phase 1 Non-IID Partitioning
Splits the sampled dataset manifest into simulated hospital node directories
with provable non-IID distribution (Dirichlet distribution + clinical specializations).

Global Constraints:
- Data Locality: Each node is completely isolated with its own train/val/test CSV splits.
- Disjointness: No image path appears in more than one node folder.
- Reproducibility: Seeded via --seed.
"""

import os
import sys
import argparse
import logging
import json
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PartitionNodes")

PATHOLOGY_CLASSES = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Effusion',
    'Emphysema',
    'Fibrosis',
    'Hernia',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pleural_Thickening',
    'Pneumonia',
    'Pneumothorax'
]

# Clinical Specialization Weights for Default 3 Hospitals
DEFAULT_HOSPITAL_PROFILES = {
    "Hospital_A": {
        "name": "Metropolitan General Hospital (Cardio & Routine)",
        "bias_classes": ["No_Finding", "Cardiomegaly", "Effusion", "Hernia"],
        "bias_multiplier": 3.5
    },
    "Hospital_B": {
        "name": "Pulmonary & Infectious Disease Center",
        "bias_classes": ["Pneumonia", "Infiltration", "Consolidation", "Edema", "Atelectasis"],
        "bias_multiplier": 3.5
    },
    "Hospital_C": {
        "name": "Oncology & Thoracic Surgery Institute",
        "bias_classes": ["Nodule", "Mass", "Emphysema", "Pneumothorax", "Fibrosis", "Pleural_Thickening"],
        "bias_multiplier": 3.5
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="MedFed AI — Non-IID Hospital Node Partitioning")
    parser.add_argument(
        "--manifest",
        type=str,
        default="C:/megafedallmodels/fedv2/sampled_manifest.csv",
        help="Path to sampled_manifest.csv from Phase 1 step 1"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="C:/megafedallmodels/fedv2",
        help="Base directory where hospital node folders will be created"
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=3,
        choices=[3, 4, 5],
        help="Number of simulated hospital nodes (3-5, default: 3)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet concentration parameter for non-IID distribution (smaller = more non-IID, default: 0.5)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.80,
        help="Fraction for training split (default: 0.80)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.10,
        help="Fraction for validation split (default: 0.10)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.10,
        help="Fraction for test split (default: 0.10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    return parser.parse_args()

def assign_non_iid_nodes(df, node_names, alpha, seed):
    """
    Assign each image to exactly one node using a Dirichlet-skewed allocation
    over primary pathology classes + hospital bias.
    Guarantees mutually disjoint sets (no image shared across nodes).
    """
    np.random.seed(seed)
    random.seed(seed)
    num_nodes = len(node_names)

    # Determine primary label category for each image
    def get_primary_category(row):
        for p in PATHOLOGY_CLASSES:
            if row[p] == 1:
                return p
        return "No_Finding"

    df = df.copy()
    df['primary_category'] = df.apply(get_primary_category, axis=1)

    all_categories = PATHOLOGY_CLASSES + ['No_Finding']
    category_node_probs = {}

    for cat in all_categories:
        # Base Dirichlet distribution per category
        dirichlet_weights = np.random.dirichlet([alpha] * num_nodes)

        # Apply hospital clinical specialization bias
        for i, node in enumerate(node_names):
            if node in DEFAULT_HOSPITAL_PROFILES:
                prof = DEFAULT_HOSPITAL_PROFILES[node]
                if cat in prof["bias_classes"]:
                    dirichlet_weights[i] *= prof["bias_multiplier"]

        # Normalize to probability distribution
        dirichlet_weights /= dirichlet_weights.sum()
        category_node_probs[cat] = dirichlet_weights

    # Assign each sample to a node according to the computed probabilities
    node_assignments = []
    for idx, row in df.iterrows():
        cat = row['primary_category']
        probs = category_node_probs[cat]
        assigned_node_idx = np.random.choice(num_nodes, p=probs)
        node_assignments.append(node_names[assigned_node_idx])

    df['assigned_node'] = node_assignments
    return df, category_node_probs

def stratified_node_split(node_df, train_ratio, val_ratio, test_ratio, seed):
    """
    Split a single node's dataset into train/val/test splits,
    stratified by primary pathology category.
    """
    val_plus_test_ratio = val_ratio + test_ratio

    # Check minimum class count for stratification
    cat_counts = node_df['primary_category'].value_counts()
    rare_cats = cat_counts[cat_counts < 3].index.tolist()

    # If some categories have < 3 samples, fall back to simple random split for those
    if len(rare_cats) > 0:
        logger.info(f"Rare categories with < 3 samples in node: {rare_cats}")
        train_df, temp_df = train_test_split(
            node_df,
            test_size=val_plus_test_ratio,
            random_state=seed,
            shuffle=True
        )
        test_fraction_of_temp = test_ratio / val_plus_test_ratio
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_fraction_of_temp,
            random_state=seed,
            shuffle=True
        )
    else:
        train_df, temp_df = train_test_split(
            node_df,
            test_size=val_plus_test_ratio,
            random_state=seed,
            stratify=node_df['primary_category'],
            shuffle=True
        )
        test_fraction_of_temp = test_ratio / val_plus_test_ratio

        temp_cat_counts = temp_df['primary_category'].value_counts()
        rare_temp = temp_cat_counts[temp_cat_counts < 2].index.tolist()
        stratify_temp = temp_df['primary_category'] if len(rare_temp) == 0 else None

        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_fraction_of_temp,
            random_state=seed,
            stratify=stratify_temp,
            shuffle=True
        )

    return train_df, val_df, test_df

def main():
    args = parse_args()
    logger.info(f"=== MedFed AI Non-IID Node Partitioning (Seed: {args.seed}, Alpha: {args.alpha}) ===")

    if not os.path.exists(args.manifest):
        raise FileNotFoundError(f"Manifest not found at {args.manifest}. Run sample_dataset.py first.")

    manifest_df = pd.read_csv(args.manifest)
    total_samples = len(manifest_df)
    logger.info(f"Loaded manifest with {total_samples:,} images.")

    # Determine node names
    node_names = [f"Hospital_{chr(65 + i)}" for i in range(args.num_nodes)] # Hospital_A, Hospital_B, ...
    logger.info(f"Partitioning across {args.num_nodes} nodes: {node_names}")

    # Assign images to nodes with Non-IID Dirichlet distribution
    assigned_df, category_node_probs = assign_non_iid_nodes(
        manifest_df,
        node_names=node_names,
        alpha=args.alpha,
        seed=args.seed
    )

    # Disjointness validation check
    seen_images = set()
    node_splits_summary = {}

    for node_name in node_names:
        node_df = assigned_df[assigned_df['assigned_node'] == node_name].copy()
        node_count = len(node_df)
        logger.info(f"\n[{node_name}] Total assigned samples: {node_count:,} ({node_count/total_samples*100:.1f}%)")

        # Check for overlap
        node_images = set(node_df['image_path'])
        overlap = seen_images.intersection(node_images)
        if len(overlap) > 0:
            raise ValueError(f"FATAL: Overlap detected! {len(overlap)} images in {node_name} already in another node.")
        seen_images.update(node_images)

        # Create node directory
        node_dir = os.path.join(args.output_dir, node_name)
        os.makedirs(node_dir, exist_ok=True)

        # Perform local stratified train/val/test split
        train_df, val_df, test_df = stratified_node_split(
            node_df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )

        # Drop temporary assignment columns before saving local CSVs
        cols_to_save = [c for c in train_df.columns if c not in ['assigned_node', 'primary_category']]

        train_path = os.path.join(node_dir, "train.csv")
        val_path = os.path.join(node_dir, "val.csv")
        test_path = os.path.join(node_dir, "test.csv")
        full_node_path = os.path.join(node_dir, "all_local.csv")

        train_df[cols_to_save].to_csv(train_path, index=False)
        val_df[cols_to_save].to_csv(val_path, index=False)
        test_df[cols_to_save].to_csv(test_path, index=False)
        node_df[cols_to_save].to_csv(full_node_path, index=False)

        logger.info(f"  -> Saved {train_path} ({len(train_df)} samples, {len(train_df)/node_count*100:.1f}%)")
        logger.info(f"  -> Saved {val_path} ({len(val_df)} samples, {len(val_df)/node_count*100:.1f}%)")
        logger.info(f"  -> Saved {test_path} ({len(test_df)} samples, {len(test_df)/node_count*100:.1f}%)")

        # Class distribution per node
        class_counts = {p: int(node_df[p].sum()) for p in PATHOLOGY_CLASSES + ['No_Finding']}
        node_splits_summary[node_name] = {
            "total_samples": node_count,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "class_counts": class_counts
        }

    # Verify total partition integrity
    total_partitioned = sum(s["total_samples"] for s in node_splits_summary.values())
    assert total_partitioned == total_samples, f"Total mismatch: {total_partitioned} vs {total_samples}"
    assert len(seen_images) == total_samples, "Disjointness check failed: image count mismatch"
    logger.info(f"\n[VALIDATION PASSED] All {total_samples:,} images partitioned into mutually disjoint sets!")

    # Save partition summary JSON
    summary_file = os.path.join(args.output_dir, "partition_summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "seed": args.seed,
            "alpha": args.alpha,
            "num_nodes": args.num_nodes,
            "total_images": total_samples,
            "nodes": node_splits_summary
        }, f, indent=2)
    logger.info(f"Saved partition summary to: {summary_file}")

if __name__ == "__main__":
    main()
