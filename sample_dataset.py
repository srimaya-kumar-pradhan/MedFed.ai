#!/usr/bin/env python3
"""
sample_dataset.py — MedFed AI Phase 1 Data Pipeline
Samples a fixed number of images (default 1,300) from each of the 12 image subfolders
in the NIH Chest X-ray dataset and merges them with multi-label annotations.

Global Constraints:
- Reproducibility: Accepts --seed argument and logs it.
- Honesty: Flags shortfall if any folder has fewer than target images.
- Reconciliation: Flags difference between 15,600 sampled images and PRD's ~14,000 target.
"""

import os
import sys
import argparse
import logging
import random
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SampleDataset")

# NIH 14 Disease Pathologies + No Finding
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

def parse_args():
    parser = argparse.ArgumentParser(description="MedFed AI — Sample NIH Chest X-ray Dataset")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="C:/Users/srinu/Music/fedlearning/DataSET",
        help="Root path to raw NIH dataset containing images_001..images_012 and Data_Entry_2017.csv"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="C:/megafedallmodels/fedv2",
        help="Directory where sampled_manifest.csv will be saved"
    )
    parser.add_argument(
        "--samples_per_folder",
        type=int,
        default=1300,
        help="Number of images to randomly sample from each of the 12 subfolders (default: 1300)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--manifest_name",
        type=str,
        default="sampled_manifest.csv",
        help="Filename for the output manifest CSV"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"=== MedFed AI Data Sampling (Seed: {args.seed}) ===")

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 1. Validate dataset directory and metadata
    data_entry_path = os.path.join(args.dataset_dir, "Data_Entry_2017.csv")
    if not os.path.exists(data_entry_path):
        raise FileNotFoundError(f"Required metadata file not found at {data_entry_path}")

    logger.info(f"Loading metadata from {data_entry_path}...")
    meta_df = pd.read_csv(data_entry_path)
    logger.info(f"Loaded {len(meta_df)} total records from Data_Entry_2017.csv")

    # Normalize column names
    meta_df.columns = [c.strip() for c in meta_df.columns]

    # Index metadata by image filename for O(1) lookup
    meta_indexed = meta_df.set_index("Image Index")

    # 2. Iterate through the 12 folders
    sampled_records = []
    total_shortfall = 0
    folder_manifest_counts = {}

    for folder_idx in range(1, 13):
        folder_name = f"images_{folder_idx:03d}"
        folder_images_path = os.path.join(args.dataset_dir, folder_name, "images")

        if not os.path.exists(folder_images_path):
            logger.warning(f"Subfolder {folder_name}/images does not exist! Recording 0 sampled.")
            folder_manifest_counts[folder_name] = {"available": 0, "sampled": 0, "shortfall": args.samples_per_folder}
            total_shortfall += args.samples_per_folder
            continue

        all_files = sorted([
            f for f in os.listdir(folder_images_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        num_available = len(all_files)

        if num_available < args.samples_per_folder:
            logger.warning(
                f"[{folder_name}] Shortfall: Only {num_available} images available (requested {args.samples_per_folder}). "
                f"Taking all available."
            )
            sampled_files = all_files
            shortfall = args.samples_per_folder - num_available
            total_shortfall += shortfall
        else:
            # Deterministic seeded random sampling without replacement
            sampled_files = random.sample(all_files, args.samples_per_folder)
            shortfall = 0

        folder_manifest_counts[folder_name] = {
            "available": num_available,
            "sampled": len(sampled_files),
            "shortfall": shortfall
        }
        logger.info(f"[{folder_name}] Available: {num_available:,} | Sampled: {len(sampled_files):,} | Shortfall: {shortfall}")

        # Build record for each sampled image
        for fname in sampled_files:
            abs_image_path = os.path.abspath(os.path.join(folder_images_path, fname))
            rel_image_path = os.path.join(folder_name, "images", fname)

            # Lookup metadata
            if fname in meta_indexed.index:
                row = meta_indexed.loc[fname]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]

                finding_labels = str(row.get("Finding Labels", "No Finding"))
                follow_up = row.get("Follow-up #", 0)
                patient_id = row.get("Patient ID", -1)
                patient_age = row.get("Patient Age", 0)
                patient_gender = str(row.get("Patient Gender", "U"))
                view_pos = str(row.get("View Position", "PA"))
            else:
                finding_labels = "No Finding"
                follow_up = 0
                patient_id = -1
                patient_age = 0
                patient_gender = "U"
                view_pos = "PA"

            # Parse multi-label binary indicators (0 or 1)
            label_list = [l.strip() for l in finding_labels.split("|")]
            record = {
                "image_name": fname,
                "source_folder": folder_name,
                "image_path": abs_image_path,
                "relative_path": rel_image_path,
                "finding_labels": finding_labels,
                "follow_up_num": follow_up,
                "patient_id": patient_id,
                "patient_age": patient_age,
                "patient_gender": patient_gender,
                "view_position": view_pos,
                "No_Finding": 1 if "No Finding" in label_list else 0
            }

            for p_class in PATHOLOGY_CLASSES:
                record[p_class] = 1 if p_class in label_list else 0

            sampled_records.append(record)

    # 3. Create DataFrame and Validate
    manifest_df = pd.DataFrame(sampled_records)
    total_sampled = len(manifest_df)
    target_total = 12 * args.samples_per_folder

    logger.info("=== Sampling Reconciliation Summary ===")
    logger.info(f"Target Total (12 × {args.samples_per_folder}): {target_total:,} images")
    logger.info(f"Actual Sampled Total: {total_sampled:,} images")
    logger.info(f"Total Shortfall: {total_shortfall:,} images")
    logger.info(f"PRD ~14,000 Target Delta: {total_sampled - 14000:+,} images (reconciled as full 12-folder quota)")

    # Pathology class distribution summary
    logger.info("Pathology class positive counts across sampled dataset:")
    for p_class in PATHOLOGY_CLASSES + ['No_Finding']:
        cnt = manifest_df[p_class].sum()
        pct = (cnt / total_sampled) * 100
        logger.info(f"  {p_class:20s}: {cnt:5d} ({pct:5.2f}%)")

    # 4. Save Manifest CSV
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, args.manifest_name)
    manifest_df.to_csv(manifest_path, index=False)
    logger.info(f"Saved manifest CSV to: {manifest_path}")

    # Also save sampling metadata / shortfall log
    summary_path = os.path.join(args.output_dir, "sampling_summary.json")
    import json
    with open(summary_path, "w") as f:
        json.dump({
            "seed": args.seed,
            "samples_per_folder": args.samples_per_folder,
            "target_total": target_total,
            "actual_total": total_sampled,
            "prd_delta": total_sampled - 14000,
            "total_shortfall": total_shortfall,
            "folder_counts": folder_manifest_counts,
            "pathology_counts": {p: int(manifest_df[p].sum()) for p in PATHOLOGY_CLASSES + ['No_Finding']}
        }, f, indent=2)
    logger.info(f"Saved sampling summary to: {summary_path}")

if __name__ == "__main__":
    main()
