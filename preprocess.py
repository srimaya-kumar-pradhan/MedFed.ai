#!/usr/bin/env python3
"""
preprocess.py — MedFed AI Phase 1 Local Node Preprocessing & EDA Report
Runs independently per hospital node (never centrally) to validate data locality,
integrity, duplicate detection, and generate presentation-ready dark-theme EDA reports.

Global Constraints:
- No Cluttered Defaults: Dark theme (#2c3e50 background family, #7f8c8d secondary),
  dropped top/right spines, percentage annotations on bars, high DPI.
- Data Locality: Operates strictly within the target node's local files.
- Reproducibility: Seeded via --seed.
"""

import os
import sys
import argparse
import logging
import hashlib
import json
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PreprocessNode")

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

# Clinical Theme Palette (PRD Compliant)
DARK_THEME_BG = "#1e293b"        # Dark slate background
CARD_BG = "#2c3e50"              # Panel background
TEXT_COLOR = "#f8fafc"           # Crisp white text
SECONDARY_TEXT = "#94a3b8"       # Muted slate text
ACCENT_BLUE = "#38bdf8"          # MedFed cyan/blue
ACCENT_EMERALD = "#10b981"       # Success green
ACCENT_AMBER = "#f59e0b"         # Warning amber
ACCENT_ROSE = "#f43f5e"          # Clinical alert rose
SPINE_COLOR = "#475569"          # Subtle border color

def parse_args():
    parser = argparse.ArgumentParser(description="MedFed AI — Local Node Preprocessing & EDA")
    parser.add_argument(
        "--node_dir",
        type=str,
        required=True,
        help="Path to local hospital node directory (e.g., C:/megafedallmodels/fedv2/Hospital_A)"
    )
    parser.add_argument(
        "--node_name",
        type=str,
        default=None,
        help="Display name for the hospital node"
    )
    parser.add_argument(
        "--min_resolution",
        type=int,
        default=224,
        help="Minimum width and height allowed (default: 224)"
    )
    parser.add_argument(
        "--check_all_images",
        action="store_true",
        help="Perform full image decode check on all files in the split"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()

def validate_node_integrity(df, node_dir, min_res=224, deep_check=False):
    """
    Validate image readability, resolution floor, corruption, and local duplicates.
    """
    logger.info(f"Validating {len(df)} images in {node_dir}...")
    valid_records = []
    corrupted_count = 0
    res_shortfall_count = 0
    hash_seen = {}
    duplicates = []

    # Sample images for deep check if not checking all
    images_to_check = df if deep_check else df.head(min(len(df), 200))

    for idx, row in images_to_check.iterrows():
        img_path = row['image_path']
        if not os.path.exists(img_path):
            corrupted_count += 1
            continue

        try:
            with Image.open(img_path) as img:
                w, h = img.size
                if w < min_res or h < min_res:
                    res_shortfall_count += 1

                # Check for MD5 duplicates
                with open(img_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                if file_hash in hash_seen:
                    duplicates.append((row['image_name'], hash_seen[file_hash]))
                else:
                    hash_seen[file_hash] = row['image_name']

        except Exception as e:
            logger.warning(f"Error opening image {img_path}: {e}")
            corrupted_count += 1

    validation_stats = {
        "total_records": len(df),
        "checked_records": len(images_to_check),
        "corrupted_or_missing": corrupted_count,
        "below_min_resolution": res_shortfall_count,
        "internal_duplicates": len(duplicates),
        "status": "HEALTHY" if corrupted_count == 0 else "WARNING"
    }
    logger.info(f"Validation Stats: {validation_stats}")
    return validation_stats

def generate_eda_chart(df, node_name, output_png_path):
    """
    Generates a high-DPI dark-theme label distribution bar chart
    adhering to PRD visual style (#2c3e50 family, dropped spines, % annotations).
    """
    logger.info(f"Generating EDA visualization for {node_name}...")

    # Calculate counts and percentages for all 14 classes + No Finding
    classes = ['No_Finding'] + PATHOLOGY_CLASSES
    counts = [int(df[c].sum()) for c in classes]
    total_imgs = len(df)
    percentages = [(cnt / total_imgs) * 100 for cnt in counts]

    # Create formatted labels
    display_names = [c.replace('_', ' ') for c in classes]

    # Plot setup
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(13, 8), dpi=300)
    fig.patch.set_facecolor(DARK_THEME_BG)
    ax.set_facecolor(CARD_BG)

    # Color palette: Highlight highest prevalence classes
    colors = []
    max_pct = max(percentages) if percentages else 1
    for p in percentages:
        if p > 25:
            colors.append(ACCENT_ROSE)
        elif p > 10:
            colors.append(ACCENT_BLUE)
        elif p > 4:
            colors.append(ACCENT_EMERALD)
        else:
            colors.append("#64748b") # muted slate

    bars = ax.barh(display_names, counts, color=colors, height=0.68, edgecolor="none", zorder=3)

    # Annotate bars with both count and percentage
    for bar, count, pct in zip(bars, counts, percentages):
        width = bar.get_width()
        x_pos = width + (max(counts) * 0.015)
        text_str = f"{count:,} ({pct:.1f}%)" if count > 0 else "0 (0.0%)"
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            text_str,
            va='center',
            ha='left',
            color=TEXT_COLOR,
            fontsize=9.5,
            fontweight='semibold'
        )

    # Style axes and drop spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(SPINE_COLOR)
    ax.spines['bottom'].set_color(SPINE_COLOR)
    ax.tick_params(colors=SECONDARY_TEXT, labelsize=10, bottom=True, left=False)
    ax.grid(axis='x', color='#334155', linestyle='--', alpha=0.6, zorder=0)

    # Titles and labels
    ax.set_xlabel("Sample Count (Images)", color=TEXT_COLOR, fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title(
        f"Local Pathology Distribution — {node_name}\n"
        f"Total Samples: {total_imgs:,} | Isolated Local Partition (Non-IID)",
        color=TEXT_COLOR,
        fontsize=13,
        fontweight='bold',
        pad=18,
        loc='left'
    )

    # Invert y-axis so top class is at top
    ax.invert_yaxis()
    plt.tight_layout()

    # Save high-DPI figure
    fig.savefig(output_png_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved EDA chart to: {output_png_path}")

def generate_html_eda_report(node_name, stats, df, chart_filename, output_html_path):
    """
    Generates a clean clinical HTML EDA report for local hospital records.
    """
    classes = ['No_Finding'] + PATHOLOGY_CLASSES
    rows_html = ""
    total = len(df)

    for c in classes:
        cnt = int(df[c].sum())
        pct = (cnt / total) * 100
        bar_w = min(100, pct * 2.5)
        rows_html += f"""
        <tr>
            <td style="padding: 10px 14px; font-weight: 500;">{c.replace('_', ' ')}</td>
            <td style="padding: 10px 14px; text-align: right; font-family: monospace;">{cnt:,}</td>
            <td style="padding: 10px 14px; text-align: right; font-family: monospace;">{pct:.2f}%</td>
            <td style="padding: 10px 14px; width: 220px;">
                <div style="background: #334155; border-radius: 4px; height: 10px; width: 100%;">
                    <div style="background: #38bdf8; border-radius: 4px; height: 10px; width: {bar_w:.1f}%;"></div>
                </div>
            </td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MedFed AI — Local Node EDA Report ({node_name})</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        .header {{
            border-bottom: 2px solid #334155;
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        .badge {{
            display: inline-block;
            background: #0284c7;
            color: #fff;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .card {{
            background: #1e293b;
            border-radius: 10px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid #334155;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}
        .stat-box {{
            background: #0f172a;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #334155;
        }}
        .stat-val {{
            font-size: 24px;
            font-weight: 700;
            color: #38bdf8;
        }}
        .stat-lbl {{
            font-size: 12px;
            color: #94a3b8;
            margin-top: 4px;
            text-transform: uppercase;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th {{
            text-align: left;
            padding: 10px 14px;
            border-bottom: 2px solid #334155;
            color: #94a3b8;
            font-size: 12px;
            text-transform: uppercase;
        }}
        tr:nth-child(even) {{
            background: rgba(255, 255, 255, 0.02);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <span class="badge">Privacy-Preserving Local Node</span>
            <h1 style="margin: 8px 0 4px 0; font-size: 26px;">{node_name} — Data Pipeline & EDA Report</h1>
            <p style="color: #94a3b8; margin: 0; font-size: 14px;">MedFed AI Federated Diagnostics Engine | Local Data Partitioning Verified</p>
        </div>

        <div class="grid">
            <div class="stat-box">
                <div class="stat-val">{stats['total_records']:,}</div>
                <div class="stat-lbl">Total Local Images</div>
            </div>
            <div class="stat-box">
                <div class="stat-val">{int(df['No_Finding'].sum()):,}</div>
                <div class="stat-lbl">Normal (No Finding)</div>
            </div>
            <div class="stat-box">
                <div class="stat-val">{int((df[PATHOLOGY_CLASSES].sum(axis=1) > 0).sum()):,}</div>
                <div class="stat-lbl">Pathology Positive</div>
            </div>
            <div class="stat-box">
                <div class="stat-val" style="color: #10b981;">{stats['status']}</div>
                <div class="stat-lbl">Integrity Status</div>
            </div>
        </div>

        <div class="card">
            <h3 style="margin-top: 0; margin-bottom: 16px; color: #f8fafc;">Pathology Distribution Histogram</h3>
            <img src="{chart_filename}" style="width: 100%; border-radius: 6px;" alt="EDA Chart">
        </div>

        <div class="card">
            <h3 style="margin-top: 0; margin-bottom: 16px; color: #f8fafc;">Local Class Breakdown</h3>
            <table>
                <thead>
                    <tr>
                        <th>Pathology Category</th>
                        <th style="text-align: right;">Count</th>
                        <th style="text-align: right;">Percentage</th>
                        <th>Distribution Bar</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
    """
    with open(output_html_path, "w") as f:
        f.write(html_content)
    logger.info(f"Saved HTML EDA report to: {output_html_path}")

def main():
    args = parse_args()
    node_dir = os.path.abspath(args.node_dir)
    node_name = args.node_name or os.path.basename(node_dir)

    logger.info(f"=== Preprocessing & EDA for {node_name} ({node_dir}) ===")

    all_csv_path = os.path.join(node_dir, "all_local.csv")
    if not os.path.exists(all_csv_path):
        all_csv_path = os.path.join(node_dir, "train.csv")
        if not os.path.exists(all_csv_path):
            raise FileNotFoundError(f"No local CSV found in {node_dir}. Run partition_nodes.py first.")

    df = pd.read_csv(all_csv_path)
    logger.info(f"Loaded {len(df)} records from {all_csv_path}")

    # Validate integrity
    stats = validate_node_integrity(
        df,
        node_dir=node_dir,
        min_res=args.min_resolution,
        deep_check=args.check_all_images
    )

    # Save stats JSON
    stats_path = os.path.join(node_dir, "eda_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "node_name": node_name,
            "validation_stats": stats,
            "class_counts": {c: int(df[c].sum()) for c in ['No_Finding'] + PATHOLOGY_CLASSES},
            "class_imbalance_ratio": float(df['No_Finding'].sum() / max(1, df[PATHOLOGY_CLASSES].sum().max()))
        }, f, indent=2)

    # Generate Chart
    chart_path = os.path.join(node_dir, "eda_distribution.png")
    generate_eda_chart(df, node_name, chart_path)

    # Generate HTML Report
    html_path = os.path.join(node_dir, "eda_report.html")
    generate_html_eda_report(node_name, stats, df, "eda_distribution.png", html_path)

    logger.info(f"Local preprocessing & EDA for {node_name} complete!\n")

if __name__ == "__main__":
    main()
