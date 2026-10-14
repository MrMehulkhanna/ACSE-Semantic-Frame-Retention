"""Compare the fused importance score with chance on the synthetic dataset.

Run after `python experiments/dataset_builder.py`.
Usage: python experiments/baseline_check.py [retention_rate]
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

csv = Path(__file__).resolve().parent.parent / "dataset" / "saml_frames.csv"
df = pd.read_csv(csv)
p = df["label"].mean()
print(f"frames: {len(df)}   labelled important: {p:.1%}")

for col in ["importance_score", "cni", "epd", "sss", "pig", "raw"]:
    print(f"AUC {col:17s} {roc_auc_score(df['label'], df[col]):.3f}")

# A random gate that keeps the same fraction of frames as the full pipeline
r = float(sys.argv[1]) if len(sys.argv) > 1 else 0.292
print(f"random retention at {r:.1%}: expected F1 {2 * p * r / (p + r):.3f}, "
      f"accuracy {p * r + (1 - p) * (1 - r):.3f}")
