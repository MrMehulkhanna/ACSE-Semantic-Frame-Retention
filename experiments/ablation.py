"""
ACSE — Ablation Study
========================

Runs the full pipeline, then re-runs with each factor zeroed/neutralised.
Measures accuracy drop, retention-rate change, and latency change per ablation.
Generates a comparative bar chart.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.temporal_memory import TemporalGraphMemory
from core.novelty_engine import CognitiveNoveltyIndex
from core.event_decay import EventPersistenceDecay
from core.surprise_engine import SemanticSurpriseScore
from core.semantic_router import MultiFactorFusion
from core.adaptive_learning import OnlineBayesianUpdater
from core.decision_gate import AdaptiveThresholdGate


class AblationStudy:
    """Per-factor ablation: zero one factor at a time and measure impact."""

    ABLATIONS = {
        "Full Pipeline": {},
        "Remove CNI": {"cni": 0.5},
        "Remove EPD": {"epd": 0.0},
        "Remove SSS": {"sss": 0.5},
        "Remove PIG": {"pig": 0.0},
        "Remove RAW": {"raw": 1.0},
        "Fixed Equal Weights": {"fixed_weights": True},
        "No Temporal Memory": {"no_memory": True},
    }

    def __init__(self, results_dir: str = "experiments/results") -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _run_pipeline_on_dataset(
        self,
        df: pd.DataFrame,
        overrides: dict,
    ) -> dict:
        """Run ACSE pipeline on dataset with optional factor overrides."""
        use_fixed_weights = overrides.pop("fixed_weights", False)
        no_memory = overrides.pop("no_memory", False)

        weights = {"cni": 0.25, "epd": 0.25, "sss": 0.25, "pig": 0.25} if use_fixed_weights else None

        gate = AdaptiveThresholdGate(target_retention_rate=0.30)
        predictions = []

        t_start = time.perf_counter()

        for _, row in df.iterrows():
            cni = overrides.get("cni", row["cni"])
            epd = overrides.get("epd", row["epd"])
            sss = overrides.get("sss", row["sss"])
            pig = overrides.get("pig", row["pig"])
            raw_val = overrides.get("raw", row["raw"])

            w = weights or {"cni": 0.4, "epd": 0.2, "sss": 0.3, "pig": 0.1}

            if no_memory:
                # Without memory, treat each frame independently — CNI meaningless
                cni = 0.5
                pig = 0.0

            score = (w["cni"] * cni - w["epd"] * epd + w["sss"] * sss + w["pig"] * pig) * raw_val
            score = float(np.clip(score, 0.0, 1.0))

            result = gate.decide(score, int(row["frame_idx"]))
            predictions.append(1 if result["decision"] == "RETAIN" else 0)

        elapsed = time.perf_counter() - t_start
        y_pred = np.array(predictions)
        y_true = df["label"].values

        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        retention = float(y_pred.mean())

        return {
            "accuracy": round(acc, 4),
            "f1": round(f1, 4),
            "retention_rate": round(retention, 4),
            "latency_s": round(elapsed, 4),
        }

    def run(self, dataset_path: str) -> pd.DataFrame:
        """Run all ablations and return comparison DataFrame."""
        df = pd.read_csv(dataset_path)
        records = []

        for name, overrides in self.ABLATIONS.items():
            print(f"  Running ablation: {name}")
            result = self._run_pipeline_on_dataset(df, overrides.copy())
            result["ablation"] = name
            records.append(result)

        results_df = pd.DataFrame(records).set_index("ablation")

        # Save CSV
        csv_path = self.results_dir / "ablation_results.csv"
        results_df.to_csv(csv_path)
        print(f"  Ablation results saved: {csv_path}")

        # Generate bar chart
        self._plot_ablation_chart(results_df)

        return results_df

    def _plot_ablation_chart(self, df: pd.DataFrame) -> None:
        """Multi-metric bar chart comparing ablations."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        metrics = [("accuracy", "Accuracy"), ("f1", "F1 Score"), ("retention_rate", "Retention Rate")]
        colors = ["#4C72B0", "#55A868", "#C44E52"]

        for ax, (col, title), color in zip(axes, metrics, colors):
            bars = ax.barh(df.index, df[col], color=color, edgecolor="white", height=0.6)
            ax.set_xlabel(title)
            ax.set_title(title)
            ax.set_xlim(0, 1.05)
            ax.grid(axis="x", alpha=0.3)

            # Add value labels
            for bar, val in zip(bars, df[col]):
                ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=9)

        plt.suptitle("ACSE Ablation Study", fontsize=14, fontweight="bold")
        plt.tight_layout()
        chart_path = self.results_dir / "ablation_chart.png"
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Ablation chart saved: {chart_path}")


if __name__ == "__main__":
    csv_path = PROJECT_ROOT / "dataset" / "saml_frames.csv"
    if csv_path.exists():
        study = AblationStudy(results_dir=str(PROJECT_ROOT / "experiments" / "results"))
        results = study.run(str(csv_path))
        print("\n", results)
    else:
        print(f"Dataset not found at {csv_path}. Run dataset_builder.py first.")
