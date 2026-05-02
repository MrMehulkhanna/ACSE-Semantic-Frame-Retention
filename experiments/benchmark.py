"""
ACSE — Benchmark Suite
========================

Measures latency (per-module and total pipeline), accuracy (vs ground truth),
and retention-rate trade-offs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.temporal_memory import TemporalGraphMemory
from core.novelty_engine import CognitiveNoveltyIndex
from core.event_decay import EventPersistenceDecay
from core.surprise_engine import SemanticSurpriseScore
from core.semantic_router import MultiFactorFusion
from core.decision_gate import AdaptiveThresholdGate


def _make_random_frame(rng, h=240, w=320):
    return rng.randint(0, 255, (h, w, 3), dtype=np.uint8)


class ACSEBenchmark:
    """Latency, accuracy, and retention benchmarks for the ACSE pipeline."""

    def __init__(self, results_dir: str = "experiments/results") -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Latency benchmark
    # ------------------------------------------------------------------

    def run_latency_benchmark(self, n_frames: int = 500) -> dict:
        """Time each module separately on random frames."""
        rng = np.random.RandomState(0)
        memory = TemporalGraphMemory()
        cni_engine = CognitiveNoveltyIndex()
        epd_engine = EventPersistenceDecay()
        sss_engine = SemanticSurpriseScore()
        fusion = MultiFactorFusion()
        gate = AdaptiveThresholdGate()

        timings = {
            "memory_update": [],
            "cni_compute": [],
            "epd_update": [],
            "sss_update": [],
            "fusion": [],
            "gate": [],
            "total": [],
        }

        prev_frame = None
        for i in range(n_frames):
            frame = _make_random_frame(rng)
            total_start = time.perf_counter()

            # Memory
            t0 = time.perf_counter()
            desc = memory.update(frame, i, salience=0.5)
            timings["memory_update"].append(time.perf_counter() - t0)

            # CNI
            t0 = time.perf_counter()
            cni = cni_engine.compute(desc, memory)
            timings["cni_compute"].append(time.perf_counter() - t0)

            # EPD
            t0 = time.perf_counter()
            epd_engine.update(frame, prev_frame, i)
            epd = epd_engine.compute_epd_score(i)
            timings["epd_update"].append(time.perf_counter() - t0)

            # SSS
            t0 = time.perf_counter()
            sss_engine.update_predictor(desc)
            sss = sss_engine.compute(desc)
            timings["sss_update"].append(time.perf_counter() - t0)

            # Fusion
            t0 = time.perf_counter()
            fused = fusion.fuse(cni, epd, sss, memory, i)
            timings["fusion"].append(time.perf_counter() - t0)

            # Gate
            t0 = time.perf_counter()
            gate.decide(fused["importance_score"], i)
            timings["gate"].append(time.perf_counter() - t0)

            timings["total"].append(time.perf_counter() - total_start)
            prev_frame = frame

        # Build report
        report = {}
        for module_name, times in timings.items():
            arr = np.array(times) * 1000  # convert to ms
            report[module_name] = {
                "mean_ms": round(float(arr.mean()), 3),
                "std_ms": round(float(arr.std()), 3),
                "p95_ms": round(float(np.percentile(arr, 95)), 3),
                "p99_ms": round(float(np.percentile(arr, 99)), 3),
            }

        report_path = self.results_dir / "latency_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Latency report saved: {report_path}")
        return report

    # ------------------------------------------------------------------
    # Accuracy benchmark
    # ------------------------------------------------------------------

    def run_accuracy_benchmark(self, dataset_path: str) -> dict:
        """Run pipeline on dataset CSV and compare to ground truth labels."""
        df = pd.read_csv(dataset_path)

        # Use importance_score > adaptive threshold as prediction
        # We re-run the gate to get decisions
        gate = AdaptiveThresholdGate(target_retention_rate=0.30)
        predictions = []
        for _, row in df.iterrows():
            result = gate.decide(row["importance_score"], int(row["frame_idx"]))
            predictions.append(1 if result["decision"] == "RETAIN" else 0)

        y_true = df["label"].values
        y_pred = np.array(predictions)
        y_scores = df["importance_score"].values

        overall = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_true, y_scores)), 4) if len(np.unique(y_true)) > 1 else 0.0,
        }

        # Per-scene metrics
        per_scene = {}
        for scene in df["scene_type"].unique():
            mask = df["scene_type"] == scene
            yt = y_true[mask]
            yp = y_pred[mask]
            ys = y_scores[mask]
            per_scene[scene] = {
                "accuracy": round(float(accuracy_score(yt, yp)), 4),
                "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
                "recall": round(float(recall_score(yt, yp, zero_division=0)), 4),
                "f1": round(float(f1_score(yt, yp, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(yt, ys)), 4) if len(np.unique(yt)) > 1 else 0.0,
            }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["DISCARD", "RETAIN"],
                    yticklabels=["Not Important", "Important"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("ACSE Confusion Matrix")
        cm_path = self.results_dir / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Confusion matrix saved: {cm_path}")

        result = {"overall": overall, "per_scene": per_scene}
        result_path = self.results_dir / "accuracy_report.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Accuracy report saved: {result_path}")
        return result

    # ------------------------------------------------------------------
    # Retention benchmark
    # ------------------------------------------------------------------

    def run_retention_benchmark(
        self,
        dataset_path: str,
        target_rates: list[float] | None = None,
    ) -> dict:
        """Measure precision/recall at different target retention rates."""
        if target_rates is None:
            target_rates = [0.20, 0.30, 0.40, 0.50]

        df = pd.read_csv(dataset_path)
        y_true = df["label"].values
        scores = df["importance_score"].values

        results = {}
        rate_vals, prec_vals, rec_vals = [], [], []

        for target_rate in target_rates:
            gate = AdaptiveThresholdGate(target_retention_rate=target_rate)
            preds = []
            for idx, score in enumerate(scores):
                r = gate.decide(float(score), idx)
                preds.append(1 if r["decision"] == "RETAIN" else 0)

            preds = np.array(preds)
            actual_rate = float(preds.mean())
            prec = float(precision_score(y_true, preds, zero_division=0))
            rec = float(recall_score(y_true, preds, zero_division=0))

            results[f"target_{target_rate}"] = {
                "target_rate": target_rate,
                "actual_rate": round(actual_rate, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
            }
            rate_vals.append(actual_rate)
            prec_vals.append(prec)
            rec_vals.append(rec)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rate_vals, prec_vals, "o-", label="Precision", linewidth=2)
        ax.plot(rate_vals, rec_vals, "s-", label="Recall", linewidth=2)
        ax.set_xlabel("Actual Retention Rate")
        ax.set_ylabel("Score")
        ax.set_title("ACSE: Retention Rate vs Precision/Recall")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_path = self.results_dir / "retention_curve.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Retention curve saved: {plot_path}")

        return results


if __name__ == "__main__":
    bm = ACSEBenchmark(results_dir=str(PROJECT_ROOT / "experiments" / "results"))

    print("=== Latency Benchmark ===")
    lat = bm.run_latency_benchmark(n_frames=200)
    for m, stats in lat.items():
        print(f"  {m:20s}: {stats['mean_ms']:.3f} ms (p95={stats['p95_ms']:.3f})")

    csv_path = PROJECT_ROOT / "dataset" / "saml_frames.csv"
    if csv_path.exists():
        print("\n=== Accuracy Benchmark ===")
        acc = bm.run_accuracy_benchmark(str(csv_path))
        print(f"  Overall: {acc['overall']}")

        print("\n=== Retention Benchmark ===")
        ret = bm.run_retention_benchmark(str(csv_path))
        for k, v in ret.items():
            print(f"  {k}: {v}")
