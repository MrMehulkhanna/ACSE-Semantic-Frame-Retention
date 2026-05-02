"""
ACSE — Module 5: Multi-Factor Fusion (Semantic Router)
========================================================

Combines CNI, EPD, SSS, PIG (Predictive Information Gain), and RAW
(Resource-Aware Weight) into a single Importance Score with adaptive weights.

RESEARCH CONTRIBUTION:
  Prior work fuses at most two signals (e.g. motion + appearance in Liu & Cheng
  2008).  Zhang et al. 2016 implicitly fuses via LSTM hidden state — opaque.
  VA-RED2 fuses input difficulty with target FLOPs — two signals only.  ReTaKe
  uses a single redundancy metric.
  Our fusion combines five heterogeneous factors — information-theoretic novelty,
  event persistence, autoregressive surprise, predictive information gain, and
  CPU load — with weights that are updated online by a Bayesian learning module.

NOVELTY CLAIM:
  A five-factor adaptive-weight fusion of information-theoretic novelty,
  event-persistence decay, autoregressive surprise, predictive information
  gain, and resource load, with online weight learning, for video frame
  retention gating.

POTENTIAL PATENT CLAIM DIRECTION:
  A method for computing a multi-factor importance score for a video frame
  comprising fusing a cognitive novelty index, an event persistence decay
  penalty, a semantic surprise score, a predictive information gain, and a
  resource-aware weight using online-adapted weights, and gating frame retention
  on said score.
"""

from __future__ import annotations

import numpy as np
import psutil
from sklearn.feature_selection import mutual_info_regression

from core.temporal_memory import TemporalGraphMemory


class MultiFactorFusion:
    """Fuse CNI, EPD, SSS, PIG, RAW into a single importance score."""

    def __init__(self, weights: dict | None = None) -> None:
        self.weights: dict = weights or {
            "cni": 0.4,
            "epd": 0.2,
            "sss": 0.3,
            "pig": 0.1,
        }

    def set_weights(self, weights: dict) -> None:
        self.weights = weights.copy()

    def compute_pig(self, memory: TemporalGraphMemory) -> float:
        """Predictive Information Gain: MI between consecutive descriptors."""
        if len(memory.memory_bank) < 4:
            return 0.0

        descs = [e["descriptor"] for e in memory.memory_bank]
        n = len(descs)
        X = np.stack(descs[: n - 1])  # D_{t-1}
        y_arr = np.stack(descs[1:n])  # D_t

        # Estimate MI via sklearn's kNN-based estimator on first few dims
        # Use first 8 components as proxy to keep it fast
        dims_to_use = min(8, X.shape[1])
        X_sub = X[:, :dims_to_use]

        mi_values = []
        for j in range(min(4, y_arr.shape[1])):
            try:
                mi = mutual_info_regression(
                    X_sub, y_arr[:, j], n_neighbors=min(3, len(X_sub) - 1), random_state=0
                )
                mi_values.append(float(mi.mean()))
            except ValueError:
                mi_values.append(0.0)

        raw_pig = float(np.mean(mi_values)) if mi_values else 0.0
        # Normalise to [0, 1] via sigmoid-like squash
        return float(np.clip(raw_pig / (raw_pig + 1.0), 0.0, 1.0))

    @staticmethod
    def compute_raw() -> float:
        """Resource-Aware Weight: 1 - CPU% / 100."""
        try:
            cpu = psutil.cpu_percent(interval=None)
        except Exception:
            cpu = 0.0
        return float(np.clip(1.0 - cpu / 100.0, 0.05, 1.0))

    def fuse(
        self,
        cni: float,
        epd: float,
        sss: float,
        memory: TemporalGraphMemory,
        frame_idx: int,
    ) -> dict:
        """Combine all factors into a single importance score."""
        pig = self.compute_pig(memory)
        raw = self.compute_raw()

        w = self.weights
        score = (
            w["cni"] * cni
            - w["epd"] * epd
            + w["sss"] * sss
            + w["pig"] * pig
        ) * raw

        score = float(np.clip(score, 0.0, 1.0))

        return {
            "importance_score": score,
            "cni": cni,
            "epd": epd,
            "sss": sss,
            "pig": pig,
            "raw": raw,
            "weights": w.copy(),
        }

    def fuse_without(self, exclude_factor: str, fusion_output: dict) -> float:
        """Re-compute importance score with one factor excluded (neutral value)."""
        w = fusion_output["weights"]
        neutrals = {"cni": 0.5, "epd": 0.0, "sss": 0.5, "pig": 0.0, "raw": 1.0}

        vals = {
            "cni": fusion_output["cni"],
            "epd": fusion_output["epd"],
            "sss": fusion_output["sss"],
            "pig": fusion_output["pig"],
            "raw": fusion_output["raw"],
        }
        vals[exclude_factor] = neutrals[exclude_factor]

        raw_val = vals["raw"]
        score = (
            w.get("cni", 0.4) * vals["cni"]
            - w.get("epd", 0.2) * vals["epd"]
            + w.get("sss", 0.3) * vals["sss"]
            + w.get("pig", 0.1) * vals["pig"]
        ) * raw_val

        return float(np.clip(score, 0.0, 1.0))
