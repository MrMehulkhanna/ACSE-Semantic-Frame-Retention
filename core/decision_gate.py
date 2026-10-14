"""
ACSE — Module 8b: Adaptive Threshold Gate
============================================

Applies the final retention decision with a threshold that adapts online to
maintain a configurable target retention rate.

Related approaches:
  Liu & Cheng 2008 adapts thresholds but only on pixel-level metrics with a
  fixed PID-like scheme.  Zhang et al. 2016 has no explicit threshold — it
  selects top-k frames post-hoc.  VA-RED2 uses a fixed target computation
  budget.  ReTaKe uses a redundancy cutoff fixed at training time.
  Our gate adapts a scalar threshold via a rate-tracking feedback loop:  if
  too many frames are retained, the threshold rises; if too few, it drops.
  The target retention rate is configurable per deployment.
"""

from collections import deque

import numpy as np


class AdaptiveThresholdGate:
    """Self-adjusting threshold to maintain target retention rate."""

    def __init__(
        self,
        initial_threshold: float = 0.5,
        target_retention_rate: float = 0.30,
        adaptation_rate: float = 0.01,
        window: int = 100,
    ) -> None:
        self.threshold: float = initial_threshold
        self.target_retention_rate: float = target_retention_rate
        self.adaptation_rate: float = adaptation_rate
        self.recent_decisions: deque = deque(maxlen=window)

    def decide(self, importance_score: float, frame_idx: int) -> dict:
        """Retain or discard based on current adaptive threshold."""
        decision = "RETAIN" if importance_score >= self.threshold else "DISCARD"
        retained = 1.0 if decision == "RETAIN" else 0.0
        self.recent_decisions.append(retained)

        # Adapt threshold after each decision
        self.adapt_threshold()

        return {
            "decision": decision,
            "score": round(importance_score, 4),
            "threshold": round(self.threshold, 4),
            "frame_idx": frame_idx,
        }

    def adapt_threshold(self) -> None:
        """Nudge threshold toward target retention rate."""
        if len(self.recent_decisions) < 5:
            return
        current_rate = float(np.mean(self.recent_decisions))
        if current_rate > self.target_retention_rate:
            self.threshold += self.adaptation_rate
        elif current_rate < self.target_retention_rate:
            self.threshold -= self.adaptation_rate
        self.threshold = float(np.clip(self.threshold, 0.1, 0.9))

    def get_stats(self) -> dict:
        current_rate = float(np.mean(self.recent_decisions)) if self.recent_decisions else 0.0
        return {
            "threshold": round(self.threshold, 4),
            "current_retention_rate": round(current_rate, 4),
            "target_rate": self.target_retention_rate,
        }
