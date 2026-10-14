"""
ACSE — Module 7: Online Bayesian Updater (Adaptive Learning)
==============================================================

Learns which factors are most predictive of "truly important" frames,
updating fusion weights continuously as new feedback arrives.

Related approaches:
  Zhuang 1998 and Liu & Cheng 2008 use fixed or heuristically-tuned
  thresholds.  Zhang et al. 2016 learns offline via supervised LSTM training.
  VA-RED2 trains a policy network offline then freezes.  ReTaKe does not
  adapt its redundancy metric over time.
  Our updater receives per-frame reward signals (self-supervised or downstream)
  and applies gradient-based weight updates with softmax normalisation, so the
  fusion weights continuously adapt to the content characteristics of the
  current stream.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import pandas as pd


class OnlineBayesianUpdater:
    """Online weight learner with softmax-normalised gradient updates."""

    def __init__(
        self,
        initial_weights: dict | None = None,
        alpha: float = 0.05,
        history_len: int = 200,
    ) -> None:
        self.weights: dict = initial_weights or {
            "cni": 0.4,
            "epd": 0.2,
            "sss": 0.3,
            "pig": 0.1,
        }
        self.alpha: float = alpha
        self.weight_history: deque = deque(maxlen=history_len)

        # Internal logits for stable softmax updates
        self._logits: dict = {}
        for k, v in self.weights.items():
            self._logits[k] = float(np.log(v + 1e-8))

        self.weight_history.append(self.weights.copy())

    @staticmethod
    def _softmax(logits: dict) -> dict:
        keys = list(logits.keys())
        vals = np.array([logits[k] for k in keys], dtype=np.float64)
        vals -= vals.max()  # numerical stability
        exp_vals = np.exp(vals)
        total = exp_vals.sum()
        return {k: float(exp_vals[i] / total) for i, k in enumerate(keys)}

    def update(self, fusion_output: dict, reward: float) -> None:
        """
        Update weights based on reward signal.

        Args:
            fusion_output: Dict with factor values (cni, epd, sss, pig).
            reward: +1 if frame was truly important, -1 if redundant.
        """
        for f in self._logits:
            factor_val = fusion_output.get(f, 0.0)
            gradient = reward * factor_val
            self._logits[f] += self.alpha * gradient

        self.weights = self._softmax(self._logits)
        self.weight_history.append(self.weights.copy())

    def get_weights(self) -> dict:
        return self.weights.copy()

    def get_weight_history(self) -> pd.DataFrame:
        records = list(self.weight_history)
        df = pd.DataFrame(records)
        df.index.name = "step"
        return df
