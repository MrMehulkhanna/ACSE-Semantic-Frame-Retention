"""
ACSE — Module 2: Cognitive Novelty Index (CNI)
================================================

Quantifies how much *new information* a frame carries relative to what the
temporal memory already contains.  This is the main factor — not pixel
difference, not motion.  It is the cosine distance between the frame's
descriptor and the memory's expected descriptor, normalised by memory entropy.

Related approaches:
  Zhuang et al. 1998 uses absolute pixel difference — no semantic compression.
  Liu & Cheng 2008 adapts thresholds but still on low-level features.  Zhang
  et al. 2016 uses LSTM hidden-state distance — requires a large pretrained
  network and GPU.  VA-RED2 gates computation but does not measure *informational
  novelty* against a decaying memory.  ReTaKe measures knowledge redundancy in
  token space, not in compressed visual descriptors.
  Our CNI operates on 128-d PCA-DCT descriptors and normalises by memory entropy,
  so the score self-calibrates: in a diverse memory the bar for "novel" is higher.
"""

import numpy as np

from core.temporal_memory import TemporalGraphMemory


class CognitiveNoveltyIndex:
    """Entropy-normalised cosine novelty against temporal memory."""

    def __init__(self) -> None:
        self._eps: float = 1e-6

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 1.0
        return float(1.0 - np.dot(a, b) / (norm_a * norm_b))

    def compute(self, descriptor: np.ndarray, memory: TemporalGraphMemory) -> float:
        """CNI(t) = cosine_dist(D_t, E_t) / (H(M_t) + eps), clipped to [0,1]."""
        expected = memory.get_expected_descriptor()
        raw_distance = self._cosine_distance(descriptor, expected)
        entropy_factor = memory.get_memory_entropy() + self._eps
        cni = raw_distance / entropy_factor
        return float(np.clip(cni, 0.0, 1.0))

    def explain(self, descriptor: np.ndarray, memory: TemporalGraphMemory) -> dict:
        """Return breakdown of CNI computation for explainability."""
        expected = memory.get_expected_descriptor()
        raw_distance = self._cosine_distance(descriptor, expected)
        entropy_factor = memory.get_memory_entropy() + self._eps
        cni = float(np.clip(raw_distance / entropy_factor, 0.0, 1.0))

        # Find dominant memory entry (highest salience)
        dominant_idx = -1
        max_sal = -1.0
        for i, entry in enumerate(memory.memory_bank):
            if entry["salience"] > max_sal:
                max_sal = entry["salience"]
                dominant_idx = i

        return {
            "raw_distance": round(raw_distance, 6),
            "entropy_factor": round(entropy_factor, 6),
            "dominant_memory_entry": dominant_idx,
            "cni": round(cni, 6),
        }
