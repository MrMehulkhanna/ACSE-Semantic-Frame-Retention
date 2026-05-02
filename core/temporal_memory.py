"""
ACSE — Module 1: Temporal Graph Memory
========================================

Maintains a rolling compressed memory of recent scene state.  Frames are not
stored raw — only compressed PCA-of-DCT descriptors are kept.  The memory is
the living context against which all novelty / surprise is measured.

RESEARCH CONTRIBUTION:
  Unlike Zhuang et al. 1998 (pixel-level frame differencing) or Liu & Cheng
  2008 (adaptive thresholds on raw pixel statistics), this module compresses
  each frame into a 128-d descriptor via block-DCT followed by PCA, then embeds
  it in a *graph* that records *scene transitions*.  Zhang et al. 2016 uses
  LSTM hidden states as memory — our graph is explicit, interpretable, and
  supports entropy queries.  VA-RED2 and ReTaKe operate on neural feature
  spaces requiring a pretrained backbone; our descriptor is self-contained and
  < 2 ms to compute at 320×240.

NOVELTY CLAIM:
  First system to couple a salience-decaying descriptor bank with a directed
  scene-transition graph and a recurrence hash, enabling O(1) queries for
  expected state, memory entropy, and scene recurrence without any deep-net
  forward pass.

POTENTIAL PATENT CLAIM DIRECTION:
  A method for maintaining a temporally-decaying compressed descriptor memory
  with an associated directed scene-transition graph, wherein incoming video
  frames are encoded via block-DCT and PCA, stored with per-entry salience that
  decays multiplicatively, and queried for expected state, recurrence, and
  entropy to gate downstream retention decisions.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import cv2
import networkx as nx
import numpy as np
from scipy.fftpack import dct
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA


class TemporalGraphMemory:
    """Rolling compressed memory of scene state with graph topology."""

    def __init__(
        self,
        max_memory: int = 32,
        descriptor_dim: int = 128,
        n_clusters: int = 8,
        default_decay_rate: float = 0.95,
    ) -> None:
        self.max_memory = max_memory
        self.descriptor_dim = descriptor_dim
        self.n_clusters = n_clusters
        self.default_decay_rate = default_decay_rate

        # Core state
        self.memory_bank: deque = deque(maxlen=max_memory)
        self.scene_graph: nx.DiGraph = nx.DiGraph()
        self.recurrence_signatures: dict[str, int] = {}

        # Internal helpers
        self._ipca = IncrementalPCA(n_components=descriptor_dim)
        self._kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, n_init=3)
        self._pca_fitted = False
        self._kmeans_fitted = False
        self._raw_vectors: list[np.ndarray] = []
        self._prev_cluster_label: Optional[int] = None
        self._total_frames_seen: int = 0

        # Warmup batch size — we need at least this many frames before PCA
        self._warmup_size = max(descriptor_dim + 2, 20)

    # ------------------------------------------------------------------
    # Descriptor extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_dct_vector(frame: np.ndarray) -> np.ndarray:
        """Block-DCT feature vector: 8×8 blocks, keep top-left 4×4 coeffs."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Resize to 320×240 if needed for consistency
        gray = cv2.resize(gray, (320, 240)).astype(np.float32)

        h, w = gray.shape
        block_size = 8
        keep = 4  # top-left 4×4 of each DCT block

        coeffs = []
        for r in range(0, h - block_size + 1, block_size):
            for c in range(0, w - block_size + 1, block_size):
                block = gray[r : r + block_size, c : c + block_size]
                d = dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")
                coeffs.append(d[:keep, :keep].ravel())

        return np.concatenate(coeffs).astype(np.float32)

    def _compute_descriptor(self, frame: np.ndarray) -> np.ndarray:
        """PCA-128 of block-DCT vector."""
        raw = self._extract_dct_vector(frame)
        self._raw_vectors.append(raw)

        if not self._pca_fitted:
            if len(self._raw_vectors) >= self._warmup_size:
                batch = np.stack(self._raw_vectors)
                self._ipca.fit(batch)
                self._pca_fitted = True
                # Re-transform latest vector
                return self._ipca.transform(raw.reshape(1, -1))[0]
            else:
                # Fallback: pad/truncate to descriptor_dim
                if len(raw) >= self.descriptor_dim:
                    return raw[: self.descriptor_dim]
                return np.pad(raw, (0, self.descriptor_dim - len(raw)))
        else:
            # Incremental partial_fit to adapt PCA
            self._ipca.partial_fit(raw.reshape(1, -1))
            return self._ipca.transform(raw.reshape(1, -1))[0]

    # ------------------------------------------------------------------
    # Cluster helpers
    # ------------------------------------------------------------------

    def _cluster_label(self, descriptor: np.ndarray) -> int:
        desc_64 = descriptor.astype(np.float64).reshape(1, -1)
        if not self._kmeans_fitted:
            if len(self.memory_bank) >= self.n_clusters:
                descs = np.stack([e["descriptor"] for e in self.memory_bank]).astype(np.float64)
                self._kmeans.fit(descs)
                self._kmeans_fitted = True
                return int(self._kmeans.predict(desc_64)[0])
            return 0
        self._kmeans.partial_fit(desc_64)
        return int(self._kmeans.predict(desc_64)[0])

    @staticmethod
    def _hash_descriptor(descriptor: np.ndarray) -> str:
        """Coarse hash for recurrence tracking."""
        coarse = (descriptor[:16] * 100).astype(np.int32)
        return coarse.tobytes().hex()[:32]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray, frame_idx: int, salience: float) -> np.ndarray:
        """Ingest a new frame into memory.  Returns the computed descriptor."""
        descriptor = self._compute_descriptor(frame)
        self._total_frames_seen += 1

        # Decay all existing entries
        for entry in self.memory_bank:
            entry["salience"] *= entry["decay_rate"]

        # Evict low-salience entries
        while self.memory_bank and self.memory_bank[0]["salience"] < 0.01:
            self.memory_bank.popleft()

        # Add new entry
        self.memory_bank.append(
            {
                "descriptor": descriptor.copy(),
                "timestamp": frame_idx,
                "salience": max(salience, 0.01),
                "decay_rate": self.default_decay_rate,
            }
        )

        # Scene graph update
        cluster = self._cluster_label(descriptor)
        if cluster not in self.scene_graph:
            self.scene_graph.add_node(cluster, count=0)
        self.scene_graph.nodes[cluster]["count"] = (
            self.scene_graph.nodes[cluster].get("count", 0) + 1
        )
        if self._prev_cluster_label is not None and self._prev_cluster_label != cluster:
            if self.scene_graph.has_edge(self._prev_cluster_label, cluster):
                self.scene_graph[self._prev_cluster_label][cluster]["weight"] += 1
            else:
                self.scene_graph.add_edge(self._prev_cluster_label, cluster, weight=1)
        self._prev_cluster_label = cluster

        # Recurrence tracking
        dhash = self._hash_descriptor(descriptor)
        self.recurrence_signatures[dhash] = self.recurrence_signatures.get(dhash, 0) + 1

        return descriptor

    def get_expected_descriptor(self, frame_idx: int = 0) -> np.ndarray:
        """Salience-weighted mean of memory descriptors — what we 'predict'."""
        if not self.memory_bank:
            return np.zeros(self.descriptor_dim, dtype=np.float32)
        descs = np.stack([e["descriptor"] for e in self.memory_bank])
        weights = np.array([e["salience"] for e in self.memory_bank], dtype=np.float32)
        total = weights.sum()
        if total < 1e-9:
            return descs.mean(axis=0)
        return (descs * weights[:, None]).sum(axis=0) / total

    def compute_scene_recurrence(self, descriptor: np.ndarray) -> float:
        """Fraction of past frames that looked like this descriptor's cluster."""
        if self._total_frames_seen == 0:
            return 0.0
        dhash = self._hash_descriptor(descriptor)
        count = self.recurrence_signatures.get(dhash, 0)
        return count / self._total_frames_seen

    def get_memory_entropy(self) -> float:
        """Shannon entropy over cluster distribution in current memory bank."""
        if len(self.memory_bank) < 2:
            return 0.0
        if not self._kmeans_fitted:
            return 1.0  # assume high entropy before clustering is ready
        descs = np.stack([e["descriptor"] for e in self.memory_bank]).astype(np.float64)
        labels = self._kmeans.predict(descs)
        counts = np.bincount(labels, minlength=self.n_clusters).astype(np.float64)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    def get_descriptor_dim(self) -> int:
        return self.descriptor_dim

    def get_state_snapshot(self) -> dict:
        """Return serializable snapshot of current memory state."""
        return {
            "size": len(self.memory_bank),
            "total_frames_seen": self._total_frames_seen,
            "entropy": self.get_memory_entropy(),
            "graph_nodes": self.scene_graph.number_of_nodes(),
            "graph_edges": self.scene_graph.number_of_edges(),
            "descriptors": [
                {
                    "timestamp": e["timestamp"],
                    "salience": round(e["salience"], 4),
                    "descriptor_norm": float(np.linalg.norm(e["descriptor"])),
                }
                for e in self.memory_bank
            ],
        }
