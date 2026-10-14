"""
ACSE — Module 8a: Resource-Aware Weighter
===========================================

Provides real-time system load monitoring to modulate frame retention.
Under heavy CPU load, only strongly novel frames pass through.

Related approaches:
  No prior video summarisation or frame selection system dynamically adjusts
  its retention threshold based on real-time system resource utilisation.
  Zhuang 1998, Liu & Cheng 2008, Zhang et al. 2016 assume unlimited compute.
  VA-RED2 targets a fixed FLOP budget.  ReTaKe has no resource awareness.
  Our resource-aware weighter continuously monitors CPU and memory load
  and multiplicatively scales the importance score, creating graceful
  quality degradation under load.
"""

from collections import deque

import numpy as np
import psutil


class ResourceAwareWeighter:
    """Monitors system resources and provides scaling weights."""

    def __init__(self, history_len: int = 100) -> None:
        self.cpu_history: deque = deque(maxlen=history_len)
        self.memory_history: deque = deque(maxlen=history_len)

    def get_raw(self) -> float:
        """RAW = 1 - CPU% / 100, clipped to [0.05, 1.0]."""
        try:
            cpu = psutil.cpu_percent(interval=None)
        except Exception:
            cpu = 0.0
        self.cpu_history.append(cpu)

        try:
            mem = psutil.virtual_memory().percent
        except Exception:
            mem = 0.0
        self.memory_history.append(mem)

        return float(np.clip(1.0 - cpu / 100.0, 0.05, 1.0))

    def get_stats(self) -> dict:
        return {
            "current_cpu": self.cpu_history[-1] if self.cpu_history else 0.0,
            "mean_cpu": float(np.mean(self.cpu_history)) if self.cpu_history else 0.0,
            "current_memory": self.memory_history[-1] if self.memory_history else 0.0,
            "mean_memory": float(np.mean(self.memory_history)) if self.memory_history else 0.0,
        }
