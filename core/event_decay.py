"""
ACSE — Module 3: Event Persistence Decay (EPD)
=================================================

Models how long a detected "event" remains relevant after it first appears.
A moving object that entered 3 frames ago is still "active" — continuation
frames are LESS novel than the arrival frame.  EPD penalises frames that merely
continue an already-detected event.

Related approaches:
  Frame-differencing systems (Zhuang 1998) treat every motion frame equally.
  Liu & Cheng 2008 adapts thresholds but has no concept of *event lifetime*.
  Zhang et al. 2016 LSTM summarisation implicitly decays through hidden state
  but cannot attribute which event is "still ongoing."  VA-RED2 gates
  computation without temporal event tracking.  ReTaKe measures redundancy in
  a token window, not across named events.
  Our EPD explicitly tracks discrete events with exponential persistence decay,
  so frames that continue a known event are down-weighted in a principled,
  interpretable way.
"""

from __future__ import annotations

import hashlib
from collections import deque
from math import exp
from typing import Optional

import cv2
import numpy as np


class EventPersistenceDecay:
    """Track discrete events and suppress continuation frames."""

    def __init__(self, decay_lambda: float = 0.15) -> None:
        self.active_events: dict[str, dict] = {}
        self.decay_lambda: float = decay_lambda

        # Background subtractor
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=40, detectShadows=False
        )
        self._prev_contour_set: set[str] = set()
        self._min_contour_area: int = 200

    @staticmethod
    def _spatial_hash(cx: int, cy: int, grid: int = 40) -> str:
        """Coarse spatial hash for event identification."""
        gx, gy = cx // grid, cy // grid
        return f"{gx}_{gy}"

    def _contour_ids(self, contours: list) -> set[str]:
        ids = set()
        for c in contours:
            if cv2.contourArea(c) < self._min_contour_area:
                continue
            m = cv2.moments(c)
            if m["m00"] < 1:
                continue
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            ids.add(self._spatial_hash(cx, cy))
        return ids

    def detect_events(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray],
        frame_idx: int,
    ) -> list[dict]:
        """Detect motion, appearance, disappearance events."""
        events: list[dict] = []

        # Ensure 3-channel for MOG2
        if len(frame.shape) == 2:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_bgr = frame

        small = cv2.resize(frame_bgr, (320, 240))
        fg_mask = self._bg_sub.apply(small)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_ids = self._contour_ids(contours)

        # Motion events — every detected contour region
        for cid in current_ids:
            events.append(
                {
                    "event_id": f"motion_{cid}",
                    "first_seen": frame_idx,
                    "last_seen": frame_idx,
                    "persistence": 1.0,
                    "event_type": "motion",
                }
            )

        # Appearance events — new contour regions not in previous frame
        appeared = current_ids - self._prev_contour_set
        for cid in appeared:
            events.append(
                {
                    "event_id": f"appear_{cid}",
                    "first_seen": frame_idx,
                    "last_seen": frame_idx,
                    "persistence": 1.0,
                    "event_type": "appearance",
                }
            )

        # Disappearance events — old contour regions no longer present
        disappeared = self._prev_contour_set - current_ids
        for cid in disappeared:
            events.append(
                {
                    "event_id": f"disappear_{cid}",
                    "first_seen": frame_idx,
                    "last_seen": frame_idx,
                    "persistence": 1.0,
                    "event_type": "disappearance",
                }
            )

        self._prev_contour_set = current_ids
        return events

    def update(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray],
        frame_idx: int,
    ) -> None:
        """Detect events and update persistence state."""
        detected = self.detect_events(frame, prev_frame, frame_idx)
        detected_ids = {e["event_id"] for e in detected}

        # Update or create entries
        for evt in detected:
            eid = evt["event_id"]
            if eid in self.active_events:
                entry = self.active_events[eid]
                entry["last_seen"] = frame_idx
                dt = frame_idx - entry["first_seen"]
                entry["persistence"] = exp(-self.decay_lambda * dt)
            else:
                self.active_events[eid] = {
                    "event_id": eid,
                    "first_seen": frame_idx,
                    "last_seen": frame_idx,
                    "persistence": 1.0,
                    "event_type": evt["event_type"],
                }

        # Decay events not detected this frame
        stale = []
        for eid, entry in self.active_events.items():
            if eid not in detected_ids:
                entry["persistence"] *= exp(-self.decay_lambda)
                if entry["persistence"] < 0.05:
                    stale.append(eid)

        for eid in stale:
            del self.active_events[eid]

    def compute_epd_score(self, frame_idx: int) -> float:
        """Mean persistence of active events — subtracted from novelty."""
        if not self.active_events:
            return 0.0
        persistences = [e["persistence"] for e in self.active_events.values()]
        return float(np.mean(persistences))

    def get_active_event_count(self) -> int:
        return len(self.active_events)
