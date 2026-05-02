"""
ACSE — Synthetic Video Dataset Builder
=========================================

Generates three scene types (STATIC_AMBIENT, DYNAMIC_TRACKING, MIXED_TRANSITION)
of 500 frames each, with ground-truth importance labels.  Also saves sample
frames as PNG.

Each scene type has carefully designed label logic so that NOT all motion is
labelled important — only semantically significant events get label=1.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Ensure project root is importable
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


class SyntheticVideoDatasetBuilder:
    """Generate controlled synthetic video frames with ground-truth labels."""

    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        frames_per_scene: int = 500,
        output_dir: str = "dataset",
        seed: int = 42,
    ) -> None:
        self.width = width
        self.height = height
        self.frames_per_scene = frames_per_scene
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Scene generators
    # ------------------------------------------------------------------

    def _generate_static_ambient(self) -> list[tuple[np.ndarray, int]]:
        """Fixed background + noise, occasional brightness steps and blur spikes."""
        # Generate a textured base background
        base = self.rng.randint(60, 180, (self.height, self.width), dtype=np.uint8)
        base = cv2.GaussianBlur(base, (15, 15), 5)

        frames_labels = []
        brightness_offset = 0
        next_brightness_event = self.rng.randint(60, 100)

        for i in range(self.frames_per_scene):
            label = 0
            frame = base.astype(np.float32) + brightness_offset

            # Random Gaussian noise
            sigma = self.rng.uniform(1, 6)
            noise = self.rng.normal(0, sigma, (self.height, self.width)).astype(np.float32)
            frame += noise

            # Brightness step event
            if i == next_brightness_event:
                delta = self.rng.randint(-40, 41)
                brightness_offset += delta
                label = 1
                next_brightness_event = i + self.rng.randint(60, 100)

            # Blur spike (5% chance)
            if self.rng.random() < 0.05:
                k = self.rng.choice([5, 7])
                frame = cv2.GaussianBlur(frame, (k, k), 0)
                label = 1

            frame = np.clip(frame, 0, 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frames_labels.append((frame_bgr, label))

        return frames_labels

    def _generate_dynamic_tracking(self) -> list[tuple[np.ndarray, int]]:
        """Animated objects on sine paths with appearance/disappearance/occlusion."""
        bg = self.rng.randint(40, 80, (self.height, self.width, 3), dtype=np.uint8)
        bg = cv2.GaussianBlur(bg, (9, 9), 3)

        # Object state
        n_objects = 2
        obj_active = [True, True]
        obj_phase = [self.rng.uniform(0, 2 * np.pi) for _ in range(n_objects)]
        obj_freq = [self.rng.uniform(0.02, 0.06) for _ in range(n_objects)]
        obj_amp_x = [self.rng.uniform(40, 120) for _ in range(n_objects)]
        obj_amp_y = [self.rng.uniform(30, 80) for _ in range(n_objects)]
        obj_cx = [self.width // 3, 2 * self.width // 3]
        obj_cy = [self.height // 2, self.height // 2]
        obj_color = [(0, 200, 255), (255, 100, 0)]
        obj_radius = [18, 22]

        next_event_frame = self.rng.randint(60, 120)
        event_type_queue = []

        frames_labels = []

        for i in range(self.frames_per_scene):
            label = 0
            frame = bg.copy()

            # Compute object positions
            positions = []
            for o in range(n_objects):
                if obj_active[o]:
                    x = int(obj_cx[o] + obj_amp_x[o] * np.sin(obj_freq[o] * i + obj_phase[o]))
                    y = int(obj_cy[o] + obj_amp_y[o] * np.cos(obj_freq[o] * i * 0.7 + obj_phase[o]))
                    x = np.clip(x, obj_radius[o], self.width - obj_radius[o])
                    y = np.clip(y, obj_radius[o], self.height - obj_radius[o])
                    positions.append((x, y))
                    cv2.circle(frame, (x, y), obj_radius[o], obj_color[o], -1)
                else:
                    positions.append(None)

            # Occlusion detection
            if positions[0] is not None and positions[1] is not None:
                dist = np.sqrt(
                    (positions[0][0] - positions[1][0]) ** 2
                    + (positions[0][1] - positions[1][1]) ** 2
                )
                if dist < (obj_radius[0] + obj_radius[1]):
                    label = 1  # occlusion event

            # Appearance / disappearance events
            if i == next_event_frame:
                target = self.rng.randint(0, n_objects)
                if obj_active[target]:
                    obj_active[target] = False
                    label = 1  # disappearance
                else:
                    obj_active[target] = True
                    label = 1  # appearance
                next_event_frame = i + self.rng.randint(60, 120)

            frames_labels.append((frame, label))

        return frames_labels

    def _generate_mixed_transition(self) -> list[tuple[np.ndarray, int]]:
        """Alternating static/dynamic with crossfade transitions and camera jitter."""
        static_bg = self.rng.randint(100, 160, (self.height, self.width, 3), dtype=np.uint8)
        static_bg = cv2.GaussianBlur(static_bg, (11, 11), 4)

        dynamic_bg = self.rng.randint(30, 70, (self.height, self.width, 3), dtype=np.uint8)
        dynamic_bg = cv2.GaussianBlur(dynamic_bg, (11, 11), 4)

        frames_labels = []
        mode = "static"
        segment_len = self.rng.randint(30, 50)
        crossfade_duration = 5
        frames_in_segment = 0
        transitioning = False
        transition_progress = 0

        # Moving object for dynamic segments
        obj_x, obj_y = self.width // 2, self.height // 2
        obj_vx, obj_vy = 3, 2

        for i in range(self.frames_per_scene):
            label = 0
            frames_in_segment += 1

            # Check for segment transition
            if frames_in_segment >= segment_len and not transitioning:
                transitioning = True
                transition_progress = 0
                label = 1  # transition start

            if transitioning:
                alpha = transition_progress / crossfade_duration
                alpha = np.clip(alpha, 0.0, 1.0)

                if mode == "static":
                    frame = cv2.addWeighted(static_bg, 1 - alpha, dynamic_bg, alpha, 0)
                else:
                    frame = cv2.addWeighted(dynamic_bg, 1 - alpha, static_bg, alpha, 0)

                transition_progress += 1
                label = 1  # transition frame

                if transition_progress > crossfade_duration:
                    transitioning = False
                    mode = "dynamic" if mode == "static" else "static"
                    segment_len = (
                        self.rng.randint(15, 30) if mode == "dynamic"
                        else self.rng.randint(30, 50)
                    )
                    frames_in_segment = 0
            else:
                if mode == "static":
                    frame = static_bg.copy()
                    noise = self.rng.normal(0, 2, frame.shape).astype(np.float32)
                    frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                else:
                    frame = dynamic_bg.copy()
                    # Moving object
                    obj_x += obj_vx
                    obj_y += obj_vy
                    if obj_x < 20 or obj_x > self.width - 20:
                        obj_vx *= -1
                    if obj_y < 20 or obj_y > self.height - 20:
                        obj_vy *= -1
                    obj_x = np.clip(obj_x, 20, self.width - 20)
                    obj_y = np.clip(obj_y, 20, self.height - 20)
                    cv2.rectangle(
                        frame,
                        (obj_x - 15, obj_y - 15),
                        (obj_x + 15, obj_y + 15),
                        (0, 255, 150),
                        -1,
                    )

            # Camera jitter
            dx, dy = self.rng.randint(-3, 4), self.rng.randint(-3, 4)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            frame = cv2.warpAffine(frame, M, (self.width, self.height))

            frames_labels.append((frame, label))

        return frames_labels

    # ------------------------------------------------------------------
    # Full pipeline: generate frames, run ACSE, save CSV
    # ------------------------------------------------------------------

    def build(self) -> pd.DataFrame:
        """Build dataset: generate all scenes, run pipeline, save CSV + PNGs."""
        scenes = {
            "STATIC_AMBIENT": self._generate_static_ambient,
            "DYNAMIC_TRACKING": self._generate_dynamic_tracking,
            "MIXED_TRANSITION": self._generate_mixed_transition,
        }

        all_records = []
        sample_frames_dir = self.output_dir / "sample_frames"
        sample_frames_dir.mkdir(parents=True, exist_ok=True)

        global_idx = 0

        for scene_name, generator in scenes.items():
            print(f"  Generating scene: {scene_name}")
            frames_labels = generator()

            # Save 3 sample frames
            sample_indices = [0, len(frames_labels) // 2, len(frames_labels) - 1]
            for si in sample_indices:
                fname = f"{scene_name.lower()}_frame_{si}.png"
                cv2.imwrite(str(sample_frames_dir / fname), frames_labels[si][0])

            # Run ACSE pipeline on this scene
            memory = TemporalGraphMemory()
            cni_engine = CognitiveNoveltyIndex()
            epd_engine = EventPersistenceDecay()
            sss_engine = SemanticSurpriseScore()
            fusion = MultiFactorFusion()
            gate = AdaptiveThresholdGate(target_retention_rate=0.30)

            prev_frame = None
            for local_idx, (frame, label) in enumerate(frames_labels):
                # Compute descriptor via memory update (low salience initially)
                desc = memory.update(frame, global_idx, salience=0.5)

                # CNI
                cni = cni_engine.compute(desc, memory)

                # EPD
                epd_engine.update(frame, prev_frame, global_idx)
                epd = epd_engine.compute_epd_score(global_idx)

                # SSS
                sss_engine.update_predictor(desc)
                sss = sss_engine.compute(desc)

                # Fusion
                fused = fusion.fuse(cni, epd, sss, memory, global_idx)
                importance = fused["importance_score"]

                # Gate
                gate_result = gate.decide(importance, global_idx)

                all_records.append(
                    {
                        "frame_idx": global_idx,
                        "scene_type": scene_name,
                        "label": label,
                        "cni": round(cni, 4),
                        "epd": round(epd, 4),
                        "sss": round(sss, 4),
                        "pig": round(fused["pig"], 4),
                        "raw": round(fused["raw"], 4),
                        "importance_score": round(importance, 4),
                    }
                )

                prev_frame = frame
                global_idx += 1

        df = pd.DataFrame(all_records)
        csv_path = self.output_dir / "saml_frames.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Dataset saved: {csv_path}  ({len(df)} rows)")
        return df


# Allow standalone execution
if __name__ == "__main__":
    builder = SyntheticVideoDatasetBuilder(
        output_dir=str(PROJECT_ROOT / "dataset"),
    )
    df = builder.build()
    print(df.head(20))
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
