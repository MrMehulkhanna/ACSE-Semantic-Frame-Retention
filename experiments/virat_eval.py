"""
experiments/virat_eval.py
==========================
VIRAT Aerial dataset evaluator for the ACSE pipeline.

ZERO changes to any existing code. This file only READS the VIRAT videos
from dataset/VIRAT-Aerial/, runs the existing ACSE modules unchanged,
and writes all evaluation artefacts to screenshots/ so they appear
automatically when the notebook runs.

Usage
-----
Run from the notebook (Cell 14) or standalone:
    python experiments/virat_eval.py

Expected folder layout
----------------------
acse_project/
  dataset/
    VIRAT-Aerial/          ← put your .mp4 / .avi / .mov files here
      *.mp4
      ...
  screenshots/             ← all output PNGs are written here
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
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Import existing ACSE modules exactly as they are ──────────────────────────
from core.temporal_memory import TemporalGraphMemory
from core.novelty_engine import CognitiveNoveltyIndex
from core.event_decay import EventPersistenceDecay
from core.surprise_engine import SemanticSurpriseScore
from core.semantic_router import MultiFactorFusion
from core.adaptive_learning import OnlineBayesianUpdater
from core.decision_gate import AdaptiveThresholdGate
from core.explanation_engine import SHAPTraceExplainer

VIRAT_DIR   = PROJECT_ROOT / "dataset" / "VIRAT-Aerial"
SCREENS_DIR = PROJECT_ROOT / "screenshots"
SCREENS_DIR.mkdir(parents=True, exist_ok=True)

# ── Styling (dark, consistent with existing project graphs) ───────────────────
BG, PANEL = "#0d1117", "#161b22"
BORDER, TXT, MUTED = "#30363d", "#e6edf3", "#8b949e"
GREEN, BLUE = "#3fb950", "#58a6ff"
ORANGE, PURPLE = "#f78166", "#bc8cff"
YELLOW, TEAL = "#e3b341", "#39d353"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": BORDER, "axes.labelcolor": TXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TXT,
    "grid.color": BORDER, "grid.linestyle": "--", "grid.linewidth": 0.5,
    "font.family": "monospace", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "legend.facecolor": PANEL, "legend.edgecolor": BORDER,
})


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — Video scanning utilities
# ═════════════════════════════════════════════════════════════════════════════

def _find_videos(folder: Path) -> list[Path]:
    """Return all video files in folder (recursive)."""
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}
    videos = []
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in exts:
            videos.append(p)
    return videos


def _auto_label_frame(frame: np.ndarray, prev_frame: np.ndarray | None,
                       fds_thresh: float = 5.0, map_thresh: float = 2.0) -> int:
    """
    Rule-based label for VIRAT frames (no manual annotations needed).
    Important (1) if frame difference score or motion area percentage exceeds
    empirically validated thresholds.  Same logic used in dataset_builder.py.
    """
    if prev_frame is None:
        return 0
    g  = cv2.cvtColor(frame,      cv2.COLOR_BGR2GRAY).astype(np.float32)
    pg = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = np.abs(g - pg)
    fds = float(np.mean(diff))
    map_pct = float(np.sum(diff > 25) / diff.size * 100)
    return 1 if (fds > fds_thresh or map_pct > map_thresh) else 0


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — Core ACSE inference on one video
# ═════════════════════════════════════════════════════════════════════════════

def run_acse_on_video(
    video_path: Path,
    max_frames: int = 1500,
    target_retention: float = 0.30,
    sample_every: int = 1,
) -> pd.DataFrame:
    """
    Run the full ACSE pipeline on a single video file.
    Returns a DataFrame with one row per processed frame:
      frame_idx, label, cni, epd, sss, pig, raw, importance_score,
      decision, threshold, latency_ms
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Instantiate ACSE modules fresh for each video (clean state)
    memory    = TemporalGraphMemory()
    cni_eng   = CognitiveNoveltyIndex()
    epd_eng   = EventPersistenceDecay()
    sss_eng   = SemanticSurpriseScore()
    fusion    = MultiFactorFusion()
    updater   = OnlineBayesianUpdater()
    gate      = AdaptiveThresholdGate(target_retention_rate=target_retention)
    explainer = SHAPTraceExplainer(fusion)

    records   = []
    prev_frame = None
    frame_idx  = 0
    read_idx   = 0

    while len(records) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        read_idx += 1

        # Sub-sampling: process every Nth frame
        if read_idx % sample_every != 0:
            continue

        label = _auto_label_frame(frame, prev_frame)

        t0   = time.perf_counter()
        desc = memory.update(frame, frame_idx, salience=0.5)
        cni  = cni_eng.compute(desc, memory)
        epd_eng.update(frame, prev_frame, frame_idx)
        epd  = epd_eng.compute_epd_score(frame_idx)
        sss_eng.update_predictor(desc)
        sss  = sss_eng.compute(desc)
        fused = fusion.fuse(cni, epd, sss, memory, frame_idx)
        fusion.set_weights(updater.get_weights())
        gate_res = gate.decide(fused["importance_score"], frame_idx)
        latency  = (time.perf_counter() - t0) * 1000  # ms

        reward = +1.0 if gate_res["decision"] == "RETAIN" else -0.5
        updater.update(fused, reward)

        records.append({
            "frame_idx":        frame_idx,
            "label":            label,
            "cni":              round(cni, 4),
            "epd":              round(epd, 4),
            "sss":              round(sss, 4),
            "pig":              round(fused["pig"], 4),
            "raw":              round(fused["raw"], 4),
            "importance_score": round(fused["importance_score"], 4),
            "decision":         gate_res["decision"],
            "threshold":        round(gate_res["threshold"], 4),
            "latency_ms":       round(latency, 3),
            "weights_cni":      round(updater.get_weights()["cni"], 4),
            "weights_epd":      round(updater.get_weights()["epd"], 4),
            "weights_sss":      round(updater.get_weights()["sss"], 4),
            "weights_pig":      round(updater.get_weights()["pig"], 4),
        })
        prev_frame = frame
        frame_idx  += 1

    cap.release()
    return pd.DataFrame(records)


# ═════════════════════════════════════════════════════════════════════════════
# PART 3 — All evaluation plots saved to screenshots/
# ═════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, name: str) -> Path:
    path = SCREENS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  [saved] {path.name}")
    return path


def plot_per_video_summary(summary: list[dict]) -> None:
    """Bar chart of accuracy, precision, recall, F1 per video file."""
    if not summary:
        return
    names   = [s["video"] for s in summary]
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors  = [GREEN, BLUE, ORANGE, PURPLE]
    x = np.arange(len(names))
    w = 0.18

    fig, ax = plt.subplots(figsize=(max(10, len(names)*2), 5), facecolor=BG)
    fig.suptitle("VIRAT-Aerial — Per-Video Classification Metrics",
                 fontsize=14, color=TXT, fontweight="bold")
    for i, (m, c) in enumerate(zip(metrics, colors)):
        vals = [s.get(m, 0) for s in summary]
        ax.bar(x + i*w, vals, w, color=c, alpha=0.85,
               edgecolor=BORDER, label=m.capitalize())
    ax.set_xticks(x + 1.5*w)
    ax.set_xticklabels([n[:28] for n in names], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(axis="y", alpha=0.4)
    _save(fig, "virat_01_per_video_metrics.png")


def plot_score_timeline(df: pd.DataFrame, video_name: str) -> None:
    """Importance score + adaptive threshold over frames."""
    fig, ax = plt.subplots(figsize=(14, 4), facecolor=BG)
    ax.plot(df["frame_idx"], df["importance_score"],
            color=BLUE, linewidth=1, alpha=0.7, label="Importance Score")
    ax.plot(df["frame_idx"], df["threshold"],
            color=ORANGE, linewidth=1.8, linestyle="--", label="Adaptive Threshold")
    retain = df[df["decision"] == "RETAIN"]
    ax.scatter(retain["frame_idx"], retain["importance_score"],
               color=GREEN, s=8, zorder=3, label="RETAIN", alpha=0.6)
    # Ground truth important frames
    important = df[df["label"] == 1]
    ax.scatter(important["frame_idx"], [-0.05]*len(important),
               color=YELLOW, s=6, marker="|", label="True Important", alpha=0.5)
    ax.set_xlabel("Frame Index"); ax.set_ylabel("Score")
    ax.set_title(f"VIRAT-Aerial — Score Timeline: {video_name[:50]}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(-0.1, 1.05)
    _save(fig, f"virat_02_timeline_{video_name[:30].replace(' ','_')}.png")


def plot_factor_distributions(df: pd.DataFrame, video_name: str) -> None:
    """Distribution of CNI, EPD, SSS, PIG split by label."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor=BG)
    fig.suptitle(f"Factor Distributions — {video_name[:50]}",
                 fontsize=13, color=TXT, fontweight="bold")
    feats = [("cni","CNI",BLUE), ("epd","EPD",ORANGE),
             ("sss","SSS",PURPLE), ("pig","PIG",YELLOW)]
    for ax, (col, title, clr) in zip(axes, feats):
        for lbl, c, nm in [(0, ORANGE, "Redundant"), (1, GREEN, "Important")]:
            vals = df[df["label"]==lbl][col]
            ax.hist(vals, bins=35, alpha=0.6, color=c, label=nm,
                    edgecolor="none", density=True)
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    _save(fig, f"virat_03_factor_dist_{video_name[:30].replace(' ','_')}.png")


def plot_confusion_matrix_combined(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   title: str = "All Videos") -> None:
    """Confusion matrix across all VIRAT videos combined."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
    sns.heatmap(cm, annot=True, fmt="d",
                cmap=sns.color_palette("mako", as_cmap=True), ax=ax,
                xticklabels=["DISCARD","RETAIN"],
                yticklabels=["Redundant","Important"],
                linewidths=0.5, linecolor=BG,
                annot_kws={"size":14,"color":"white"})
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"ACSE on VIRAT-Aerial — Confusion Matrix\n{title}")
    _save(fig, "virat_04_confusion_matrix.png")


def plot_roc_curve(y_true: np.ndarray,
                   y_scores: np.ndarray) -> None:
    """ROC curve for combined VIRAT evaluation."""
    if len(np.unique(y_true)) < 2:
        print("  [skip] ROC: only one class in labels")
        return
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
    ax.plot(fpr, tpr, color=GREEN, linewidth=2.5,
            label=f"ACSE (AUC = {auc:.4f})")
    ax.plot([0,1],[0,1],"--", color=MUTED, linewidth=1.2, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ACSE on VIRAT-Aerial — ROC Curve")
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, "virat_05_roc_curve.png")


def plot_retention_by_video(summary: list[dict]) -> None:
    """Actual vs target retention rate per video."""
    if not summary:
        return
    names     = [s["video"][:28] for s in summary]
    actual    = [s.get("retention_rate", 0) for s in summary]
    target    = summary[0].get("target_rate", 0.30)
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(max(9, len(names)*2), 4), facecolor=BG)
    bars = ax.bar(x, actual, color=TEAL, alpha=0.85,
                  edgecolor=BORDER, width=0.55, label="Actual retention")
    ax.axhline(target, color=ORANGE, linewidth=2,
               linestyle="--", label=f"Target ({target*100:.0f}%)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Retention Rate"); ax.set_ylim(0, 1.05)
    ax.set_title("VIRAT-Aerial — Actual vs Target Retention Rate")
    ax.legend(); ax.grid(axis="y", alpha=0.4)
    for bar, v in zip(bars, actual):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.012,
                f"{v*100:.1f}%", ha="center", fontsize=9, color=TXT)
    _save(fig, "virat_06_retention_rates.png")


def plot_latency_profile(all_latencies: list[float]) -> None:
    """Histogram + CDF of per-frame pipeline latency."""
    arr = np.array(all_latencies)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)
    fig.suptitle("ACSE on VIRAT-Aerial — Per-Frame Latency Profile",
                 fontsize=13, color=TXT, fontweight="bold")

    axes[0].hist(arr, bins=50, color=BLUE, alpha=0.85, edgecolor=BORDER)
    axes[0].axvline(np.mean(arr), color=ORANGE, linewidth=2,
                    linestyle="--", label=f"Mean={np.mean(arr):.2f}ms")
    axes[0].axvline(np.percentile(arr, 95), color=YELLOW, linewidth=1.8,
                    linestyle=":", label=f"p95={np.percentile(arr,95):.2f}ms")
    axes[0].set_xlabel("Latency (ms)"); axes[0].set_ylabel("Count")
    axes[0].set_title("Latency Distribution"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    sorted_lat = np.sort(arr)
    cdf = np.arange(1, len(sorted_lat)+1) / len(sorted_lat)
    axes[1].plot(sorted_lat, cdf*100, color=GREEN, linewidth=2.5)
    axes[1].axvline(1.0, color=ORANGE, linewidth=1.8, linestyle="--", label="1ms target")
    axes[1].set_xlabel("Latency (ms)"); axes[1].set_ylabel("CDF (%)")
    axes[1].set_title("Cumulative Latency Distribution")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    _save(fig, "virat_07_latency_profile.png")


def plot_weight_evolution(all_dfs: list[pd.DataFrame],
                          video_names: list[str]) -> None:
    """How adaptive weights evolved over all frames across all videos."""
    if not all_dfs:
        return
    combined = pd.concat(all_dfs, ignore_index=True)
    weight_cols = ["weights_cni","weights_epd","weights_sss","weights_pig"]
    colors_w = [BLUE, ORANGE, GREEN, PURPLE]
    labels_w = ["CNI","EPD","SSS","PIG"]

    fig, ax = plt.subplots(figsize=(14, 4), facecolor=BG)
    for col, clr, lbl in zip(weight_cols, colors_w, labels_w):
        ax.plot(combined.index, combined[col],
                color=clr, linewidth=1.5, alpha=0.8, label=lbl)
    ax.set_xlabel("Cumulative Frame Index"); ax.set_ylabel("Weight")
    ax.set_title("ACSE — Adaptive Weight Evolution on VIRAT-Aerial")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 0.8)
    _save(fig, "virat_08_weight_evolution.png")


def plot_factor_correlation_heatmap(all_dfs: list[pd.DataFrame]) -> None:
    """Correlation matrix of all 5 factors vs label."""
    if not all_dfs:
        return
    combined = pd.concat(all_dfs, ignore_index=True)
    cols = ["cni","epd","sss","pig","raw","importance_score","label"]
    corr = combined[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, linewidths=0.5, linecolor=BG,
                annot_kws={"size":10})
    ax.set_title("ACSE on VIRAT-Aerial — Factor Correlation Matrix",
                 color=TXT)
    _save(fig, "virat_09_correlation_heatmap.png")


def plot_scene_reduction_summary(all_dfs: list[pd.DataFrame],
                                 video_names: list[str]) -> None:
    """Redundancy reduction percentage per video."""
    if not all_dfs:
        return
    reductions = []
    for df, name in zip(all_dfs, video_names):
        n_total   = len(df)
        n_discard = (df["decision"] == "DISCARD").sum()
        reductions.append({
            "video":     name[:28],
            "reduction": n_discard / max(n_total, 1) * 100,
            "total":     n_total,
            "retained":  (df["decision"] == "RETAIN").sum(),
        })

    fig, ax = plt.subplots(figsize=(max(9, len(reductions)*2), 4), facecolor=BG)
    names  = [r["video"] for r in reductions]
    reds   = [r["reduction"] for r in reductions]
    x = np.arange(len(names))
    bars = ax.bar(x, reds, color=TEAL, alpha=0.85, edgecolor=BORDER, width=0.6)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Redundancy Reduction (%)"); ax.set_ylim(0, 110)
    ax.set_title("ACSE on VIRAT-Aerial — Frames Eliminated as Redundant")
    ax.grid(axis="y", alpha=0.4)
    for bar, v in zip(bars, reds):
        ax.text(bar.get_x()+bar.get_width()/2, v+1.2,
                f"{v:.1f}%", ha="center", fontsize=9, color=TXT)
    _save(fig, "virat_10_redundancy_reduction.png")


def plot_overall_summary_card(metrics: dict,
                              n_videos: int,
                              n_frames: int) -> None:
    """One-page summary card with all key numbers."""
    fig = plt.figure(figsize=(14, 6), facecolor=BG)
    fig.suptitle("ACSE — VIRAT-Aerial Evaluation Summary",
                 fontsize=15, color=TXT, fontweight="bold", y=0.97)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4)

    stat_data = [
        ("Accuracy",   f"{metrics.get('accuracy',0)*100:.2f}%",  GREEN),
        ("Precision",  f"{metrics.get('precision',0)*100:.2f}%", BLUE),
        ("Recall",     f"{metrics.get('recall',0)*100:.2f}%",    ORANGE),
        ("F1 Score",   f"{metrics.get('f1',0)*100:.2f}%",        PURPLE),
        ("ROC-AUC",    f"{metrics.get('roc_auc',0):.4f}",        YELLOW),
        ("Reduction",  f"{metrics.get('reduction_pct',0):.1f}%", TEAL),
        ("Videos",     str(n_videos),                            MUTED),
        ("Frames",     f"{n_frames:,}",                          MUTED),
    ]
    positions = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3)]
    for (r,c), (label, val, col) in zip(positions, stat_data):
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor(PANEL)
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
        ax.text(0.5, 0.65, val, ha="center", va="center",
                fontsize=22, fontweight="bold", color=col,
                transform=ax.transAxes)
        ax.text(0.5, 0.25, label, ha="center", va="center",
                fontsize=10, color=MUTED, transform=ax.transAxes)
        for spine in ["top","bottom","left","right"]:
            ax.spines[spine].set_visible(False)
        rect = plt.Rectangle((0,0),1,1, fill=False,
                              edgecolor=BORDER, linewidth=1.5,
                              transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)

    _save(fig, "virat_00_summary_card.png")


# ═════════════════════════════════════════════════════════════════════════════
# PART 4 — Main runner
# ═════════════════════════════════════════════════════════════════════════════

def run_virat_evaluation(
    virat_dir: Path | None = None,
    max_frames_per_video: int = 1500,
    target_retention: float = 0.30,
    sample_every: int = 1,
) -> dict:
    """
    Main entry point.  Scans VIRAT-Aerial/ for videos, runs ACSE on each,
    saves all evaluation plots to screenshots/, returns results dict.

    Parameters
    ----------
    virat_dir           : path to VIRAT-Aerial folder. Default = dataset/VIRAT-Aerial/
    max_frames_per_video: cap per video to avoid multi-hour runs. Default 1500.
    target_retention    : target retention rate for AdaptiveThresholdGate. Default 0.30.
    sample_every        : process every Nth frame (1 = all frames). For large
                          videos use 2 or 3 to speed up.
    """
    if virat_dir is None:
        virat_dir = VIRAT_DIR

    virat_dir = Path(virat_dir)
    if not virat_dir.exists():
        print(f"\n[!] VIRAT directory not found: {virat_dir}")
        print("    Create folder: dataset/VIRAT-Aerial/")
        print("    Place your .mp4 / .avi video files inside it.")
        return {}

    videos = _find_videos(virat_dir)
    if not videos:
        print(f"\n[!] No video files found in {virat_dir}")
        print("    Supported formats: .mp4 .avi .mov .mkv .webm")
        return {}

    print(f"\n{'='*60}")
    print(f"  ACSE — VIRAT-Aerial Evaluation")
    print(f"  Found {len(videos)} video(s) in {virat_dir}")
    print(f"  Max frames per video : {max_frames_per_video}")
    print(f"  Target retention     : {target_retention*100:.0f}%")
    print(f"  Sample every         : {sample_every} frame(s)")
    print(f"  Output → {SCREENS_DIR}")
    print(f"{'='*60}\n")

    all_dfs        = []
    all_y_true     = []
    all_y_pred     = []
    all_y_scores   = []
    all_latencies  = []
    video_names    = []
    per_video_summary = []

    for vi, vpath in enumerate(videos):
        vname = vpath.stem
        print(f"[{vi+1}/{len(videos)}] Processing: {vpath.name}")

        try:
            df = run_acse_on_video(
                vpath,
                max_frames=max_frames_per_video,
                target_retention=target_retention,
                sample_every=sample_every,
            )
        except Exception as exc:
            print(f"  [!] Error on {vpath.name}: {exc}")
            continue

        if df.empty:
            print(f"  [!] No frames extracted from {vpath.name}")
            continue

        y_true  = df["label"].values
        y_pred  = (df["decision"] == "RETAIN").astype(int).values
        y_scores= df["importance_score"].values

        # Metrics
        acc  = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec  = float(recall_score(y_true, y_pred, zero_division=0))
        f1   = float(f1_score(y_true, y_pred, zero_division=0))
        auc  = float(roc_auc_score(y_true, y_scores)) \
               if len(np.unique(y_true)) > 1 else 0.0
        n_retain   = (df["decision"] == "RETAIN").sum()
        ret_rate   = n_retain / len(df)
        reduction  = 1.0 - ret_rate

        per_video_summary.append({
            "video":          vname,
            "n_frames":       len(df),
            "accuracy":       round(acc,  4),
            "precision":      round(prec, 4),
            "recall":         round(rec,  4),
            "f1":             round(f1,   4),
            "roc_auc":        round(auc,  4),
            "retention_rate": round(ret_rate,  4),
            "reduction_pct":  round(reduction*100, 2),
            "target_rate":    target_retention,
            "mean_latency_ms": round(float(df["latency_ms"].mean()), 3),
            "p95_latency_ms":  round(float(df["latency_ms"].quantile(0.95)), 3),
        })

        print(f"  Frames={len(df):,}  Acc={acc:.4f}  F1={f1:.4f}  "
              f"AUC={auc:.4f}  Reduction={reduction*100:.1f}%  "
              f"Lat={df['latency_ms'].mean():.2f}ms")

        # Per-video plots
        plot_score_timeline(df, vname)
        plot_factor_distributions(df, vname)

        all_dfs.append(df)
        all_y_true.extend(y_true.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_scores.extend(y_scores.tolist())
        all_latencies.extend(df["latency_ms"].tolist())
        video_names.append(vname)

    if not all_dfs:
        print("\n[!] No videos were successfully processed.")
        return {}

    # ── Combined metrics ──────────────────────────────────────────────────────
    y_true_all  = np.array(all_y_true)
    y_pred_all  = np.array(all_y_pred)
    y_score_all = np.array(all_y_scores)

    overall_acc  = float(accuracy_score(y_true_all, y_pred_all))
    overall_prec = float(precision_score(y_true_all, y_pred_all, zero_division=0))
    overall_rec  = float(recall_score(y_true_all, y_pred_all, zero_division=0))
    overall_f1   = float(f1_score(y_true_all, y_pred_all, zero_division=0))
    overall_auc  = float(roc_auc_score(y_true_all, y_score_all)) \
                   if len(np.unique(y_true_all)) > 1 else 0.0
    overall_reduction = 1.0 - float(np.mean(y_pred_all))

    combined_metrics = {
        "accuracy":      round(overall_acc,  4),
        "precision":     round(overall_prec, 4),
        "recall":        round(overall_rec,  4),
        "f1":            round(overall_f1,   4),
        "roc_auc":       round(overall_auc,  4),
        "reduction_pct": round(overall_reduction*100, 2),
        "mean_latency_ms": round(float(np.mean(all_latencies)), 3),
        "p95_latency_ms":  round(float(np.percentile(all_latencies, 95)), 3),
        "p99_latency_ms":  round(float(np.percentile(all_latencies, 99)), 3),
        "n_videos":      len(all_dfs),
        "n_frames_total": len(y_true_all),
    }

    # ── Combined plots ────────────────────────────────────────────────────────
    print("\n[*] Generating combined evaluation plots …")
    plot_overall_summary_card(combined_metrics, len(all_dfs), len(y_true_all))
    plot_per_video_summary(per_video_summary)
    plot_confusion_matrix_combined(y_true_all, y_pred_all,
                                   f"{len(all_dfs)} video(s), "
                                   f"{len(y_true_all):,} frames")
    plot_roc_curve(y_true_all, y_score_all)
    plot_retention_by_video(per_video_summary)
    plot_latency_profile(all_latencies)
    plot_weight_evolution(all_dfs, video_names)
    plot_factor_correlation_heatmap(all_dfs)
    plot_scene_reduction_summary(all_dfs, video_names)

    # ── Save results JSON ──────────────────────────────────────────────────────
    results = {
        "overall":   combined_metrics,
        "per_video": per_video_summary,
    }
    json_path = SCREENS_DIR / "virat_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {json_path.name}")

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  VIRAT-Aerial Evaluation Complete")
    print(f"  Videos processed : {len(all_dfs)}")
    print(f"  Total frames     : {len(y_true_all):,}")
    print(f"  Accuracy         : {overall_acc*100:.2f}%")
    print(f"  Precision        : {overall_prec*100:.2f}%")
    print(f"  Recall           : {overall_rec*100:.2f}%")
    print(f"  F1 Score         : {overall_f1*100:.2f}%")
    print(f"  ROC-AUC          : {overall_auc:.4f}")
    print(f"  Frame Reduction  : {overall_reduction*100:.1f}%")
    print(f"  Mean Latency     : {np.mean(all_latencies):.2f} ms")
    print(f"  p95 Latency      : {np.percentile(all_latencies,95):.2f} ms")
    print(f"  Plots saved to   : {SCREENS_DIR}")
    print(f"{'='*60}\n")

    return results


# ─── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    run_virat_evaluation(
        max_frames_per_video=1500,
        target_retention=0.30,
        sample_every=1,
    )
