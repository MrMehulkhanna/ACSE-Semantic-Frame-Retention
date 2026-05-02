"""
ACSE — Flask Backend
======================

Full working Flask application that serves the UI and provides REST API
endpoints for video upload, processing, frame explanations, weight history,
and memory state inspection.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.temporal_memory import TemporalGraphMemory
from core.novelty_engine import CognitiveNoveltyIndex
from core.event_decay import EventPersistenceDecay
from core.surprise_engine import SemanticSurpriseScore
from core.semantic_router import MultiFactorFusion
from core.explanation_engine import SHAPTraceExplainer
from core.adaptive_learning import OnlineBayesianUpdater
from core.decision_gate import AdaptiveThresholdGate

app = Flask(__name__, static_folder=None)
CORS(app)

UPLOAD_DIR = PROJECT_ROOT / "uploads"
RESULTS_DIR = PROJECT_ROOT / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
UI_DIR = PROJECT_ROOT / "ui"

# Global pipeline state (shared across requests for status)
_global_memory = TemporalGraphMemory()
_global_updater = OnlineBayesianUpdater()
_global_gate = AdaptiveThresholdGate()

# Job storage: job_id -> {video_path, results, explanations, frames}
_jobs: dict[str, dict] = {}


# ---------------------------------------------------------------
# UI routes
# ---------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(str(UI_DIR), "index.html")


@app.route("/ui/<path:filename>")
def serve_ui(filename):
    return send_from_directory(str(UI_DIR), filename)


# ---------------------------------------------------------------
# API routes
# ---------------------------------------------------------------

@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({
        "status": "ready",
        "weights": _global_updater.get_weights(),
        "memory": _global_memory.get_state_snapshot(),
        "gate_stats": _global_gate.get_stats(),
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file in request"}), 400

    file = request.files["video"]
    job_id = str(uuid.uuid4())[:12]

    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / file.filename
    file.save(str(video_path))

    _jobs[job_id] = {
        "video_path": str(video_path),
        "results": None,
        "explanations": {},
        "retained_frames": [],
        "status": "uploaded",
    }

    return jsonify({"job_id": job_id, "filename": file.filename, "status": "uploaded"})


@app.route("/api/process/<job_id>", methods=["POST"])
def api_process(job_id):
    if job_id not in _jobs:
        return jsonify({"error": "Job not found"}), 404

    job = _jobs[job_id]
    if job["status"] == "processing":
        return jsonify({"error": "Already processing"}), 409

    job["status"] = "processing"
    video_path = job["video_path"]

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        job["status"] = "error"
        return jsonify({"error": f"Cannot open video: {video_path}"}), 400

    # Pipeline components
    memory = TemporalGraphMemory()
    cni_engine = CognitiveNoveltyIndex()
    epd_engine = EventPersistenceDecay()
    sss_engine = SemanticSurpriseScore()
    fusion = MultiFactorFusion()
    updater = OnlineBayesianUpdater()
    explainer = SHAPTraceExplainer(fusion)
    gate = AdaptiveThresholdGate(target_retention_rate=0.30)

    frames_dir = RESULTS_DIR / job_id / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_results = []
    explanations = {}
    retained_frames = []
    prev_frame = None
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        desc = memory.update(frame, idx, salience=0.5)
        cni = cni_engine.compute(desc, memory)
        epd_engine.update(frame, prev_frame, idx)
        epd = epd_engine.compute_epd_score(idx)
        sss_engine.update_predictor(desc)
        sss = sss_engine.compute(desc)
        fused = fusion.fuse(cni, epd, sss, memory, idx)
        fusion.set_weights(updater.get_weights())
        gate_result = gate.decide(fused["importance_score"], idx)
        fused["decision"] = gate_result["decision"]

        explanation = explainer.explain_decision(fused, idx)
        frame_results.append({
            "frame_idx": idx,
            "importance_score": fused["importance_score"],
            "cni": cni,
            "epd": epd,
            "sss": sss,
            "pig": fused["pig"],
            "raw": fused["raw"],
            "decision": gate_result["decision"],
            "threshold": gate_result["threshold"],
        })

        if gate_result["decision"] == "RETAIN":
            frame_path = frames_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            retained_frames.append(f"frame_{idx:06d}.jpg")
            # Positive reward for retained frames (self-supervised proxy)
            updater.update(fused, +1.0)
        else:
            updater.update(fused, -0.5)

        explanations[str(idx)] = explanation

        prev_frame = frame
        idx += 1

    cap.release()

    # Store results
    job["results"] = frame_results
    job["explanations"] = explanations
    job["retained_frames"] = retained_frames
    job["status"] = "done"
    job["total_frames"] = idx
    job["retained_count"] = len(retained_frames)

    # Update global state for status endpoint
    global _global_memory, _global_updater, _global_gate
    _global_memory = memory
    _global_updater = updater
    _global_gate = gate

    # Save full results JSON
    result_json = RESULTS_DIR / job_id / "results.json"
    with open(result_json, "w") as f:
        json.dump(frame_results, f, indent=2)

    return jsonify({
        "job_id": job_id,
        "status": "done",
        "total_frames": idx,
        "retained_frames": len(retained_frames),
        "retention_rate": round(len(retained_frames) / max(idx, 1), 4),
    })


@app.route("/api/explanation/<job_id>/<int:frame_id>", methods=["GET"])
def api_explanation(job_id, frame_id):
    if job_id not in _jobs:
        return jsonify({"error": "Job not found"}), 404
    explanations = _jobs[job_id].get("explanations", {})
    key = str(frame_id)
    if key not in explanations:
        return jsonify({"error": f"No explanation for frame {frame_id}"}), 404
    return jsonify(explanations[key])


@app.route("/api/weights/history", methods=["GET"])
def api_weight_history():
    df = _global_updater.get_weight_history()
    return jsonify(df.to_dict(orient="list"))


@app.route("/api/memory/state", methods=["GET"])
def api_memory_state():
    return jsonify(_global_memory.get_state_snapshot())


@app.route("/api/frames/<job_id>", methods=["GET"])
def api_frames(job_id):
    if job_id not in _jobs:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "job_id": job_id,
        "frames": _jobs[job_id].get("retained_frames", []),
    })


@app.route("/api/frames/<job_id>/<filename>", methods=["GET"])
def api_frame_image(job_id, filename):
    frames_dir = RESULTS_DIR / job_id / "frames"
    return send_from_directory(str(frames_dir), filename)


@app.route("/api/results/<job_id>", methods=["GET"])
def api_results(job_id):
    if job_id not in _jobs:
        return jsonify({"error": "Job not found"}), 404
    job = _jobs[job_id]
    return jsonify({
        "job_id": job_id,
        "status": job.get("status"),
        "total_frames": job.get("total_frames", 0),
        "retained_count": job.get("retained_count", 0),
        "results": job.get("results", []),
    })


@app.route("/api/cleanup/<job_id>", methods=["DELETE"])
def api_cleanup(job_id):
    if job_id not in _jobs:
        return jsonify({"error": "Job not found"}), 404

    # Remove files
    job_upload_dir = UPLOAD_DIR / job_id
    job_results_dir = RESULTS_DIR / job_id
    if job_upload_dir.exists():
        shutil.rmtree(job_upload_dir)
    if job_results_dir.exists():
        shutil.rmtree(job_results_dir)
    del _jobs[job_id]

    return jsonify({"status": "cleaned", "job_id": job_id})


if __name__ == "__main__":
    print("=" * 60)
    print("  ACSE — Adaptive Contextual Semantic Retention Engine")
    print("  Backend running at http://127.0.0.1:5050")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5050, debug=True)
