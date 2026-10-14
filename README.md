# 🎥 ACSE-Semantic-Frame-Retention

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-green?style=for-the-badge&logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-Vision-red?style=for-the-badge&logo=opencv)
![Machine Learning](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn)

**Semantic-Aware Redundancy Reduction System for Video Streams**

A system designed to identify semantically important events in video streams and discard redundant frames. It is aimed at resource-constrained settings such as edge devices or bandwidth-limited streaming, where storing every frame is wasteful.

---

## ✨ Key Features

*   🧠 **Cognitive Core Engine:** Utilizes advanced memory and novelty detection to evaluate frame importance.
*   ⏳ **Temporal Memory Management:** Tracks objects and activities over time, prioritizing sustained actions and intelligently decaying obsolete events.
*   🚦 **Decision Gate:** Employs machine learning classifiers to make high-confidence decisions on whether to retain or discard frames based on computed ACSE (Action, Context, Semantic, Event) features.
*   💡 **Explanation Engine:** Provides human-readable reasoning for every retention or rejection decision.
*   🌐 **Interactive UI & Backend:** A built-in Flask application to easily test the pipeline with video files and visualize retained frames and metrics.
*   ⚡ **Self-Pruning:** Adapts dynamically to ongoing activities, reducing computational overhead when scenes become static.

---

## 🏗️ Project Architecture

The core of the system resides in the `core/` directory, orchestrating multiple engines to process video intelligently:

*   **`temporal_memory.py`**: Manages short-term and long-term memory for tracking objects and states.
*   **`novelty_engine.py`**: Calculates the mathematical surprise or novelty of a new frame compared to recent history.
*   **`event_decay.py`**: Handles the gradual decay of event importance if no new significant action occurs.
*   **`semantic_router.py`**: Routes features to appropriate processing nodes.
*   **`explanation_engine.py`**: Generates text-based justifications for frame retention.
*   **`decision_gate.py`**: The final classifier that green-lights or drops a frame based on aggregated features.

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed. 

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MrMehulkhanna/ACSE-Semantic-Frame-Retention.git
   cd ACSE-Semantic-Frame-Retention
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Start the Flask backend server to launch the video processing pipeline.
```bash
cd backend
python app.py
```
*(The server will typically run on `http://127.0.0.1:5000`)*

---

## 📊 Evaluation & Metrics

The system continuously computes various classification metrics:
*   **Accuracy:** Overall correctness of frame retention decisions.
*   **F1-Score:** The harmonic mean of precision and recall.
*   **ROC/AUC:** Used to evaluate the decision gate's confidence threshold.

### Current results (synthetic benchmark, October 2026)

Reproduce with `python experiments/dataset_builder.py` (about 25 minutes on a laptop CPU), then `python experiments/ablation.py` and `python experiments/baseline_check.py`.

The dataset has 1,500 synthetic frames (3 scene types × 500), of which 8.4% are labelled important.

| Variant | Accuracy | F1 | Frames kept |
|---|---|---|---|
| Full pipeline | 0.668 | 0.117 | 29.2% |
| Without event persistence (EPD) | 0.678 | 0.130 | 28.6% |
| Fixed equal weights | 0.704 | 0.108 | 24.8% |
| Without temporal memory | 0.678 | 0.104 | 27.5% |
| Random gate keeping 29.2% (expected) | 0.673 | 0.130 | 29.2% |

**What this shows:** the fused importance score is currently at chance (AUC 0.495), while the event-persistence factor on its own reaches AUC 0.694. The fusion step is losing the useful signal, so the next step is to learn the fusion weights from labelled frames and to compare against single-factor baselines. `experiments/benchmark.py` measures about 0.73 s per frame on average (95th percentile 1.66 s), mostly in the temporal-memory update (0.45 s) and the surprise engine's online regression (0.19 s), so the pipeline is not yet fast enough for edge devices.

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License
This project is licensed under the MIT License.
