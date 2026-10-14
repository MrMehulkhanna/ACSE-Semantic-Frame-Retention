# 🎥 ACSE-Semantic-Frame-Retention

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-green?style=for-the-badge&logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-Vision-red?style=for-the-badge&logo=opencv)
![Machine Learning](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn)

**Semantic-Aware Redundancy Reduction System for Video Streams**

A system that processes video streams, identifies semantically important events and discards redundant frames. It is aimed at resource-constrained settings such as edge devices or bandwidth-limited streaming, where storing every frame is wasteful.

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

These metrics provide quantitative validation of the ACSE model's decision-making performance during runtime.

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License
This project is licensed under the MIT License.
