"""
ACSE — Module 4: Semantic Surprise Score (SSS)
=================================================

Measures how much the current frame *violates* the system's learned prediction
model.  Novelty = different from memory.  Surprise = different from what was
*predicted* to come next, based on an online autoregressive model.

Related approaches:
  Zhuang 1998 and Liu & Cheng 2008 have no temporal prediction — they only
  compare adjacent frames.  Zhang et al. 2016 uses LSTM for summarisation but
  the hidden state is opaque and cannot report a calibrated "surprise" score.
  VA-RED2 predicts computation reduction but not scene content.  ReTaKe has no
  autoregressive component.
  Our SSS trains a lightweight SGD regressor *online* on the stream of
  descriptors, producing a prediction-error surprise signal that is
  z-normalised against a rolling error history and sigmoidally squashed.
"""

from collections import deque

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor


class SemanticSurpriseScore:
    """Online autoregressive surprise against predicted descriptors."""

    def __init__(self, descriptor_dim: int = 128, window: int = 3) -> None:
        self.descriptor_dim = descriptor_dim
        self.window = window

        # SGD regressor — predicts D_t from concat(D_{t-3}, D_{t-2}, D_{t-1})
        base = SGDRegressor(
            loss="squared_error",
            learning_rate="adaptive",
            eta0=0.001,
            random_state=42,
        )
        self.predictor = MultiOutputRegressor(base)
        self.descriptor_history: deque = deque(maxlen=window + 1)
        self.is_warm: bool = False
        self.prediction_errors: deque = deque(maxlen=50)
        self._frames_seen: int = 0
        self._eps: float = 1e-6

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 1.0
        return float(1.0 - np.dot(a, b) / (na * nb))

    @staticmethod
    def _sigmoid(x: float) -> float:
        x = np.clip(x, -10.0, 10.0)
        return float(1.0 / (1.0 + np.exp(-x)))

    def update_predictor(self, descriptor: np.ndarray) -> None:
        """Partial-fit the SGD predictor on the latest descriptor."""
        self.descriptor_history.append(descriptor.copy())
        self._frames_seen += 1

        if len(self.descriptor_history) >= self.window + 1:
            history = list(self.descriptor_history)
            X = np.concatenate(history[-(self.window + 1) : -1]).reshape(1, -1).astype(np.float64)
            y = history[-1].reshape(1, -1).astype(np.float64)
            self.predictor.partial_fit(X, y)

        self.is_warm = self._frames_seen >= 10

    def compute(self, descriptor: np.ndarray) -> float:
        """SSS(t) = sigmoid((error - mu) / (sigma + eps))."""
        if not self.is_warm or len(self.descriptor_history) < self.window:
            return 0.5  # neutral when cold

        history = list(self.descriptor_history)
        if len(history) < self.window:
            return 0.5

        X = np.concatenate(history[-self.window :]).reshape(1, -1).astype(np.float64)
        D_pred = self.predictor.predict(X)[0]
        prediction_error = self._cosine_distance(descriptor, D_pred)

        self.prediction_errors.append(prediction_error)

        if len(self.prediction_errors) < 3:
            return 0.5

        mu = float(np.mean(self.prediction_errors))
        sigma = float(np.std(self.prediction_errors))
        z = (prediction_error - mu) / (sigma + self._eps)
        return self._sigmoid(z)

    def get_prediction_error_stats(self) -> dict:
        if not self.prediction_errors:
            return {"mean": 0.0, "std": 0.0, "count": 0}
        return {
            "mean": float(np.mean(self.prediction_errors)),
            "std": float(np.std(self.prediction_errors)),
            "count": len(self.prediction_errors),
        }
