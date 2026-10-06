"""Live congestion prediction used by the *Adaptive + ML* routing strategy.

The predictor wraps the trained Random Forest and exposes exactly one method
that the routing engine cares about:

``link_penalty(u, v, attributes) -> float``

which has the same signature as ``core.adaptive_routing.PenaltyFunction``.
The routing engine therefore never imports this module – it only receives the
callback.  That is what keeps *routing logic* and *ML logic* separate, and it
is why the project still runs when the model file is missing.

Penalty mapping (see ``config.ML_COST_PENALTY``):

===========  ==============  =======================================
prediction   penalty         effect on routing
===========  ==============  =======================================
Normal (0)   +0.00           link keeps its ordinary dynamic cost
Warning (1)  +0.35           link becomes less attractive
Congested(2) +0.90           link is normally avoided altogether
===========  ==============  =======================================

Run ``python -m ml.predict`` to see a few sample predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from config import ML_COST_PENALTY, ML_FEATURES, ML_MODEL_PATH, ML_STATE_LABELS
from ml.model import extract_features, features_to_row


def _canonical(u: str, v: str) -> tuple[str, str]:
    return (u, v) if u <= v else (v, u)


class CongestionPredictor:
    """Loads the trained model and predicts the congestion state of a link."""

    def __init__(self, model_path: Path = ML_MODEL_PATH, enabled: bool = True) -> None:
        self.model_path = Path(model_path)
        self.enabled = enabled
        self._model = None
        self._load_error: str | None = None
        self._previous_levels: dict[tuple[str, str], int] = {}
        self.prediction_count = 0

        if enabled:
            self._load()

    # -- model management ---------------------------------------------------
    def _load(self) -> None:
        if not self.model_path.exists():
            self._load_error = (
                f"Model file not found at {self.model_path}. "
                "Run `python -m ml.train` to create it."
            )
            return
        try:
            self._model = joblib.load(self.model_path)
        except Exception as exc:  # pragma: no cover - defensive
            self._load_error = f"Could not load model: {exc}"

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def error(self) -> str | None:
        return self._load_error

    @property
    def active(self) -> bool:
        """True when the ML module will actually influence routing."""
        return self.enabled and self.available

    def reset(self) -> None:
        """Forget the history of previous predictions (new simulation run)."""
        self._previous_levels.clear()
        self.prediction_count = 0

    # -- prediction ---------------------------------------------------------
    def predict(self, features: Mapping[str, float]) -> tuple[int, float]:
        """Return ``(congestion level, confidence)`` for one feature vector."""
        if not self.available:
            return 0, 0.0
        # A named DataFrame avoids sklearn's "feature names" warning.
        row = pd.DataFrame([features_to_row(features)], columns=ML_FEATURES)
        probabilities = self._model.predict_proba(row)[0]
        classes = list(self._model.classes_)
        best = int(np.argmax(probabilities))
        return int(classes[best]), float(probabilities[best])

    def predict_link(self, u: str, v: str, attributes: Mapping[str, Any]) -> int:
        """Predict the congestion class of link ``u -- v`` and remember it."""
        key = _canonical(u, v)
        features = extract_features(attributes, previous_level=self._previous_levels.get(key, 0))
        level, _ = self.predict(features)
        self._previous_levels[key] = level
        self.prediction_count += 1
        return level

    def link_penalty(self, u: str, v: str, attributes: Mapping[str, Any]) -> float:
        """``PenaltyFunction`` implementation consumed by the routing engine."""
        if not self.active:
            return 0.0
        return float(ML_COST_PENALTY[self.predict_link(u, v, attributes)])

    # -- reporting ----------------------------------------------------------
    def inspect_link(self, u: str, v: str, attributes: Mapping[str, Any]) -> dict[str, Any]:
        """Non-mutating inspection used by the dashboard (no state update)."""
        key = _canonical(u, v)
        features = extract_features(attributes, previous_level=self._previous_levels.get(key, 0))
        level, confidence = self.predict(features)
        return {
            "link": f"{u}-{v}",
            "level": level,
            "state": ML_STATE_LABELS[level],
            "confidence": confidence,
            "penalty": float(ML_COST_PENALTY[level]) if self.active else 0.0,
            "features": features,
        }

    def predict_path(self, path: list[str] | tuple[str, ...],
                     link_attributes) -> list[dict[str, Any]]:
        """Inspect every link of a path using ``link_attributes(u, v)``."""
        return [
            self.inspect_link(u, v, link_attributes(u, v))
            for u, v in zip(path, path[1:])
        ]

    def summary(self) -> dict[str, Any]:
        """Status block shown on the dashboard's ML page."""
        counts = {label: 0 for label in ML_STATE_LABELS.values()}
        for level in self._previous_levels.values():
            counts[ML_STATE_LABELS[level]] += 1
        return {
            "enabled": self.enabled,
            "available": self.available,
            "active": self.active,
            "model_path": str(self.model_path),
            "error": self._load_error,
            "features": list(ML_FEATURES),
            "penalties": dict(ML_COST_PENALTY),
            "prediction_count": self.prediction_count,
            "tracked_links": len(self._previous_levels),
            "state_counts": counts,
        }


def load_predictor(enabled: bool = True, model_path: Path = ML_MODEL_PATH) -> CongestionPredictor:
    """Factory used by the dashboard and the experiment runner."""
    return CongestionPredictor(model_path=model_path, enabled=enabled)


if __name__ == "__main__":  # pragma: no cover - manual utility
    from core.topology import build_predefined_topology

    predictor = load_predictor()
    print("Model available:", predictor.available, predictor.error or "")
    topo = build_predefined_topology()
    for u, v in topo.links:
        info = predictor.inspect_link(u, v, topo.link_attributes(u, v))
        print(f"  {info['link']:<6} -> {info['state']:<9} "
              f"p={info['confidence']:.2f} penalty={info['penalty']:.2f}")
