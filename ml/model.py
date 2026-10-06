"""Shared definitions of the ML congestion-prediction module.

This file deliberately contains **no model code** – it only defines

* the feature vector used by both training and inference,
* the documented ground-truth labelling rule,
* helpers that convert raw link attributes into model features.

Keeping these together guarantees that the training pipeline and the live
router always agree on the input format (a classic source of ML bugs).
"""

from __future__ import annotations

from typing import Any, Mapping

from config import (
    ML_FEATURES,
    ML_SEVERITY_THRESHOLDS,
    ML_SEVERITY_WEIGHTS,
    ML_TARGET,
)

CLASS_LABELS = {0: "Normal", 1: "Warning", 2: "Congested"}


def severity_score(
    utilization: float,
    latency_ms: float,
    packet_loss: float,
    weights: Mapping[str, float] = ML_SEVERITY_WEIGHTS,
) -> float:
    """Weighted congestion severity of one link, in the range [0, ~1].

    .. math::

        severity = 0.40 \\cdot utilization
                 + 0.30 \\cdot \\min(latency/100,\\,1)
                 + 0.30 \\cdot \\min(loss/0.15,\\,1)

    Utilisation carries the most weight because a link that is nearly full is
    congested even if its fixed delay is still small.
    """
    return (
        weights["utilization"] * max(0.0, min(utilization, 1.0))
        + weights["latency"] * min(latency_ms / 100.0, 1.0)
        + weights["loss"] * min(packet_loss / 0.15, 1.0)
    )


def label_from_severity(
    severity: float,
    thresholds: Mapping[str, float] = ML_SEVERITY_THRESHOLDS,
) -> int:
    """Map a severity score onto the three congestion classes 0 / 1 / 2."""
    if severity >= thresholds["congested"]:
        return 2
    if severity >= thresholds["warning"]:
        return 1
    return 0


def congestion_label(utilization: float, latency_ms: float, packet_loss: float) -> int:
    """Documented ground truth used to label the synthetic dataset."""
    return label_from_severity(severity_score(utilization, latency_ms, packet_loss))


def extract_features(
    attributes: Mapping[str, Any],
    previous_level: int = 0,
    utilization: float | None = None,
) -> dict[str, float]:
    """Convert raw link attributes into the model's feature vector.

    Parameters
    ----------
    attributes:
        Live link attributes (``latency``, ``bandwidth``, ``traffic``...).
    previous_level:
        Congestion class predicted for this link during the previous routing
        interval.  A live router always knows it, so it is a legitimate feature.
    utilization:
        Pre-computed ``traffic / bandwidth``; calculated when not supplied.
    """
    bandwidth = max(float(attributes["bandwidth"]), 1e-9)
    if utilization is None:
        utilization = min(float(attributes["traffic"]) / bandwidth, 1.0)

    return {
        "latency_ms": float(attributes["latency"]),
        "packet_loss": float(attributes["packet_loss"]),
        "bandwidth_utilization": float(utilization),
        "traffic_mbps": float(attributes["traffic"]),
        # Buffer depth grows non-linearly with utilisation (see ml.dataset).
        "queue_length": float(attributes.get("congestion", 0.0)) ** 2 * 60.0,
        "previous_level": float(previous_level),
    }


def features_to_row(features: Mapping[str, float]) -> list[float]:
    """Order the feature mapping according to :data:`config.ML_FEATURES`."""
    return [float(features[name]) for name in ML_FEATURES]


def validate_dataset_columns(columns) -> None:
    missing = [c for c in ML_FEATURES + [ML_TARGET] if c not in columns]
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {missing}")
