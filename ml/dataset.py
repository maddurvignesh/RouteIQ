"""Synthetic congestion dataset generation.

Real router telemetry is not publicly available, so the project creates its
own labelled dataset.  This is stated openly in the report – nothing is
downloaded or copied.

How the data is produced
------------------------
1. ``num_streams`` simulated links are created.  Each link gets a random
   capacity (100 – 1000 Mbps), a fixed baseline delay and a random target
   utilisation.
2. Over ``steps`` consecutive observations the utilisation performs a small
   random walk around its target (so congestion is **temporally correlated**,
   just like on a real link).
3. Latency, packet loss and queue length are derived from utilisation through
   simple queueing relationships, plus measurement noise.
4. The class label is assigned with the documented severity rule in
   :mod:`ml.model`.
5. ``previous_level`` is the label of the previous observation of the same
   link – a router always knows the previous state of a link, so this feature
   is available at inference time and does not leak future information.

Run ``python -m ml.dataset`` to regenerate ``data/congestion_dataset.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import DATASET_CSV, ML_DATASET_SIZE, ML_RANDOM_STATE, ML_TARGET
from ml.model import congestion_label

BANDWIDTH_CHOICES = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0]
STEPS_PER_STREAM = 50
MIN_UTILIZATION = 0.02
MAX_UTILIZATION = 0.995


def generate_dataset(
    n_samples: int = ML_DATASET_SIZE,
    seed: int = ML_RANDOM_STATE,
    steps_per_stream: int = STEPS_PER_STREAM,
) -> pd.DataFrame:
    """Build a labelled congestion dataset of roughly ``n_samples`` rows."""
    if n_samples < steps_per_stream:
        raise ValueError(f"n_samples must be >= {steps_per_stream}")
    if steps_per_stream < 2:
        raise ValueError("steps_per_stream must be at least 2 (needed for previous_level)")

    rng = np.random.default_rng(seed)
    num_streams = int(np.ceil(n_samples / steps_per_stream))

    rows: list[dict] = []
    for stream in range(num_streams):
        capacity = float(rng.choice(BANDWIDTH_CHOICES))
        target_util = float(rng.uniform(0.10, 0.99))
        base_latency = float(rng.uniform(4.0, 45.0))
        base_loss = float(rng.uniform(0.001, 0.020))

        util = target_util
        previous_level = 0
        for step in range(steps_per_stream):
            util += (target_util - util) * 0.12 + float(rng.normal(0.0, 0.035))
            util = float(np.clip(util, MIN_UTILIZATION, MAX_UTILIZATION))

            latency = base_latency * (1.0 + 4.0 * util ** 3) + float(rng.normal(0.0, 1.5))
            latency = float(np.clip(latency, 1.0, 250.0))

            loss = base_loss + 0.15 * util ** 3 + float(rng.normal(0.0, 0.004))
            loss = float(np.clip(loss, 0.0005, 0.50))

            queue = util ** 2 * 60.0 * max(0.0, 1.0 + float(rng.normal(0.0, 0.15)))
            queue_length = int(round(queue))

            level = congestion_label(util, latency, loss)
            rows.append({
                "link_id": f"L{stream:03d}",
                "timestep": step,
                "bandwidth_mbps": capacity,
                "traffic_mbps": round(util * capacity, 3),
                "latency_ms": round(latency, 3),
                "packet_loss": round(loss, 6),
                "bandwidth_utilization": round(util, 6),
                "queue_length": float(queue_length),
                "previous_level": float(previous_level),
                ML_TARGET: level,
            })
            previous_level = level

    frame = pd.DataFrame(rows)
    return frame.head(n_samples).reset_index(drop=True)


def save_dataset(frame: pd.DataFrame, path: Path = DATASET_CSV) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load_dataset(path: Path = DATASET_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python -m ml.dataset` first."
        )
    return pd.read_csv(path)


def ensure_dataset(path: Path = DATASET_CSV, **kwargs) -> pd.DataFrame:
    """Load the dataset, generating it first when it does not exist yet."""
    if not path.exists():
        save_dataset(generate_dataset(**kwargs), path)
    return load_dataset(path)


def class_distribution(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame[ML_TARGET].value_counts().sort_index()
    return {int(k): int(v) for k, v in counts.items()}


if __name__ == "__main__":  # pragma: no cover - manual utility
    data = generate_dataset()
    target = save_dataset(data)
    print(f"Wrote {len(data)} samples to {target}")
    print("Class distribution:", class_distribution(data))
