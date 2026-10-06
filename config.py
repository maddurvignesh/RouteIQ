"""Central configuration for the *Adaptive Routing* project.

Every tunable constant in the project lives in this module so that no other
module contains "magic numbers".  If you want to change the behaviour of the
cost function, the packet simulator or the experiment suite, you change it
here.

Layout of this file
-------------------
1. Filesystem paths
2. Cost-function weights and normalisation bounds
3. Topology defaults
4. Packet-simulation settings
5. Experiment settings
6. Machine-learning settings
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Filesystem paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
DOCS_DIR = BASE_DIR / "docs"

for _dir in (DATA_DIR, MODELS_DIR, RESULTS_DIR, DOCS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

DATASET_CSV = DATA_DIR / "congestion_dataset.csv"
ML_MODEL_PATH = MODELS_DIR / "congestion_model.joblib"
ML_MODEL_REPORT_PATH = MODELS_DIR / "training_report.json"
EXPERIMENT_METRICS_CSV = RESULTS_DIR / "experiment_metrics.csv"
COMPARISON_SUMMARY_CSV = RESULTS_DIR / "comparison_summary.csv"

# ---------------------------------------------------------------------------
# 2. Adaptive cost function
# ---------------------------------------------------------------------------
#
#   cost(link) = w1 * norm_latency
#              + w2 * congestion
#              + w3 * norm_packet_loss
#              + w4 * bandwidth_penalty
#              + ml_penalty(predicted_state)
#
# ``norm_*`` scale raw link values into the interval [0, 1] using the
# normalisation bounds below, so that all four terms are comparable.


@dataclass(frozen=True)
class CostWeights:
    """Weights of the adaptive link-cost formula (must sum to ~1.0)."""

    latency: float = 0.35
    congestion: float = 0.30
    packet_loss: float = 0.25
    bandwidth: float = 0.10

    def __post_init__(self) -> None:
        total = self.latency + self.congestion + self.packet_loss + self.bandwidth
        if not 0.98 <= total <= 1.02:
            raise ValueError(f"Adaptive cost weights must sum to 1.0, got {total:.3f}")

    def as_dict(self) -> dict[str, float]:
        return {
            "Latency (w1)": self.latency,
            "Congestion (w2)": self.congestion,
            "Packet loss (w3)": self.packet_loss,
            "Bandwidth (w4)": self.bandwidth,
        }


DEFAULT_WEIGHTS = CostWeights()

# Raw values that are treated as "worst case" (=> normalised value 1.0).
LATENCY_NORMALISATION_MS = 100.0   # a 100 ms link scores 1.0 on latency
BANDWIDTH_NORMALISATION_MBPS = 1000.0  # a 1 Gbps link scores 1.0 on bandwidth
PACKET_LOSS_NORMALISATION = 0.20   # a 20 % loss link scores 1.0 on loss

# Extra cost added when the ML module predicts a link state.
# Normal -> no penalty, Warning -> moderate penalty, Congested -> high penalty.
ML_COST_PENALTY: dict[int, float] = {0: 0.00, 1: 0.35, 2: 0.90}
ML_STATE_LABELS: dict[int, str] = {0: "Normal", 1: "Warning", 2: "Congested"}

# Congestion levels used when colouring links / reporting status.
CONGESTION_WARNING_THRESHOLD = 0.50
CONGESTION_HIGH_THRESHOLD = 0.70

# Routing strategies understood by the system.
STRATEGY_SHORTEST = "Shortest Path"
STRATEGY_ADAPTIVE = "Adaptive Routing"
STRATEGY_ADAPTIVE_ML = "Adaptive + ML"
ALL_STRATEGIES = (STRATEGY_SHORTEST, STRATEGY_ADAPTIVE, STRATEGY_ADAPTIVE_ML)

# How many alternative routes are shown to the user on the dashboard.
MAX_ALTERNATIVE_ROUTES = 4

# ---------------------------------------------------------------------------
# 3. Topology defaults
# ---------------------------------------------------------------------------

DEFAULT_TOPOLOGY_NAME = "predefined"
DEFAULT_SRC = "S"
DEFAULT_DST = "T"

# Reasonable ranges used by the random topology generator.
RANDOM_TOPOLOGY_SEED = 7


# ---------------------------------------------------------------------------
# 4. Packet simulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PacketConfig:
    """Knobs of the simulated packet transmitter."""

    num_packets: int = 80            # packets sent per simulation step
    packet_size_kb: int = 8          # application payload size
    inter_arrival_ms: float = 5.0    # time between two consecutive packets
    max_retries: int = 2             # retransmissions after a drop
    jitter_fraction: float = 0.10    # +/- 10 % delay variation
    queue_scale_ms: float = 12.0     # how strongly congestion adds delay
    max_congestion_for_delay: float = 0.97  # avoids infinite queue delay
    retransmission_timeout_ms: float = 40.0  # wait before re-sending a lost packet


DEFAULT_PACKET_CONFIG = PacketConfig()

# ---------------------------------------------------------------------------
# 5. Experiment / simulation engine
# ---------------------------------------------------------------------------

EXPERIMENT_SEED = 42
EXPERIMENT_STEPS = 5          # time steps per experiment run
DEFAULT_SIM_STEPS = 5         # time steps used by the dashboard simulator

# ---------------------------------------------------------------------------
# 6. Machine learning
# ---------------------------------------------------------------------------

ML_FEATURES = [
    "latency_ms",
    "packet_loss",
    "bandwidth_utilization",
    "traffic_mbps",
    "queue_length",
    "previous_level",
]

ML_TARGET = "congestion_level"

ML_RANDOM_STATE = 42
ML_TEST_SIZE = 0.25
ML_DATASET_SIZE = 5000
ML_MIN_ACCEPTABLE_ACCURACY = 0.80  # training aborts with an error below this

# Label rule (documented ground truth) -------------------------------
# severity = 0.40 * utilization + 0.30 * min(latency/100, 1)
#                     + 0.30 * min(packet_loss / 0.15, 1)
# level 2 (Congested)  if severity >= 0.65
# level 1 (Warning)    if severity >= 0.40
# level 0 (Normal)     otherwise
ML_SEVERITY_WEIGHTS = {"utilization": 0.40, "latency": 0.30, "loss": 0.30}
ML_SEVERITY_THRESHOLDS = {"warning": 0.40, "congested": 0.65}
