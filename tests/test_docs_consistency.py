"""Consistency checks between the documentation and the actual artefacts.

The report, the README and the presentation quote concrete numbers (accuracies,
latencies, percentages, routes).  Those numbers are only meaningful if they
still match what the code produces, so they are asserted here.

This module also guards a subtle regression: ``tests/test_simulation.py`` once
called ``run_all_experiments(save=True)``, which overwrote ``results/*.csv``
with a run **without** the ML penalty and silently made Adaptive + ML identical
to Adaptive Routing.  The dedicated results written by
``SimulationEngine.run_experiments()`` are what the docs quote, so this module
asserts the ML row is genuinely different.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from config import (
    BANDWIDTH_NORMALISATION_MBPS,
    COMPARISON_SUMMARY_CSV,
    DATASET_CSV,
    DEFAULT_WEIGHTS,
    EXPERIMENT_METRICS_CSV,
    LATENCY_NORMALISATION_MS,
    ML_COST_PENALTY,
    ML_DATASET_SIZE,
    PACKET_LOSS_NORMALISATION,
    STRATEGY_ADAPTIVE,
    STRATEGY_ADAPTIVE_ML,
    STRATEGY_SHORTEST,
)
from ml.dataset import class_distribution
from ml.train import load_training_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"

ALL_DOCS = sorted(DOCS_DIR.glob("*.md")) + [PROJECT_ROOT / "README.md"]

REQUIRED_FILES = [
    "README.md",
    "docs/report.md",
    "docs/architecture.md",
    "docs/methodology.md",
    "docs/experiments.md",
    "docs/presentation.md",
    "docs/viva.md",
    "docs/diagrams.py",
    "docs/build_presentation.py",
    "docs/export_slides.ps1",
    "docs/screenshot.py",
    "visualization/theme.py",
    "docs/Adaptive_Routing_Presentation.pptx",
    "docs/architecture.png",
    "docs/routing_decision.png",
    "docs/topology.png",
    "docs/results_summary.png",
    "docs/ml_confusion.png",
    "docs/ml_importance.png",
]


@pytest.fixture(scope="module")
def summary() -> pd.DataFrame:
    assert COMPARISON_SUMMARY_CSV.exists(), "run SimulationEngine().run_experiments() first"
    return pd.read_csv(COMPARISON_SUMMARY_CSV)


@pytest.fixture(scope="module")
def metrics() -> pd.DataFrame:
    assert EXPERIMENT_METRICS_CSV.exists(), "run SimulationEngine().run_experiments() first"
    return pd.read_csv(EXPERIMENT_METRICS_CSV)


# ---------------------------------------------------------------------------
# Constants quoted in the documentation
# ---------------------------------------------------------------------------

def test_cost_weights_match_the_docs() -> None:
    assert (DEFAULT_WEIGHTS.latency, DEFAULT_WEIGHTS.congestion,
            DEFAULT_WEIGHTS.packet_loss, DEFAULT_WEIGHTS.bandwidth) == (0.35, 0.30, 0.25, 0.10)


def test_normalisation_bounds_match_the_docs() -> None:
    assert (LATENCY_NORMALISATION_MS, PACKET_LOSS_NORMALISATION,
            BANDWIDTH_NORMALISATION_MBPS) == (100.0, 0.20, 1000.0)


def test_ml_penalties_match_the_docs() -> None:
    assert ML_COST_PENALTY == {0: 0.0, 1: 0.35, 2: 0.90}


# ---------------------------------------------------------------------------
# Dataset and model
# ---------------------------------------------------------------------------

def test_dataset_shape_and_class_balance() -> None:
    data = pd.read_csv(DATASET_CSV)
    assert len(data) == ML_DATASET_SIZE == 5000
    assert class_distribution(data) == {0: 2280, 1: 1360, 2: 1360}


def test_training_report_matches_the_docs() -> None:
    report = load_training_report()
    assert round(report["accuracy"], 4) == 0.9848
    per_class = report["classification_report"]
    assert round(per_class["macro avg"]["f1-score"], 3) == 0.983
    assert [round(per_class[k]["f1-score"], 3) for k in ("0", "1", "2")] == [0.992, 0.972, 0.985]


def test_feature_importances_match_the_docs() -> None:
    importance = load_training_report()["feature_importances"]
    order = ["previous_level", "packet_loss", "bandwidth_utilization",
             "latency_ms", "queue_length"]
    assert [round(importance[name] * 100) for name in order] == [26, 25, 22, 13, 11]
    assert min(importance, key=importance.get) == "traffic_mbps"


# ---------------------------------------------------------------------------
# Experiment results
# ---------------------------------------------------------------------------

EXPECTED_SUMMARY = {
    # strategy: (latency, loss %, delivery %, congestion freq %, throughput)
    STRATEGY_SHORTEST: (354.74, 5.83, 94.17, 40.00, 25.32),
    STRATEGY_ADAPTIVE: (151.05, 0.29, 99.71, 26.67, 36.23),
    STRATEGY_ADAPTIVE_ML: (125.58, 0.29, 99.71, 13.33, 40.07),
}


def test_sixteen_eight_runs_are_recorded(metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    assert len(metrics) == 18
    assert len(summary) == 3
    assert set(summary["strategy"]) == set(EXPECTED_SUMMARY)


@pytest.mark.parametrize("strategy", list(EXPECTED_SUMMARY))
def test_summary_row_matches_the_docs(summary: pd.DataFrame, strategy: str) -> None:
    row = summary[summary["strategy"] == strategy].iloc[0]
    expected = EXPECTED_SUMMARY[strategy]
    actual = (
        round(row["avg_latency_ms"], 2),
        round(row["packet_loss_pct"], 2),
        round(row["delivery_rate_pct"], 2),
        round(row["congestion_frequency_pct"], 2),
        round(row["throughput_mbps"], 2),
    )
    assert actual == expected


def test_ml_row_is_genuinely_different(summary: pd.DataFrame) -> None:
    """Regression guard: the results must not be a penalty-free run."""
    assert (summary[summary["strategy"] == STRATEGY_ADAPTIVE_ML]["avg_latency_ms"].iloc[0]
            < summary[summary["strategy"] == STRATEGY_ADAPTIVE]["avg_latency_ms"].iloc[0])


def test_utilisation_figures_agree(summary: pd.DataFrame) -> None:
    assert summary["network_utilization_pct"].nunique() == 1
    assert [round(v, 2) for v in summary["route_utilization_pct"]] == [43.33, 26.50, 23.17]


EXPECTED_SCENARIO_ROWS = {
    ("exp1_normal", STRATEGY_SHORTEST): (41.88, 0.00, 0.304),
    ("exp1_normal", STRATEGY_ADAPTIVE): (41.88, 0.00, 0.304),
    ("exp1_normal", STRATEGY_ADAPTIVE_ML): (41.88, 0.00, 0.304),
    ("exp2_congestion", STRATEGY_SHORTEST): (803.25, 5.75, 1.462),
    ("exp2_congestion", STRATEGY_ADAPTIVE): (85.65, 0.00, 0.608),
    ("exp2_congestion", STRATEGY_ADAPTIVE_ML): (85.65, 0.00, 0.608),
    ("exp3_packet_loss", STRATEGY_SHORTEST): (76.25, 23.50, 0.889),
    ("exp3_packet_loss", STRATEGY_ADAPTIVE): (85.65, 0.00, 0.608),
    ("exp4_latency", STRATEGY_SHORTEST): (165.37, 0.00, 0.724),
    ("exp4_latency", STRATEGY_ADAPTIVE): (85.65, 0.00, 0.608),
    ("exp5_traffic_spike", STRATEGY_SHORTEST): (238.43, 0.00, 0.512),
    ("exp5_traffic_spike", STRATEGY_ADAPTIVE): (238.43, 0.00, 0.512),
    ("exp5_traffic_spike", STRATEGY_ADAPTIVE_ML): (85.65, 0.00, 0.608),
    ("exp6_multi_congestion", STRATEGY_SHORTEST): (803.25, 5.75, 1.462),
    ("exp6_multi_congestion", STRATEGY_ADAPTIVE): (369.02, 1.75, 1.148),
    ("exp6_multi_congestion", STRATEGY_ADAPTIVE_ML): (369.02, 1.75, 1.148),
}


@pytest.mark.parametrize("key", sorted(EXPECTED_SCENARIO_ROWS))
def test_scenario_row_matches_the_docs(metrics: pd.DataFrame, key: tuple[str, str]) -> None:
    scenario, strategy = key
    row = metrics[(metrics["scenario"] == scenario) & (metrics["strategy"] == strategy)]
    assert len(row) == 1
    actual = (
        round(row["avg_latency_ms"].iloc[0], 2),
        round(row["packet_loss_pct"].iloc[0], 2),
        round(row["avg_route_cost"].iloc[0], 3),
    )
    assert actual == EXPECTED_SCENARIO_ROWS[key]


EXPECTED_FINAL_ROUTES = {
    ("exp2_congestion", STRATEGY_ADAPTIVE): "S -> B -> D -> F -> T",
    ("exp5_traffic_spike", STRATEGY_ADAPTIVE_ML): "S -> B -> D -> F -> T",
    ("exp6_multi_congestion", STRATEGY_ADAPTIVE): "S -> A -> D -> E -> T",
}


@pytest.mark.parametrize("key", sorted(EXPECTED_FINAL_ROUTES))
def test_final_route_matches_the_docs(metrics: pd.DataFrame, key: tuple[str, str]) -> None:
    scenario, strategy = key
    row = metrics[(metrics["scenario"] == scenario) & (metrics["strategy"] == strategy)]
    assert row["routes"].iloc[0].split(" | ")[-1] == EXPECTED_FINAL_ROUTES[key]


def test_headline_percentages_reproduce(summary: pd.DataFrame) -> None:
    latency = dict(zip(summary["strategy"], summary["avg_latency_ms"]))
    throughput = dict(zip(summary["strategy"], summary["throughput_mbps"]))

    adaptive_gain = (latency[STRATEGY_SHORTEST] - latency[STRATEGY_ADAPTIVE]) / latency[STRATEGY_SHORTEST] * 100
    ml_gain = (latency[STRATEGY_ADAPTIVE] - latency[STRATEGY_ADAPTIVE_ML]) / latency[STRATEGY_ADAPTIVE] * 100
    ml_vs_shortest = (latency[STRATEGY_SHORTEST] - latency[STRATEGY_ADAPTIVE_ML]) / latency[STRATEGY_SHORTEST] * 100
    thr_gain = (throughput[STRATEGY_ADAPTIVE_ML] - throughput[STRATEGY_SHORTEST]) / throughput[STRATEGY_SHORTEST] * 100

    assert round(adaptive_gain, 1) == 57.4
    assert round(ml_gain, 1) == 16.9
    assert round(ml_vs_shortest, 1) == 64.6
    assert round(thr_gain, 1) == 58.3

    loss = dict(zip(summary["strategy"], summary["packet_loss_pct"]))
    loss_drop = (loss[STRATEGY_SHORTEST] - loss[STRATEGY_ADAPTIVE_ML]) / loss[STRATEGY_SHORTEST] * 100
    assert round(loss_drop, 1) == 95.0


# ---------------------------------------------------------------------------
# Documentation files themselves
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("relative", REQUIRED_FILES)
def test_required_file_exists(relative: str) -> None:
    assert (PROJECT_ROOT / relative).exists(), f"missing {relative}"


@pytest.mark.parametrize("doc", ALL_DOCS, ids=lambda p: p.name)
def test_document_is_substantial(doc: Path) -> None:
    assert len(doc.read_text(encoding="utf-8")) > 1500


@pytest.mark.parametrize("doc", ALL_DOCS, ids=lambda p: p.name)
@pytest.mark.parametrize("token", ["TODO", "FIXME", "XXX", "Lorem ipsum", "<your"])
def test_document_has_no_placeholders(doc: Path, token: str) -> None:
    assert token not in doc.read_text(encoding="utf-8")


def test_experiments_doc_lists_every_run(metrics: pd.DataFrame) -> None:
    text = (DOCS_DIR / "experiments.md").read_text(encoding="utf-8")
    for scenario in metrics["scenario"].unique():
        assert scenario in text, f"{scenario} is missing from docs/experiments.md"
