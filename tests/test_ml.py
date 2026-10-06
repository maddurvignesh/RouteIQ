"""Unit and integration tests for the ML congestion-prediction module."""

from __future__ import annotations

import pytest

from config import (
    ML_COST_PENALTY,
    ML_FEATURES,
    ML_MIN_ACCEPTABLE_ACCURACY,
    ML_MODEL_PATH,
    ML_TARGET,
)
from core.adaptive_routing import AdaptiveRouter
from core.network_conditions import ConditionSimulator, ConditionType
from core.topology import build_predefined_topology
from ml.dataset import class_distribution, generate_dataset, load_dataset, save_dataset
from ml.model import (
    CLASS_LABELS,
    congestion_label,
    extract_features,
    features_to_row,
    severity_score,
    validate_dataset_columns,
)
from ml.predict import CongestionPredictor, load_predictor
from ml.train import train_model


# ---------------------------------------------------------------------------
# Feature / label definitions
# ---------------------------------------------------------------------------

def test_severity_score_bounds() -> None:
    assert severity_score(0.0, 0.0, 0.0) == pytest.approx(0.0)
    # fully saturated, very slow, very lossy link -> maximum severity
    assert severity_score(1.0, 1000.0, 1.0) == pytest.approx(1.0)
    assert 0.0 < severity_score(0.5, 50.0, 0.075) < 1.0


def test_congestion_label_thresholds() -> None:
    assert congestion_label(0.1, 5.0, 0.001) == 0     # Normal
    assert congestion_label(0.65, 40.0, 0.08) == 1    # Warning
    assert congestion_label(0.95, 90.0, 0.30) == 2    # Congested


def test_class_labels_cover_all_levels() -> None:
    assert CLASS_LABELS == {0: "Normal", 1: "Warning", 2: "Congested"}


def test_extract_features_matches_training_schema() -> None:
    attrs = {"latency": 40.0, "packet_loss": 0.05, "bandwidth": 500.0,
             "traffic": 400.0, "congestion": 0.8}
    features = extract_features(attrs, previous_level=1)
    assert set(features) == set(ML_FEATURES)
    assert features["bandwidth_utilization"] == pytest.approx(0.8)
    assert features["queue_length"] == pytest.approx(0.8 ** 2 * 60.0)
    assert features["previous_level"] == 1.0
    assert len(features_to_row(features)) == len(ML_FEATURES)


def test_extract_features_caps_utilization_at_one() -> None:
    attrs = {"latency": 10, "packet_loss": 0.01, "bandwidth": 100.0,
             "traffic": 500.0, "congestion": 0.5}
    assert extract_features(attrs)["bandwidth_utilization"] == 1.0


def test_validate_dataset_columns_rejects_bad_frame() -> None:
    with pytest.raises(ValueError):
        validate_dataset_columns(["latency_ms", "something_else"])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def test_generated_dataset_has_required_columns() -> None:
    frame = generate_dataset(n_samples=200, seed=1)
    validate_dataset_columns(frame.columns)
    assert len(frame) == 200
    assert set(frame[ML_TARGET].unique()) <= {0, 1, 2}


def test_generated_dataset_is_balanced_enough() -> None:
    frame = generate_dataset(n_samples=3000, seed=5)
    distribution = class_distribution(frame)
    assert len(distribution) == 3
    total = sum(distribution.values())
    assert all(count / total > 0.10 for count in distribution.values())


def test_previous_level_is_the_previous_observation() -> None:
    frame = generate_dataset(n_samples=500, seed=2)
    for _, group in frame.groupby("link_id"):
        labels = group[ML_TARGET].tolist()
        previous = group["previous_level"].tolist()
        assert previous[0] == 0.0
        assert previous[1:] == [float(v) for v in labels[:-1]]


def test_features_are_physically_consistent() -> None:
    frame = generate_dataset(n_samples=1000, seed=3)
    assert (frame["traffic_mbps"] <= frame["bandwidth_mbps"]).all()
    assert frame["bandwidth_utilization"].between(0.0, 1.0).all()
    assert (frame["packet_loss"] >= 0.0).all()
    assert (frame["latency_ms"] > 0.0).all()


def test_dataset_round_trip(tmp_path) -> None:
    frame = generate_dataset(n_samples=200, seed=4)
    path = tmp_path / "dataset.csv"
    save_dataset(frame, path)
    reloaded = load_dataset(path)
    assert len(reloaded) == len(frame)
    assert list(reloaded.columns) == list(frame.columns)


def test_missing_dataset_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.csv")


def test_too_small_dataset_rejected() -> None:
    with pytest.raises(ValueError):
        generate_dataset(n_samples=10, seed=1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def training_report() -> dict:
    if not ML_MODEL_PATH.exists():
        return train_model(save=True)
    from ml.train import load_training_report

    report = load_training_report()
    if report is None:
        return train_model(save=True)
    return report


def test_model_reaches_required_accuracy(training_report) -> None:
    assert training_report["accuracy"] >= ML_MIN_ACCEPTABLE_ACCURACY
    assert training_report["accuracy"] >= 0.90


def test_training_report_contents(training_report) -> None:
    assert training_report["dataset_size"] > 1000
    assert training_report["train_size"] + training_report["test_size"] == training_report["dataset_size"]
    assert len(training_report["confusion_matrix"]) == 3
    assert set(training_report["feature_importances"]) == set(ML_FEATURES)
    assert all(0.0 <= value <= 1.0 for value in training_report["feature_importances"].values())
    assert abs(sum(training_report["feature_importances"].values()) - 1.0) < 0.05


def test_feature_importances_are_sorted(training_report) -> None:
    values = list(training_report["feature_importances"].values())
    assert values == sorted(values, reverse=True)


def test_model_file_is_written(tmp_path) -> None:
    path = tmp_path / "model.joblib"
    report = tmp_path / "report.json"
    train_model(model_path=path, report_path=report, save=True)
    assert path.exists() and report.exists()


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

def test_predictor_loads_trained_model() -> None:
    predictor = load_predictor()
    assert predictor.available
    assert predictor.active
    assert predictor.error is None


def test_predictor_returns_valid_levels() -> None:
    predictor = load_predictor()
    healthy = {"latency": 10.0, "packet_loss": 0.005, "bandwidth": 1000.0,
               "traffic": 200.0, "congestion": 0.20}
    level = predictor.predict_link("X", "Y", healthy)
    assert level in CLASS_LABELS


def test_predictor_penalties_match_config() -> None:
    predictor = load_predictor()
    assert ML_COST_PENALTY == {0: 0.0, 1: 0.35, 2: 0.90}
    assert predictor.link_penalty("A", "B", {"latency": 5, "packet_loss": 0.001,
                                            "bandwidth": 1000, "traffic": 50,
                                            "congestion": 0.05}) == 0.0


def test_predictor_flags_a_saturated_link() -> None:
    predictor = load_predictor()
    saturated = {"latency": 60.0, "packet_loss": 0.25, "bandwidth": 100.0,
                 "traffic": 97.0, "congestion": 0.97}
    assert predictor.predict_link("A", "B", saturated) == 2
    assert predictor.link_penalty("A", "B", saturated) == pytest.approx(0.90)


def test_disabled_predictor_never_penalises() -> None:
    predictor = CongestionPredictor(enabled=False)
    assert not predictor.active
    assert predictor.available is False
    saturated = {"latency": 90.0, "packet_loss": 0.4, "bandwidth": 100.0,
                 "traffic": 99.0, "congestion": 0.97}
    assert predictor.link_penalty("A", "B", saturated) == 0.0
    assert predictor.predict_link("A", "B", saturated) == 0


def test_predictor_handles_missing_model(tmp_path) -> None:
    predictor = CongestionPredictor(model_path=tmp_path / "missing.joblib")
    assert not predictor.available
    assert predictor.error is not None and "not found" in predictor.error
    assert predictor.link_penalty("A", "B", {}) == 0.0


def test_predict_path_reports_every_hop() -> None:
    topo = build_predefined_topology()
    predictor = load_predictor()
    reports = predictor.predict_path(["S", "A", "C", "T"], topo.link_attributes)
    assert len(reports) == 3
    assert reports[0]["link"] == "S-A"
    assert all(r["state"] in CLASS_LABELS.values() for r in reports)


def test_predictor_summary_block() -> None:
    predictor = load_predictor()
    summary = predictor.summary()
    assert summary["active"] is True
    assert summary["features"] == ML_FEATURES
    assert set(summary["penalties"]) == {0, 1, 2}


def test_predictor_reset_clears_state() -> None:
    predictor = load_predictor()
    predictor.predict_link("A", "B", {"latency": 10, "packet_loss": 0.01,
                                      "bandwidth": 100, "traffic": 50,
                                      "congestion": 0.5})
    assert predictor.prediction_count == 1
    predictor.reset()
    assert predictor.prediction_count == 0
    assert predictor.summary()["tracked_links"] == 0


def test_inspect_link_does_not_change_state() -> None:
    predictor = load_predictor()
    predictor.reset()
    attrs = {"latency": 30.0, "packet_loss": 0.05, "bandwidth": 500.0,
             "traffic": 400.0, "congestion": 0.8}
    predictor.inspect_link("A", "B", attrs)
    assert predictor.summary()["tracked_links"] == 0


# ---------------------------------------------------------------------------
# ML-assisted routing integration
# ---------------------------------------------------------------------------

def test_ml_strategy_avoids_a_flagged_backbone() -> None:
    topo = build_predefined_topology()
    predictor = load_predictor()
    predictor.reset()
    ConditionSimulator(topo).apply(
        ConditionType.CONGESTION, links=[("S", "A"), ("A", "C"), ("C", "T")], severity=1.0
    )
    router = AdaptiveRouter(topo, penalty_fn=predictor.link_penalty)
    plain = router.route("S", "T", strategy="Adaptive Routing")
    guided = router.route("S", "T", strategy="Adaptive + ML")
    assert plain.path == ("S", "B", "D", "F", "T")
    assert guided.path == ("S", "B", "D", "F", "T")


def test_ml_gives_an_early_warning_on_a_single_spiking_link() -> None:
    """The core claim of the ML component: it reacts before the cost formula."""
    topo = build_predefined_topology()
    predictor = load_predictor()
    predictor.reset()
    ConditionSimulator(topo).apply(
        ConditionType.TRAFFIC_SPIKE, links=[("S", "A")], severity=1.0
    )
    router = AdaptiveRouter(topo, penalty_fn=predictor.link_penalty)
    plain = router.route("S", "T", strategy="Adaptive Routing")
    guided = router.route("S", "T", strategy="Adaptive + ML")

    assert plain.path == ("S", "A", "C", "T")      # cost formula is not convinced yet
    assert guided.path != plain.path               # ML has already reacted
    assert guided.path == ("S", "B", "D", "F", "T")
    assert guided.cost > plain.cost                # the penalty is really applied


def test_router_still_works_with_ml_disabled() -> None:
    topo = build_predefined_topology()
    router = AdaptiveRouter(topo, penalty_fn=None)
    result = router.route("S", "T", strategy="Adaptive + ML")
    assert result.path == ("S", "A", "C", "T")
    assert result.cost == pytest.approx(0.304, abs=0.01)
