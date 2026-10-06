# Architecture

> **Scope.** This is a *network simulation and adaptive routing decision
> system*. No packets leave the machine; every number produced by the project
> comes from code in this repository.

---

## 1. Layered view

```
┌──────────────────────────────────────────────────────────────────────┐
│  PRESENTATION      app.py  →  visualization/dashboard.py (6 pages)  │
│                     visualization/charts.py   (Plotly figures)      │
│                     visualization/topology.py  (graph figure)       │
└───────────────▲──────────────────────────────────────────────────────┘
                │  only SimulationEngine + pandas DataFrames
┌───────────────┴──────────────────────────────────────────────────────┐
│  ORCHESTRATION     simulation/engine.py  (SimulationEngine facade)   │
│                    simulation/experiments.py  (6 x 3 test matrix)    │
│                    simulation/scenarios.py     (timelines)           │
│                    simulation/metrics.py       (StepRecord, KPIs)    │
└───────▲──────────────────────▲──────────────────────▲────────────────┘
        │                      │                      │
┌───────┴────────┐   ┌─────────┴─────────┐   ┌────────┴────────────┐
│  ROUTING       │   │  NETWORK MODEL    │   │  ML                 │
│  core/routing  │   │  core/topology    │   │  ml/dataset         │
│  core/adaptive │   │  core/conditions  │   │  ml/model, ml/train │
│  _routing.py   │   │  core/packet_     │   │  ml/predict         │
│                │   │  simulator.py     │   │                     │
└───────▲────────┘   └─────────▲─────────┘   └────────┬────────────┘
        │                      │                      │
        └──────────────────────┴─────────── PenaltyFunction callback ──┘
┌──────────────────────────────────────────────────────────────────────┐
│  CONFIGURATION     config.py   (weights, bounds, paths, seeds)       │
└──────────────────────────────────────────────────────────────────────┘
```

### Dependency rules (enforced by review and by the test suite)

| Layer | May import | Must **not** import |
|---|---|---|
| `core` | `config`, `networkx`, stdlib | `ml`, `simulation`, `visualization`, `streamlit` |
| `ml` | `config`, `core.topology` (read-only), `sklearn` | `core.routing`, `simulation`, `visualization` |
| `simulation` | `core`, `ml.predict` (via `engine` only), `config` | `visualization`, `streamlit` |
| `visualization` | `simulation.engine`, `config`, `plotly`, `streamlit` | `core.*` directly (goes through the engine) |
| `app.py` | `visualization.dashboard` | anything else |

The one intentional cross-link is `ml.predict.CongestionPredictor.link_penalty`,
which is passed into `core.adaptive_routing.AdaptiveRouter` as a plain
**`PenaltyFunction` callback**. The router never imports the ML package, so the
system still runs when `models/congestion_model.joblib` is missing.

---

## 1b. The theme layer

Streamlit ships one flat colour scheme, so `visualization/theme.py` injects a
`<style>` block (`st.markdown(..., unsafe_allow_html=True)`) plus a small
stack of `position: fixed` decorative divs.

| Layer | Technique |
|---|---|
| base | `radial-gradient` on the fixed container |
| aurora | rotating `conic-gradient`, `blur(90px)`, 68 s loop |
| orbs ×3 | blurred circles on independent `translate3d`+`scale` drift paths |
| grid | two `linear-gradient`s panned via `background-position` |
| noise | inlined SVG `feTurbulence` data URI (no extra request) |
| vignette | radial darkening so text stays readable over the motion |

Four palettes are defined in `THEMES`: **Aurora** (default), **Sunset**,
**Arctic** and **Minimal** (flat, animation disabled). All colours live in CSS
custom properties declared once per theme, so the shared rules below them never
change. Users pick a theme in the sidebar; deployments can pin one with the
`ARX_THEME` environment variable.

Two details that matter:

* **The `<style>` tag must be the only thing in its markdown call.** Mixed with
  other HTML, Streamlit's sanitiser markdown-parses the CSS and prints it on the
  page. Hence the two separate `st.markdown` calls in `inject()`.
* **Motion respects the user.** `prefers-reduced-motion: reduce` and the
  `Minimal` theme both switch every animation off.

Plotly figures are given transparent `paper_bgcolor`/`plot_bgcolor` and the
`plotly_dark` template so they blend into the glass cards instead of sitting on
a white rectangle.

---

## 2. Module map

| Package / file | Responsibility | Key types |
|---|---|---|
| `config.py` | Every tunable constant: cost weights, normalisation bounds, ML penalties, seeds, paths | `CostWeights`, `PacketConfig` |
| `core/topology.py` | Graph of routers/links with live **and** baseline (`base_*`) attributes | `Topology`, `build_predefined_topology`, `build_random_topology` |
| `core/network_conditions.py` | Applies/repairs simulated faults on the live attributes | `ConditionType`, `ConditionSimulator`, `ConditionEvent` |
| `core/routing.py` | Load-oblivious baseline: hop-count shortest path + alternative enumeration | `shortest_path_route`, `enumerate_routes`, `RouteResult`, `Alternative` |
| `core/adaptive_routing.py` | Dynamic link cost, path cost, route selection for the three strategies | `link_cost`, `link_cost_breakdown`, `path_cost`, `AdaptiveRouter` |
| `core/packet_simulator.py` | Deterministic packet transmission with delay, queueing, loss, retransmission | `PacketSimulator`, `Packet`, `TransmissionReport` |
| `ml/model.py` | Feature vector definition + documented labelling rule (shared by train & inference) | `features_from_attributes`, `label_from_severity` |
| `ml/dataset.py` | Synthetic labelled dataset with temporally correlated links | `ensure_dataset`, `generate_dataset`, `class_distribution` |
| `ml/train.py` | Train/evaluate the classifier, persist model + report | `train_model`, `load_training_report` |
| `ml/predict.py` | Live inference; maps predicted class → routing penalty | `CongestionPredictor`, `load_predictor` |
| `simulation/metrics.py` | Per-step record and aggregate KPIs | `StepRecord`, `build_step_record`, `compute_metrics` |
| `simulation/scenarios.py` | The six experiment timelines | `Scenario`, `StepAction`, `build_timeline` |
| `simulation/experiments.py` | Runs one *(scenario × strategy)* or the whole matrix, saves CSVs | `run_experiment`, `run_all_experiments`, `summarise` |
| `simulation/engine.py` | Facade used by the dashboard and integration tests | `SimulationEngine` |
| `visualization/topology.py` | Interactive Plotly graph + link table | `build_topology_figure`, `build_link_table` |
| `visualization/charts.py` | All other figures (metric bars, timelines, confusion matrix, …) | `metric_bar`, `confusion_matrix_figure`, … |
| `visualization/theme.py` | Injected CSS: animated background + glass surfaces | `THEMES`, `build_css`, `inject` |
| `visualization/dashboard.py` | Six pages + sidebar | `run`, `render_sidebar` |
| `tests/` | 277 unit + integration tests | — |

---

## 3. Data flow

### 3.1 One routing decision

```
Topology (live link attributes)
        │
        ├── strategy == "Shortest Path"
        │         └── core/routing.shortest_path_route()   ← static hop count
        │
        └── strategy ∈ {"Adaptive Routing", "Adaptive + ML"}
                  │
                  ├─(a) normalise each link: latency/100ms, loss/0.20,
                  │     congestion, 1 − bandwidth/1000Mbps
                  ├─(b) weighted sum → w1·L + w2·C + w3·P + w4·B
                  ├─(c) + ml_penalty(u,v)        ← only when the ML hook is set
                  ├─(d) Dijkstra over the dynamic link cost
                  └─(e) enumerate the k cheapest alternatives for explanation
                              │
                              ▼
                        RouteResult { path, cost, reason, alternatives }
                              │
                              ▼
                   PacketSimulator.send(route) → TransmissionReport
                              │
                              ▼
                   StepRecord → SimulationMetrics (latency, loss, throughput …)
```

### 3.2 One experiment

`simulation/experiments.run_experiment` builds a **fresh** topology and a
**fresh** `ConditionSimulator` per run so runs never contaminate each other:

```
for action in scenario.steps:
        1. conditions.apply(...)            # mutate live link attributes
        2. router.route(src, dst, strategy) # recompute the decision
        3. packets.send(route)              # simulate 80 packets
        4. build_step_record(...)           # capture KPIs for this step
aggregate → SimulationMetrics → DataFrame row (written to results/)
```

`run_all_experiments` repeats this for the **6 scenarios × 3 strategies = 18
runs** and writes

* `results/experiment_metrics.csv` — one row per run (18 rows × 18 columns)
* `results/comparison_summary.csv` — averages per strategy (3 rows)

### 3.3 The ML pipeline

```
ml/dataset.py  ──►  data/congestion_dataset.csv  (5 000 rows)
                              │
                        ml/train.py  (RandomForest, 75/25 split, seed 42)
                              │
              ┌───────────────┴────────────────┐
              ▼                                ▼
 models/congestion_model.joblib      models/training_report.json
              │                       (accuracy, confusion matrix,
              ▼                        feature importances)
   ml/predict.CongestionPredictor
              │   link_penalty(u, v, attrs) -> {0.0 | 0.35 | 0.90}
              ▼
   core.adaptive_routing.AdaptiveRouter  (injected as PenaltyFunction)
```

---

## 4. State management

`SimulationEngine` holds **all** mutable simulation state:

| Field | Meaning |
|---|---|
| `topology` | live graph (mutated by conditions) |
| `conditions` | `ConditionSimulator` — remembers the `base_*` values |
| `router` | `AdaptiveRouter` (rebuilt whenever weights/ML hook change) |
| `predictor` | `CongestionPredictor` (loaded once, reset on `reset()`) |
| `_route` | the currently selected route |
| `_history` | list of `StepRecord` (drives the timeline charts) |
| `_packets` | every `Packet` ever transmitted (drives the latency histogram) |
| `_events` | human-readable log lines for the sidebar |

The dashboard stores exactly one object in `st.session_state`:
`st.session_state["engine"]`. Everything else (experiment frames, ML report)
is cached in separate keys so that a re-run of the script never loses user
work.

---

## 5. Directory layout

```
adaptive-routing/
├── app.py                    # Streamlit entry point
├── config.py                 # all constants
├── requirements.txt
├── pytest.ini
├── core/                     # network model + routing algorithms
├── ml/                       # dataset, training, inference
├── simulation/               # scenarios, metrics, experiments, engine
├── visualization/            # dashboard + charts
├── tests/                    # 162 tests
├── data/                     # generated dataset (CSV)
├── models/                   # trained model + training report
├── results/                  # experiment CSVs
└── docs/                     # this documentation
```

---

## 6. Extension points

| Want to… | Change |
|---|---|
| re-weight the cost function | `config.CostWeights` (validated to sum to 1.0) |
| add a network fault | `core.network_conditions.ConditionType` + `_degrade_link` |
| add an experiment | register a `Scenario` in `simulation/scenarios.py` |
| swap the ML model | `ml.train.train_model` (any sklearn classifier with `feature_importances_` or permutation importance) |
| change the penalty mapping | `config.ML_COST_PENALTY` |
| add a dashboard page | add a function in `visualization/dashboard.py`, register it in `PAGES` and `handlers` |
