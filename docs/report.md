# Adaptive Routing Based on Real-Time Network Conditions

**Project report** — 3rd Year, Computer Networks & Artificial Intelligence / Machine Learning

> Prepared from the code and results in this repository. Every figure quoted
> below can be regenerated with the commands in the *Appendix*; no number in
> this report was invented.

---

## Abstract

Classical routing protocols such as OSPF and RIP choose a path from a static
metric — usually hop count — and keep using it until the topology itself
changes. Congestion, latency spikes and packet loss are invisible to that
metric, so traffic continues to be pushed through a path that has become the
worst path in the network. This project implements and evaluates an **adaptive
routing decision system** that recomputes routes from *live* link conditions,
and extends it with a **machine-learning congestion predictor** that can act
before the reactive heuristic has seen enough evidence.

The system is a deterministic network simulation built in Python with
NetworkX, scikit-learn, pandas and Streamlit. A hand-designed 8-router,
13-link topology carries traffic between `S` and `T`; a condition simulator
injects six types of degradation; a packet simulator with explicit
transmission delay, non-linear queueing delay, packet loss and retransmission
produces the performance figures. Three strategies are compared over six
scenarios (18 runs, 7 200 packets): hop-count shortest path, adaptive routing
on a four-term weighted cost function, and the same adaptive router augmented
with a Random-Forest congestion classifier.

Adaptive routing reduced average end-to-end latency by **57.4 %** (354.7 ms →
151.0 ms), packet loss by **95.0 %** (5.83 % → 0.29 %) and raised delivery rate
from **94.2 %** to **99.7 %**. Adding the ML penalty further reduced latency by
**16.9 %** (151.0 ms → 125.6 ms) and cut the share of steps spent on a
congested path from 26.7 % to **13.3 %**, by triggering a proactive reroute in
the one scenario where the reactive heuristic alone did not act.

**Keywords:** adaptive routing, link-state, congestion prediction, Random
Forest, network simulation, real-time metrics.

---

## 1. Introduction

### 1.1 Background

A routing protocol answers a single question: *which next-hop should carry
this traffic?* Classical answers are computed from topology alone. A link-state
protocol floods link availability, runs Dijkstra, and installs the minimum-hop
(or minimum-static-cost) path. The path is recalculated only when a link goes
up or down.

Real networks rarely fail outright; they **degrade**. Buffers fill, queues grow,
delay rises, packets are dropped by overflow, and available bandwidth falls —
while every link remains "up". A protocol that only watches topology sees none
of this.

### 1.2 Motivation

The mismatch between *static metric* and *dynamic reality* produces the
familiar symptom of a congested network: the link everybody agrees is "the
shortest path" is precisely the link that is saturated, and all traffic is
pinned to it. Load balancing across alternate paths, reacting to measured
delay, and predicting congestion before it fully develops are the three
standard responses; this project implements all three in one system.

### 1.3 Problem statement

> Design and evaluate a routing mechanism that selects paths using **real-time
> network conditions**, and determine whether an ML-based congestion predictor
> adds measurable value over a purely reactive weighted-cost rule.

### 1.4 Objectives

1. Model a network of routers and links with live, mutable state.
2. Implement a load-oblivious baseline (hop-count shortest path) for
   comparison.
3. Implement an adaptive cost function from four normalised real-time
   metrics and select minimum-cost paths with it.
4. Simulate packet transmission (delay, queueing, loss, retransmission) so
   that performance is measured, not asserted.
5. Train a supervised classifier to predict per-link congestion state and
   inject its output as a routing penalty.
6. Compare the three strategies on controlled scenarios and report honest
   results, including the cases where the ML module does *not* help.

### 1.5 Scope and limitations

This is a **simulation and decision system**, not a production router or a
packet-level emulator such as NS-3. It runs on a single machine; no packets
leave the process. The topology has 8 routers and 13 links; three strategies
are compared; the ML model is a classical ensemble trained on a generated
dataset. Conclusions are therefore qualitative.

---

## 2. Related concepts

The design draws on four established ideas.

**Distance-vector and link-state routing.** RIP (distance vector) and OSPF
(link state) both compute a path from a metric that does not change with load.
The baseline strategy in this project reproduces exactly that behaviour.

**Dynamic / adaptive routing.** Routing on measured delay — as in delay-based
IGP metrics and end-to-end measurements — is standard practice: an operator
manually raises the OSPF cost of a slow link so traffic avoids it. The project
automates that idea by recomputing the metric at run time from simulated
telemetry.

**Equal-cost multi-path (ECMP) and load balancing.** Modern networks spread
traffic over several equal-cost next hops. This project shows the *decision*
side of the same problem: choosing which of several available paths should
carry the load when their costs are not equal.

**Learning-based traffic engineering.** Predicting congestion and pre-empting
it is an active research direction. Here a small, explainable supervised
classifier is used rather than a deep model, because the deliverable must be
auditable by an examiner.

Nothing in this report is attributed to a source that was not used; no
external dataset was downloaded.

---

## 3. System design

### 3.1 Architecture

The system is layered so that concerns never leak into each other:

```
presentation   →  visualization/  (Streamlit pages, Plotly charts)
orchestration  →  simulation/     (engine, scenarios, metrics, experiments)
routing        →  core/routing.py, core/adaptive_routing.py
network model  →  core/topology.py, core/network_conditions.py,
                  core/packet_simulator.py
machine learning → ml/            (dataset, model, train, predict)
configuration  →  config.py       (every constant in one place)
```

The ML module is connected to the router through a single callback with the
signature `penalty(u, v, attributes) -> float`. The router never imports the ML
package, so the system still works when the model file is absent — it simply
falls back to plain adaptive routing.

### 3.2 Network model

Eight routers (`S, A, B, C, D, E, F, T`) are joined by 13 links. Each link
carries `bandwidth`, `latency`, `packet_loss`, `congestion`, `traffic` and
`reliability`. Two invariants are enforced:

* every link keeps a `base_*` copy of its healthy values, so a fault can always
  be undone exactly;
* `congestion == traffic / bandwidth`, so the routing cost, the packet
  simulator and the ML features describe the same physical quantity.

The hop-count shortest path is `S → A → C → T` (3 hops); its main alternative
is `S → B → D → F → T` (4 hops).

### 3.3 Network-condition simulator

Six degradation types are supported (`CONGESTION`, `MULTI_CONGESTION`,
`TRAFFIC_SPIKE`, `HIGH_LATENCY`, `HIGH_PACKET_LOSS`, `LOW_BANDWIDTH`) plus
`NORMAL` to restore health. A severity parameter in `(0, 1]` scales the effect.
When no explicit link list is given, the simulator targets the current
hop-count shortest path — the experiment design deliberately "congests the
route a naive protocol would keep using". Congestion is clamped at 0.97 so the
queueing curve never becomes singular.

### 3.4 Routing algorithms

**Baseline.** Dijkstra over hop count.

**Adaptive.** Every link receives a dynamic cost built from four terms, each
normalised into `[0, 1]`:

| Term | Definition |
|---|---|
| latency | `min(latency_ms / 100, 1)` |
| congestion | `congestion` (already 0…1) |
| packet loss | `min(packet_loss / 0.20, 1)` |
| bandwidth | `1 − min(bandwidth / 1000, 1)` |

```
cost(link) = 0.35·latency + 0.30·congestion + 0.25·loss + 0.10·bandwidth
             [+ ML penalty]
```

The weights are validated to sum to 1.0. Dijkstra is then run with this cost
as the edge weight, so the router returns the minimum-cost path **under current
conditions**. The four cheapest alternative paths are also enumerated, purely
for explanation.

**Adaptive + ML.** Identical, plus a per-link penalty of `0.00 / 0.35 / 0.90`
for a predicted state of Normal / Warning / Congested. Penalties are computed
once per routing decision for all links, so the decision is internally
consistent.

### 3.5 Packet simulator

A seeded, deterministic per-hop model. It is not a TCP/IP stack; it reproduces
the three effects that matter to routing:

* **transmission delay** — `size / bandwidth` per hop;
* **queueing delay** — `k · c / (1 − c)` with `c` the congestion and
  `k = 12 ms`, the classic non-linear buffer curve;
* **loss** — `1 − (1 − loss) × reliability` per hop, with up to two
  retransmissions after a 40 ms timeout.

Each hop receives a seeded ±10 % jitter. Defaults: 80 packets of 8 KB per
step, 5 ms inter-arrival, seed 42.

### 3.6 Machine-learning module

**Dataset.** Real router telemetry is not publicly available, so 5 000
observations are generated from 100 simulated link streams × 50 steps. Link
utilisation performs a random walk, so samples are temporally correlated as
they are on a real link; latency, loss and queue length are derived from
utilisation through queueing relationships plus measurement noise.

**Labels.** A documented rule, so the ground truth is auditable:

```
severity = 0.40·utilisation + 0.30·min(latency/100, 1) + 0.30·min(loss/0.15, 1)
severity ≥ 0.65 → Congested   severity ≥ 0.40 → Warning   else → Normal
```

Class balance: 2 280 / 1 360 / 1 360.

**Features (6).** `latency_ms`, `packet_loss`, `bandwidth_utilization`,
`traffic_mbps`, `queue_length`, `previous_level`. `previous_level` is the
label of the previous observation of the same link — always available to a
router, and it is what makes the prediction temporal rather than per-sample.
The feature list and labelling rule live in one module that both training and
inference import, so the two can never disagree on the input format.

**Model.** A Random Forest (200 trees, depth 12, `min_samples_leaf = 2`,
seed 42) trained on a 75/25 split. Random Forest was chosen because it trains
in under a second on 5 000 rows, needs no feature scaling, reports feature
importances that justify the feature set, tolerates the injected noise, and is
explainable in a viva. Training aborts if accuracy < 0.80.

**Inference.** `CongestionPredictor.link_penalty(u, v, attrs)` returns the
penalty for the predicted class. If the model file is missing it returns 0.0
and the router degrades to plain adaptive routing.

---

## 4. Implementation

The project contains about 5 900 lines of Python across 26 modules and 9 test
files.

| Package | Modules | Responsibility |
|---|---|---|
| `core` | 5 | topology, conditions, baseline routing, adaptive routing, packet simulation |
| `ml` | 4 | dataset generation, shared definitions, training, live prediction |
| `simulation` | 4 | metrics, scenarios, experiment runner, engine facade |
| `visualization` | 3 | dashboard pages, charts, topology figure |
| `tests` | 9 | 162 tests (unit + headless dashboard integration) |
| root | 2 | `config.py` (all constants), `app.py` (Streamlit entry point) |

Design decisions worth noting:

* **Facade.** `SimulationEngine` owns all mutable state, so the dashboard and
  the tests never wire components together by hand; the Streamlit app keeps a
  single object in `st.session_state`.
* **No magic numbers.** Weights, normalisation bounds, penalties and seeds all
  live in `config.py`, with dataclass validation (e.g. weights must sum to 1).
* **Explainability.** `RouteResult` carries a human-readable `reason` and the
  rejected alternatives; `link_cost_breakdown` returns each term of the cost
  formula so the dashboard can show *why* a path won.
* **Graceful degradation.** Every component works when its optional input
  (model file, random topology, saved results) is absent.
* **Testability.** The dashboard is tested by running the real `app.py`
  headlessly with Streamlit's `AppTest`, so a broken page fails the suite.

---

## 5. Experiments and results

### 5.1 Setup

Six scenarios × three strategies = 18 runs, 5 steps each, 80 packets per step,
seed 42, one fixed topology. Each run rebuilds the topology so runs cannot
contaminate each other. Full tables are in `docs/experiments.md`.

### 5.2 Aggregate results

| Strategy | Avg latency (ms) | Loss (%) | Throughput (Mbps) | Delivery (%) | Steps on a congested path (%) |
|---|---:|---:|---:|---:|---:|
| Shortest Path | 354.74 | 5.83 | 25.32 | 94.17 | 40.00 |
| Adaptive Routing | 151.05 | 0.29 | 36.23 | 99.71 | 26.67 |
| Adaptive + ML | **125.58** | **0.29** | **40.07** | **99.71** | **13.33** |

Network utilisation is identical for all three strategies (31.32 %) — the same
traffic is offered every time; only the chosen route differs. Route
utilisation falls from 43.33 % to 23.17 %: the adaptive strategies move load
off the hot links instead of adding load.

### 5.3 Interpretation

1. **Adaptivity is the big win.** −57.4 % latency, −95.0 % loss, +5.5 points
   of delivery rate versus hop count. The baseline's failure is stark in
   `exp2` (803 ms vs 86 ms) and `exp3` (23.5 % loss vs 0 %).
2. **No cost on a healthy network.** In `exp1` all three strategies choose the
   identical route with identical metrics, so adaptation does not degrade the
   normal case.
3. **The ML module earns its place narrowly and honestly.** It changes the
   outcome in exactly one of six scenarios (`exp5`, a single-link traffic
   spike). There the weighted formula alone leaves the router on the backbone
   at 238 ms, while the ML penalty triggers a reroute at 86 ms — a 64 %
   latency reduction and the reason the aggregate ML latency (125.6 ms) beats
   plain adaptive (151.0 ms).
4. **The ML route is not always the cheapest *right now*.** In `exp5` the ML
   route's current weighted cost is 0.608 versus 0.512 for the route it
   abandoned: the classifier is buying a *future* improvement. This is the
   defining difference between a reactive and a proactive signal, and it is
   reported rather than hidden.
5. **Limits are visible.** In `exp6`, where the backbone *and* its alternative
   are both degraded, all strategies degrade to ~369 ms. Adaptive routing can
   find the least-bad path; it cannot create capacity.

---

## 6. Conclusion

The project demonstrates, in a controlled and fully reproducible simulation,
that routing decisions built from real-time link conditions substantially
outperform hop-count routing: latency falls by more than half, packet loss by
almost 95 %, and delivery rate rises from 94 % to 99.7 %. A supervised
congestion classifier adds a proactive component that improves on the reactive
rule by a further 16.9 % in the scenario designed to isolate it, and halves the
share of time spent on a congested path.

The work also establishes the boundaries honestly: on a healthy network all
strategies behave identically; when the whole neighbourhood is degraded,
adaptation only finds the least-bad option; and the ML module changes the
outcome in one scenario out of six rather than across the board.

### Future work

1. **Weight sensitivity / learning the weights.** Replace the hand-chosen
   `0.35/0.30/0.25/0.10` with a tuned or learned set, and sweep them to show
   robustness.
2. **Real traces.** Train on public backbone/edge traffic traces instead of a
   generated dataset, and report accuracy against measured congestion.
3. **Larger topologies.** Run the random-topology generator across many seeds
   and sizes (20–100 routers) and report scaling behaviour.
4. **More baselines.** Add weighted-random ECMP and a simple QoS-routing
   baseline for a fairer comparison.
5. **Feedback loop.** Feed simulated packet outcomes back into the link state
   (self-induced congestion) so the system is closed-loop rather than
   condition-driven.
6. **Deployment.** Express the cost function as an OSPF/IS-IS external metric
   or as an SDN controller application, and validate it in a Mininet testbed.

---

## Appendix — how to reproduce this report

```bash
pip install -r requirements.txt

python -m ml.dataset        # writes data/congestion_dataset.csv (5 000 rows)
python -m ml.train          # writes models/congestion_model.joblib + report
python -m pytest            # 162 tests

streamlit run app.py        # dashboard → "Performance Comparison"
                            #            → "Run all experiments"
```

Outputs: `results/experiment_metrics.csv` (18 runs),
`results/comparison_summary.csv` (3-row method comparison),
`models/training_report.json` (accuracy, confusion matrix, importances).

| Document | Contents |
|---|---|
| `docs/architecture.md` | layering, module map, data flow, dependency rules |
| `docs/methodology.md` | algorithms, formulas, dataset, metrics |
| `docs/experiments.md` | full result tables and per-scenario analysis |
| `docs/presentation.md` | slide-by-slide presentation content |
| `docs/viva.md` | anticipated examiner questions and answers |
