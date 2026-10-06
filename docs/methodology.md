# Methodology

This document describes *how* the system works: the network model, the routing
algorithms, the packet simulation, the machine-learning module and the way
every reported number is produced.

---

## 1. Problem statement

A classic link-state / distance-vector protocol (OSPF, RIP) computes a path
from a **static** metric — normally hop count — and keeps using it until the
*topology itself* changes. Link congestion, latency spikes and packet loss are
invisible to that metric. The consequence is well known: the protocol keeps
funnelling traffic through the "shortest" path while that path degrades,
because cost and congestion are simply not the same thing.

The project therefore asks:

> **Can a routing decision that reacts to *real-time* link conditions — and,
> additionally, to a machine-learning prediction of those conditions — deliver
> lower delay, lower loss and higher delivery rate than hop-count routing?**

It answers the question inside a **discrete-step network simulation** with
three routing strategies compared on identical scenarios.

---

## 2. Network model

### 2.1 Topology

The default topology is a hand-designed 8-router, 13-link network:

```
        S ---- A ---- C ---- T          backbone (3 hops, 10 ms/link)
        |      |      |      |
        B ---- D ---- E      |
              |      |      |
              F ----+------+
```

| Link | Bandwidth (Mbps) | Latency (ms) | Loss | Congestion | Reliability |
|---|---|---|---|---|---|
| S-A | 1000 | 10 | 0.005 | 0.20 | 0.995 |
| S-B | 500 | 30 | 0.010 | 0.25 | 0.99 |
| A-C | 1000 | 10 | 0.005 | 0.20 | 0.995 |
| A-D | 300 | 35 | 0.020 | 0.30 | 0.98 |
| B-D | 500 | 20 | 0.010 | 0.25 | 0.99 |
| C-D | 400 | 28 | 0.015 | 0.28 | 0.985 |
| C-E | 300 | 40 | 0.020 | 0.30 | 0.98 |
| C-T | 1000 | 10 | 0.005 | 0.20 | 0.995 |
| D-E | 500 | 25 | 0.010 | 0.25 | 0.99 |
| D-F | 800 | 12 | 0.005 | 0.15 | 0.995 |
| E-F | 600 | 18 | 0.008 | 0.20 | 0.99 |
| E-T | 500 | 30 | 0.010 | 0.25 | 0.99 |
| F-T | 800 | 14 | 0.005 | 0.15 | 0.995 |

*Hop-count shortest path:* **S → A → C → T** (3 hops).
*Main alternate:* **S → B → D → F → T** (4 hops).
There are **20** loop-free S→T paths of ≤ 5 hops; the router exposes the four
cheapest ones as "alternatives" on the dashboard.

`build_random_topology(seed)` produces alternative graphs for stress testing;
the experiments always use the predefined one so results are reproducible and
comparable.

### 2.2 Link state invariant

Every link stores both a **baseline** (`base_bandwidth`, `base_latency`, …)
and a **live** copy. Conditions only mutate the live copy, which lets
`Topology.reset_conditions()` restore a healthy network exactly.

A deliberate invariant is enforced throughout the model:

```
congestion  ==  traffic / bandwidth        (utilisation)
```

so `traffic = congestion × bandwidth` always holds. This keeps the routing
cost, the packet simulator and the ML features talking about the same physical
quantity.

### 2.3 Network conditions

`ConditionSimulator` applies seven condition types. Severity `s ∈ (0, 1]`
controls how hard the link is hit:

| Condition | Effect on the affected links |
|---|---|
| `NORMAL` | restore live values to baseline |
| `CONGESTION` | `congestion += 0.75·s`, `latency ×= (1 + 2·s)`, `loss += 0.15·s` |
| `MULTI_CONGESTION` | same rule applied to several links at once |
| `TRAFFIC_SPIKE` | `congestion += 0.75·s`, `latency ×= (1 + s)` |
| `HIGH_LATENCY` | `latency ×= (1 + 5·s)` (loss untouched — isolates delay) |
| `HIGH_PACKET_LOSS` | loss driven to `0.05 + 0.15·s` |
| `LOW_BANDWIDTH` | `bandwidth ×= (1 − 0.7·s)` (floor 50 Mbps), `congestion += 0.5·s` |

Congestion is clamped at `0.97` so the queueing curve below never reaches its
asymptote (which would make delay infinite).

When no explicit link list is given, the simulator degrades the links of the
**current hop-count shortest path** — precisely the "congest the route a naive
protocol would keep using" experiment we want to show.

---

## 3. Routing strategies

### 3.1 Baseline — Shortest Path

`core/routing.shortest_path_route` runs Dijkstra on hop count and returns the
first 3-hop path. It never looks at latency, loss or congestion. This is the
control group.

### 3.2 Adaptive cost function

Each link gets a **dynamic cost** built from four normalised terms:

| Term | Definition | Rationale |
|---|---|---|
| `L` | `min(latency_ms / 100, 1)` | 100 ms is treated as "bad" |
| `C` | `congestion ∈ [0, 1]` | already normalised by definition |
| `P` | `min(packet_loss / 0.20, 1)` | 20 % loss is treated as "bad" |
| `B` | `1 − min(bandwidth / 1000, 1)` | low capacity is penalised |

```
cost(link) = 0.35·L + 0.30·C + 0.25·P + 0.10·B + penalty_ML(link)
```

All four weights live in `config.CostWeights` and are validated (they must
sum to 1.0). Because every term is in `[0, 1]`, the base cost of a link lies in
`[0, 1]` and the ML penalty is an *additive* term on the same order of
magnitude — a penalty of `0.90` is roughly the cost of a completely saturated,
100 ms, 20 %-loss link.

Path cost = sum of link costs along the path; Dijkstra is run with
`link_cost` as the NetworkX edge-weight function, so the router returns the
**minimum-cost** path under the *current* conditions.

### 3.3 The three strategies

| Strategy | Cost function used by Dijkstra |
|---|---|
| `Shortest Path` | hop count (static) |
| `Adaptive Routing` | the four weighted terms |
| `Adaptive + ML` | the four weighted terms **+ ML penalty** |

The ML penalty is evaluated **once per routing decision, for every link**
(`AdaptiveRouter.compute_penalties`), so predictions are consistent within a
single decision and do not thrash mid-computation.

### 3.4 Why show alternatives?

`RouteResult.alternatives` lists the four cheapest loop-free paths (NetworkX
`shortest_simple_paths` ordered by the same dynamic weight). They are not used
to make the decision; they exist so a human can see *what the router rejected
and why* — a requirement for any explainable routing system.

---

## 4. Packet simulation

`PacketSimulator` is a deterministic (seeded), per-hop model. It is **not** a
TCP/IP stack; it reproduces only the three effects that matter to a routing
protocol.

**1. Transmission (serialization) delay** per hop

```
d_tx = (packet_size_kb × 8) / bandwidth_mbps   [ms]
```

**2. Queueing delay** — the classic non-linear buffer curve

```
d_q = k · c / (1 − c),      c = congestion,   k = 12 ms
```

Delay explodes as the link approaches saturation, exactly like a real router
buffer. `c` is clamped to `0.97` to keep the value finite.

**3. Packet loss** per hop

```
p_drop = 1 − (1 − packet_loss) × reliability
```

i.e. protocol loss (buffer overflow / bit errors) combined with the probability
that the link hardware is down.

A dropped packet is retransmitted up to `max_retries = 2` times with a
`retransmission_timeout_ms = 40` wait — the stop-and-wait behaviour used to
teach reliability. If every attempt fails, the packet is marked `Dropped`.

Each hop additionally gets a seeded `±10 %` jitter so the histogram of
end-to-end delays looks like measurements rather than a set of spikes.

Defaults: **80 packets** per step, 8 KB payload, 5 ms inter-arrival, seed
`42` → results are bit-for-bit reproducible.

---

## 5. Machine-learning module

### 5.1 Dataset

Real router telemetry is not publicly available, so the dataset is generated
by `ml/dataset.py` and this is stated openly — nothing is downloaded or
copied.

* **100 simulated link streams** × **50 consecutive observations** = 5 000 rows.
* Each stream gets a capacity chosen from {100…1000 Mbps}, a baseline delay
  and a target utilisation; utilisation performs a random walk around the
  target, so congestion is **temporally correlated** (a real link does not
  jump from idle to full between samples).
* Latency, loss and queue length are *derived* from utilisation through simple
  queueing relationships plus Gaussian measurement noise.
* `previous_level` is the label of the previous observation of the **same**
  link — a router always knows the previous state, so this is a legitimate
  feature, not future leakage.

Class distribution: `{Normal: 2280, Warning: 1360, Congested: 1360}`.

### 5.2 Ground-truth labelling rule

The label is a documented, deterministic function of the observable state —
so the model is learning a rule a network engineer could write down:

```
severity = 0.40·utilisation + 0.30·min(latency/100, 1) + 0.30·min(loss/0.15, 1)

severity >= 0.65  ->  2 Congested
severity >= 0.40  ->  1 Warning
otherwise         ->  0 Normal
```

Utilisation carries the largest weight because a nearly-full link is
congested even when its fixed delay is still small.

### 5.3 Features (6)

| Feature | Why it is included |
|---|---|
| `latency_ms` | end-to-end delay is the user-visible symptom |
| `packet_loss` | buffer overflow / bit errors |
| `bandwidth_utilization` | `traffic / bandwidth` — the direct measure of load |
| `traffic_mbps` | absolute offered load |
| `queue_length` | `congestion² × 60` — buffer depth grows non-linearly |
| `previous_level` | congestion is persistent; the last state is predictive |

`ml/model.py` owns the feature list and the labelling rule so **training and
inference can never disagree** on the input format.

### 5.4 Model choice

**Random Forest** (`n_estimators=200`, `max_depth=12`, `min_samples_leaf=2`,
`random_state=42`):

* trains in well under a second on 5 000 rows — no GPU needed;
* handles mixed scales without feature normalisation;
* reports **feature importances**, which the report needs to justify the
  feature set;
* robust to the injected measurement noise;
* explainable to a viva panel, unlike a deep network.

Split: **75 / 25**, seeded (42).
Acceptance gate: training aborts below `ML_MIN_ACCEPTABLE_ACCURACY = 0.80`.

### 5.5 Inference → routing penalty

`ml/predict.CongestionPredictor.link_penalty(u, v, attrs)` has exactly the
signature of `core.adaptive_routing.PenaltyFunction`, so the router only ever
sees a callback:

| predicted class | state | additive cost |
|---|---|---|
| 0 | Normal | `0.00` |
| 1 | Warning | `0.35` |
| 2 | Congested | `0.90` |

The predictor remembers `previous_level` per link between decisions, which is
what makes the prediction *temporal* rather than per-sample.

If the model file is missing, `active` is `False`, `link_penalty` returns `0.0`
and the system degrades gracefully to plain Adaptive Routing.

---

## 6. Metrics

All KPIs are computed from packets that were really simulated — none is
hard-coded.

| Metric | Definition |
|---|---|
| avg latency | mean end-to-end delay of *delivered* packets (ms) |
| packet loss % | packets not delivered after all retransmissions / sent |
| delivery rate % | delivered / sent |
| throughput | delivered payload per second (Mbit/s) |
| route changes | how often the selected route changed between steps |
| route utilisation | mean `traffic/bandwidth` over the links the route uses |
| network utilisation | same mean over **all** links |
| congestion frequency | % of steps where the selected route crossed a congested link |
| avg route cost | mean dynamic cost of the selected routes |
| avg hops | mean hop count of the selected routes |

---

## 7. Reproducibility

| Knob | Value | Where |
|---|---|---|
| experiment seed | 42 | `config.EXPERIMENT_SEED` |
| simulation steps per run | 5 | `config.EXPERIMENT_STEPS` |
| packet seed | 42 | same seed passed to `PacketSimulator` |
| dataset seed | 42 | `config.ML_RANDOM_STATE` |
| train/test split seed | 42 | `ml/train.py` |

Re-running `python -m pytest` and the experiment suite reproduces every table
in `docs/experiments.md` exactly.
