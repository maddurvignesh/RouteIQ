# Presentation outline

Slide-by-slide content for the project presentation. Each slide lists the
title, the bullets to put on the slide, the figure to show, and speaker notes.
Aim for **12–15 minutes** plus questions.

> **The deck already exists and is generated from this content:**
> `docs/Adaptive_Routing_Presentation.pptx` (16 slides, 16:9, with speaker
> notes in the notes pane). Rebuild it with `python docs/build_presentation.py`
> after editing `STUDENT` / `GUIDE` at the top of that script.

---

## Slide 1 — Title

**Adaptive Routing Based on Real-Time Network Conditions**

* Subtitle: *A simulation study of reactive and ML-assisted routing*
* Name, roll number, branch, guide, date

> Speaker notes: One-line hook — "Classical routing asks *what is shortest?*;
> this project asks *what is fastest right now?*"

---

## Slide 2 — The problem

* Routing protocols (OSPF/RIP) pick a path from a **static** metric: hop count
* The path is recomputed only when a link goes **up or down**
* Networks rarely fail — they **degrade**: buffers fill, delay rises, packets drop
* Result: all traffic stays pinned to "the shortest path" while it saturates

> Show: two-panel cartoon — "hop count says 3 hops (best)" vs "reality: 800 ms
> queue" .
>
> Speaker notes: This is the standard congestion-collapse story. The metric and
> the objective have drifted apart.

---

## Slide 3 — Goal and objectives

* Build a routing decision that reacts to **live link conditions**
* Measure it against a load-oblivious baseline with a real packet simulation
* Add an **ML congestion predictor** and test whether it adds value
* Report the results honestly, including where ML does **not** help

---

## Slide 4 — System architecture

* Layers: presentation → orchestration → routing / network model → ML → config
* `SimulationEngine` facade owns all state
* ML connects through a single **callback**: `penalty(u, v, attrs) -> float`
  → the router never imports the ML code

> Show: `docs/architecture.md` layer diagram (or `docs/architecture.png`).
>
> Speaker notes: The callback boundary is the key design decision — it is why
> the system still runs if the model file is deleted.

---

## Slide 5 — The network

* **8 routers, 13 links** (S, A, B, C, D, E, F, T)
* Hop-count shortest path: `S → A → C → T` (3 hops, backbone)
* Main alternate: `S → B → D → F → T` (4 hops)
* Every link: bandwidth, latency, loss, congestion, traffic, reliability
* Live values + `base_*` copies → faults can always be undone exactly
* Invariant: `congestion = traffic / bandwidth`

> Show: topology figure from the dashboard's *Network Topology* page.

---

## Slide 6 — Simulating real-time conditions

| Condition | Effect |
|---|---|
| `CONGESTION` | queue fills; congestion, traffic and latency rise |
| `TRAFFIC_SPIKE` | utilisation jumps on one link |
| `HIGH_LATENCY` | delay ×6, loss untouched |
| `HIGH_PACKET_LOSS` | loss driven to 5–20 % |
| `LOW_BANDWIDTH` | capacity cut up to 70 % |

* Severity parameter scales the effect; congestion clamped at **0.97**
* Default target: **the current shortest path** — "congest the route a naive
  protocol would keep using"

---

## Slide 7 — Strategy 1: baseline

* Dijkstra on **hop count** — exactly what OSPF/RIP do
* Never looks at latency, loss or congestion
* This is the control group

---

## Slide 8 — Strategy 2: adaptive cost

```
cost = 0.35·min(latency/100, 1)
     + 0.30·congestion
     + 0.25·min(loss/0.20, 1)
     + 0.10·(1 − min(bandwidth/1000, 1))
```

* Every term normalised to `[0, 1]` → the four are comparable
* Weights validated to sum to 1.0 (in `config.py`)
* Dijkstra with this cost as the edge weight → **minimum cost under current
  conditions**
* Also lists the four cheapest rejected alternatives → explainability

> Show: the *Adaptive Routing* page's link-by-link cost breakdown table.

---

## Slide 9 — Strategy 3: ML-assisted

* 6 features: latency, loss, utilisation, traffic, queue length,
  `previous_level`
* **Random Forest**, 5 000 samples, 75/25 split, seed 42
* Labels from an explicit rule:
  `severity = 0.40·util + 0.30·min(lat/100,1) + 0.30·min(loss/0.15,1)`
* Predicted class → additive penalty **0.00 / 0.35 / 0.90**
* Computed once per decision, for every link

---

## Slide 10 — ML results (before routing)

* Test accuracy **98.48 %** on 1 250 held-out samples
* Macro F1 **0.983** (Normal 0.992, Warning 0.972, Congested 0.985)
* Feature importance: `previous_level` 26 %, `packet_loss` 25 %,
  `bandwidth_utilization` 22 %, `latency` 13 %, `queue_length` 11 %

> Show: confusion matrix + feature-importance bar chart (ML Prediction page).
>
> Speaker notes: Accuracy measures consistency with our own labelling rule.
> The real evidence that the model is useful is Slide 13.

---

## Slide 11 — Experimental design

* **6 scenarios × 3 strategies = 18 runs**, 5 steps each
* **80 packets/step** → 400 per run, **7 200 packets total**
* Same topology, same conditions, same seed (42) for every strategy
* Fresh topology per run → runs cannot contaminate each other
* Metrics: latency, loss, delivery rate, throughput, route changes,
  congestion frequency, route/network utilisation

---

## Slide 12 — Headline results

| Strategy | Avg latency (ms) | Loss (%) | Delivery (%) | Steps on congested path (%) |
|---|---:|---:|---:|---:|
| Shortest Path | 354.7 | 5.83 | 94.2 | 40.0 |
| Adaptive Routing | 151.0 | 0.29 | 99.7 | 26.7 |
| Adaptive + ML | **125.6** | **0.29** | **99.7** | **13.3** |

* Adaptive vs baseline: **−57 %** latency, **−95 %** loss, **+43 %** throughput
* ML vs adaptive: a further **−17 %** latency, congestion exposure **halved**

> Show: grouped bar chart from the *Performance Comparison* page.

---

## Slide 13 — Where the win comes from

**exp2 — congestion on the shortest path**

| Strategy | Route | Latency |
|---|---|---:|
| Shortest Path | `S→A→C→T` (stuck) | **803 ms** |
| Adaptive (+ML) | `S→B→D→F→T` | **86 ms** |

**exp5 — single-link traffic spike (the ML experiment)**

| Strategy | Route | Latency |
|---|---|---:|
| Shortest Path | `S→A→C→T` | 238 ms |
| Adaptive | `S→A→C→T` (does not move) | 238 ms |
| Adaptive + ML | `S→B→D→F→T` | **86 ms** |

> Speaker notes: In exp5 the ML route's *current* cost is slightly higher
> (0.608 vs 0.512). The model is buying a **future** improvement — that is the
> difference between a reactive and a proactive signal.

---

## Slide 14 — When it does not help

* **Healthy network (exp1):** all three strategies pick the identical route —
  adaptation costs nothing, but gains nothing either
* **Whole neighbourhood degraded (exp6):** everyone falls to ~369 ms; adaptive
  routing finds the least-bad path but cannot create capacity
* **ML changes the outcome in 1 of 6 scenarios**, not all of them — stated
  openly

---

## Slide 15 — Limitations

1. Simulation, not a testbed (NS-3 / Mininet / real routers)
2. Synthetic dataset → accuracy reflects our own labelling rule
3. Small fixed topology (8 routers); random-topology builder not yet swept
4. Only three strategies; no ECMP / QoS / SDN baseline
5. Hand-chosen weights, no sensitivity analysis yet

---

## Slide 16 — Conclusion & future work

**Conclusion:** reacting to real-time conditions halves latency and removes
95 % of packet loss; a small explainable classifier adds a proactive step that
reactive routing cannot take.

**Next:** learned/tuned weights · real traffic traces · 20–100 router
topologies · ECMP and QoS baselines · closed-loop feedback · deployment as an
OSPF metric or SDN controller app.

---

## Demo script (5 minutes, optional)

1. *Overview* — show the healthy network and the metrics (route `S→A→C→T`).
2. *Network Topology* — point at link attributes; show `congestion = traffic/bandwidth`.
3. *Adaptive Routing* — show the cost formula and the link-by-link breakdown.
4. *Network Simulation* — press **Increase congestion**; watch the route flip to
   `S→B→D→F→T` and the packet latency histogram shift left.
5. *ML Prediction* — show the confusion matrix and the per-link predictions.
6. *Performance Comparison* — run the full 18-run matrix and show the table.
