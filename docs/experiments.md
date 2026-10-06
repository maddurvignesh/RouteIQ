# Experiments and results

Everything in this document was produced by running the code in this
repository. Reproduce it with:

```bash
python -m ml.dataset      # regenerate data/congestion_dataset.csv
python -m ml.train        # retrain models/congestion_model.joblib
python -m pytest          # 162 tests
```

then pressing **Run all experiments** on the dashboard's *Performance
Comparison* page (or calling `SimulationEngine().run_experiments()`).

---

## 1. Design

| Dimension | Value |
|---|---|
| Scenarios | 6 |
| Routing strategies | 3 |
| Runs | **18** |
| Time steps per run | 5 |
| Packets per step | 80 (→ 400 packets per run, 7 200 in total) |
| Seed | 42 (conditions, packets, ML split) |
| Topology | predefined 8-router / 13-link graph |
| Source → destination | S → T |

Each run builds a **fresh** topology and condition simulator, so no run can
contaminate another. Every strategy sees exactly the same timeline.

### Strategies under test

| # | Strategy | Decision metric |
|---|---|---|
| 1 | **Shortest Path** | hop count (static) — the load-oblivious control |
| 2 | **Adaptive Routing** | `0.35·L + 0.30·C + 0.25·P + 0.10·B` |
| 3 | **Adaptive + ML** | the same four terms **+ predicted-state penalty** (0 / 0.35 / 0.90) |

### Scenarios

| ID | Condition | Targeted links | What it tests |
|---|---|---|---|
| `exp1_normal` | none | — | sanity check: all strategies agree on a healthy network |
| `exp2_congestion` | `CONGESTION` | whole shortest path | classic bufferbloat on the "best" route |
| `exp3_packet_loss` | `HIGH_PACKET_LOSS` | whole shortest path | reliability vs. delay trade-off |
| `exp4_latency` | `HIGH_LATENCY` | whole shortest path | hop count is a poor proxy for delay |
| `exp5_traffic_spike` | `TRAFFIC_SPIKE` | **only S-A** | early warning: does the ML hook move earlier? |
| `exp6_multi_congestion` | `MULTI_CONGESTION` | backbone **and** its main alternative | behaviour when no good path remains |

For scenarios 1–4 the runner automatically targets the *current hop-count
shortest path* — "congest the route a naive protocol would keep using".

---

## 2. Results

### 2.1 Per-scenario results (18 runs)

| Scenario | Strategy | Avg latency (ms) | Loss (%) | Delivery (%) | Route changes | Avg route cost |
|---|---|---:|---:|---:|---:|---:|
| exp1_normal | Shortest Path | 41.88 | 0.00 | 100.00 | 0 | 0.304 |
| exp1_normal | Adaptive Routing | 41.88 | 0.00 | 100.00 | 0 | 0.304 |
| exp1_normal | Adaptive + ML | 41.88 | 0.00 | 100.00 | 0 | 0.304 |
| exp2_congestion | Shortest Path | 803.25 | 5.75 | 94.25 | 0 | 1.462 |
| exp2_congestion | Adaptive Routing | **85.65** | **0.00** | **100.00** | 1 | 0.608 |
| exp2_congestion | Adaptive + ML | **85.65** | **0.00** | **100.00** | 1 | 0.608 |
| exp3_packet_loss | Shortest Path | 76.25 | 23.50 | 76.50 | 0 | 0.889 |
| exp3_packet_loss | Adaptive Routing | **85.65** | **0.00** | **100.00** | 1 | 0.608 |
| exp3_packet_loss | Adaptive + ML | **85.65** | **0.00** | **100.00** | 1 | 0.608 |
| exp4_latency | Shortest Path | 165.37 | 0.00 | 100.00 | 0 | 0.724 |
| exp4_latency | Adaptive Routing | **85.65** | 0.00 | 100.00 | 1 | 0.608 |
| exp4_latency | Adaptive + ML | **85.65** | 0.00 | 100.00 | 1 | 0.608 |
| exp5_traffic_spike | Shortest Path | 238.43 | 0.00 | 100.00 | 0 | 0.512 |
| exp5_traffic_spike | Adaptive Routing | 238.43 | 0.00 | 100.00 | 0 | **0.512** |
| exp5_traffic_spike | Adaptive + ML | **85.65** | 0.00 | 100.00 | 1 | 0.608 |
| exp6_multi_congestion | Shortest Path | 803.25 | 5.75 | 94.25 | 0 | 1.462 |
| exp6_multi_congestion | Adaptive Routing | 369.02 | 1.75 | 98.25 | 1 | 1.148 |
| exp6_multi_congestion | Adaptive + ML | 369.02 | 1.75 | 98.25 | 1 | 1.148 |

`avg_route_cost` is always the **dynamic weighted cost of the selected route**
(the four terms, without the ML penalty) so the column is comparable across
strategies.

### 2.2 Summary averaged over the six scenarios

| Strategy | Avg latency (ms) | Loss (%) | Throughput (Mbps) | Delivery (%) | Route changes | Congestion frequency (%) | Avg route cost | Route util. (%) | Avg hops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Shortest Path | 354.74 | 5.83 | 25.32 | 94.17 | 0.00 | 40.00 | 0.892 | 43.33 | 3.00 |
| Adaptive Routing | 151.05 | 0.29 | 36.23 | 99.71 | 0.67 | 26.67 | 0.631 | 26.50 | 3.53 |
| Adaptive + ML | **125.58** | **0.29** | **40.07** | **99.71** | 0.83 | **13.33** | 0.647 | 23.17 | 3.67 |

Network utilisation is identical for all three (31.32 %) — the *same traffic*
is offered in every run; only the **route** chosen differs. That is why route
utilisation (43.33 % → 23.17 %) is the meaningful figure: the adaptive
strategies push load away from the hot links.

### 2.3 Headline improvements

| Comparison | Latency | Packet loss | Throughput |
|---|---|---|---|
| Adaptive Routing vs Shortest Path | **−57.4 %** (354.7 → 151.0 ms) | **−95.0 %** (5.83 % → 0.29 %) | **+43.1 %** (25.3 → 36.2 Mbps) |
| Adaptive + ML vs Shortest Path | **−64.6 %** (354.7 → 125.6 ms) | **−95.0 %** (5.83 % → 0.29 %) | **+58.3 %** (25.3 → 40.1 Mbps) |
| Adaptive + ML vs Adaptive Routing | **−16.9 %** (151.0 → 125.6 ms) | equal | +10.6 % (36.2 → 40.1 Mbps) |

Delivery rate rises from **94.17 %** (hop count) to **99.71 %** (both adaptive
strategies), and the share of steps in which the selected route crossed a
congested link falls from **40 %** to **13.3 %** with ML.

---

## 3. Analysis, experiment by experiment

### exp1 — Normal conditions (control)

All three strategies pick `S → A → C → T`, latency 41.9 ms, loss 0 %,
100 % delivery, dynamic cost 0.304.

**Conclusion:** when nothing is wrong the adaptive cost function *agrees* with
hop count. There is no penalty for adapting on a healthy network — an important
fairness check.

### exp2 — Congestion on the shortest path

The naive route's three links degrade: queueing delay follows `k·c/(1−c)` and
blows up to **803 ms** with 5.75 % loss. Adaptive Routing detects the cost
change at step 1 and moves to `S → B → D → F → T`: **85.6 ms, 0 % loss**.

**Conclusion:** a 9.4× latency reduction from reacting to live conditions
instead of hop count.

### exp3 — High packet loss

Shortest Path keeps the route (it cannot see loss) → **23.5 % packet loss,
76.5 % delivery**, with end-to-end delay dragged up by retransmissions.
Adaptive Routing leaves the route and reaches **100 % delivery**.

Note the teaching point: the naive route has a *slightly lower* raw latency
(76.3 ms vs 85.6 ms) because the metric only averages **delivered** packets —
a classic survivorship artefact. The loss and delivery columns are the honest
comparison here.

### exp4 — High latency

Backbone latency × 6: Shortest Path 165.4 ms, Adaptive 85.6 ms by taking the
4-hop but faster path. **Hop count is a poor proxy for real delay.**

### exp5 — Sudden traffic spike on a single link (the ML experiment)

Only `S-A` is hit. The weighted formula alone does **not** consider the route
bad enough to abandon: plain Adaptive Routing stays put at **238.4 ms**.
The ML hook predicts a *Warning* state on `S-A`, adds `+0.35` to its cost, and
the router switches to `S → B → D → F → T` at **85.6 ms**.

This is the only scenario where Adaptive + ML differs from plain Adaptive, and
it is the scenario designed to isolate that difference:

* the ML route's *current* dynamic cost is slightly higher (0.608 vs 0.512) —
  the model is buying a **future** improvement, not a present one;
* latency drops by **64 %** (238.4 → 85.6 ms).

**Conclusion:** the ML module contributes *early warning*, not a better
instantaneous cost model. It is the component that moves first when the
hand-tuned weights are still undecided.

### exp6 — Multiple congested links

Both the backbone and its main alternative degrade, so no good path exists.
All strategies pay: Shortest 803 ms, both adaptive strategies 369 ms with
1.75 % residual loss by choosing `S → A → D → E → T`.

**Conclusion:** adaptive routing degrades *gracefully* — it finds the least-bad
path — but it cannot create capacity that the network does not have. This marks
the honest boundary of the technique.

---

## 4. Threats to validity and limitations

1. **Synthetic dataset.** The congestion classifier is trained on generated
   data whose labels come from an explicit formula, so the reported 98.5 %
   accuracy measures *consistency with that formula*, not real-world accuracy.
   The value of the module is demonstrated end-to-end by **exp5**, where the
   prediction changes an actual routing decision.
2. **Simulation, not a testbed.** No real routers, no NS-3, no packets leave
   the machine. Conclusions are qualitative.
3. **Small topology.** 8 routers / 13 links. The random-topology builder exists
   for stress testing but the reported experiments use one fixed graph.
4. **Three strategies.** No OSPF with ECMP, no QoS routing, no SDN controller
   comparison.
5. **Fixed weights.** `0.35 / 0.30 / 0.25 / 0.10` were chosen a priori; a
   sensitivity sweep would be the natural next step.
6. **ML adds value in one of six scenarios.** That is reported rather than
   hidden: a proactive signal that fires only when the reactive heuristic is
   undecided is exactly what a hybrid design is for.

---

## 5. Files produced

| File | Content |
|---|---|
| `results/experiment_metrics.csv` | 18 rows × 18 columns, one row per run (includes the full route sequence) |
| `results/comparison_summary.csv` | 3 rows, averages per strategy |
| `data/congestion_dataset.csv` | 5 000 labelled observations |
| `models/congestion_model.joblib` | trained Random Forest |
| `models/training_report.json` | accuracy, confusion matrix, feature importances |
