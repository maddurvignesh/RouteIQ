# Viva voce — anticipated questions

Questions an examiner is likely to ask, grouped by area, with short answers
grounded in this repository. Numbers quoted here match `docs/experiments.md`
and `models/training_report.json`.

---

## A. Classical routing (baseline knowledge)

**Q1. What is the difference between distance-vector and link-state routing?**
Distance vector (RIP) routers advertise their distance to every destination to
their neighbours and build the table hop by hop; convergence is slow and
count-to-infinity can occur. Link state (OSPF) floods the raw state of every
link to all routers and each runs Dijkstra on the same map — faster convergence
but more memory and flooding traffic. Our baseline reproduces the *outcome* of
link state: a minimum-hop path computed from topology only.

**Q2. Why is hop count a poor metric?**
Because hop count says nothing about delay, capacity or load. A 3-hop path over
three 100 ms, 100 Mbps links is worse than a 4-hop path over four 10 ms, 1 Gbps
links. In our `exp4` experiment the naive route took 165 ms while the adaptive
router took 86 ms on a *longer* path.

**Q3. What causes a router to recompute a path in OSPF?**
A change in the advertised link state — link up/down, cost change, or a
timely LSA refresh. Ordinary congestion does **not** trigger recomputation
unless the operator manually changes the cost, which is exactly the gap this
project fills.

**Q4. What is ECMP and how does it relate to this project?**
Equal-Cost Multi-Path spreads traffic across several next hops that have the
same cost. It is a *load-sharing* technique; this project is a *path-selection*
technique. The two are complementary: once our cost function has ranked the
paths, ECMP could be used to spread traffic across the top-ranked ones.

**Q5. What is congestion collapse?**
When the network carries mostly packets that will be dropped anyway, so
bandwidth is wasted on retransmissions and discarded data. Our baseline
exhibits a mild version of it: in `exp3` it delivered only 76.5 % of packets
while paying 23.5 % loss.

**Q6. Explain queueing delay.**
A packet waits in the router's output buffer while the previous packet is
transmitted. As utilisation approaches 100 % the wait grows non-linearly. We
model it with `d = k · c / (1 − c)`, `k = 12 ms`, `c` the link congestion —
delay explodes as `c → 1`.

**Q7. What is the difference between latency, jitter and throughput?**
Latency is the end-to-end delay of one packet; jitter is its variation over
time; throughput is the useful data delivered per second. We report latency
(mean of delivered packets), and jitter is implicitly present because every hop
adds seeded ±10 % variation.

---

## B. The adaptive cost function

**Q8. Write the adaptive cost formula.**
`cost = 0.35·min(lat/100,1) + 0.30·congestion + 0.25·min(loss/0.20,1)
+ 0.10·(1 − min(bw/1000,1))`, plus an optional ML penalty.

**Q9. Why normalise the terms?**
Because latency is in milliseconds, loss is a fraction and bandwidth is in
Mbps — adding raw values would be meaningless. Each term is mapped to `[0, 1]`
so the weights are true percentages of importance.

**Q10. Why those particular weights?**
They are a priori engineering judgement: delay and loss are what users notice,
congestion is the leading indicator of both, and capacity is a tie-breaker.
They are centralised in `config.CostWeights` and validated to sum to 1.0, so a
sensitivity sweep is a one-line experiment — listed as future work.

**Q11. Is the cost additive across a path?**
Yes — path cost is the sum of link costs, which makes Dijkstra applicable.
Additivity is an approximation (real end-to-end delay also depends on
cross-traffic), but it is the standard assumption in link-state routing.

**Q12. How does the router avoid loops?**
Dijkstra on a connected, undirected graph always returns a simple (loop-free)
path. The alternatives are generated with `shortest_simple_paths`, which is
explicitly loop-free.

**Q13. What is the role of `RouteResult.reason`?**
Explainability. Every decision carries a human-readable justification plus the
four rejected alternatives and their costs, so a user can see *why* a path was
chosen rather than having to trust a black box.

---

## C. Network simulation and conditions

**Q14. Is this a real network?**
No. It is a deterministic, discrete-step simulation running in one Python
process. No packet leaves the machine. The report states this explicitly
throughout.

**Q15. What does a "step" represent?**
One routing interval: apply the scheduled condition → recompute the route →
transmit 80 packets → record metrics. Experiments use 5 steps.

**Q16. What conditions can be simulated and how are they scaled?**
Six degradations (`CONGESTION`, `MULTI_CONGESTION`, `TRAFFIC_SPIKE`,
`HIGH_LATENCY`, `HIGH_PACKET_LOSS`, `LOW_BANDWIDTH`) plus `NORMAL`. A severity
`s ∈ (0,1]` scales each effect; congestion is clamped at 0.97 to keep the
queueing curve finite.

**Q17. How do you know the simulator is not rigged in favour of the adaptive
strategy?**
Three safeguards: (1) the condition targets the *hop-count shortest path*, i.e.
the route every strategy would start from; (2) all strategies receive the
identical timeline on an identical fresh topology; (3) `exp1` is a control —
with no faults, all three strategies pick the same route and produce identical
metrics.

**Q18. How is the packet simulator different from a real TCP stack?**
It models only what matters to path selection: per-hop transmission delay,
queueing delay, a drop probability and stop-and-wait retransmission (2 retries,
40 ms timeout). It does not model congestion windows, slow start, ACKs or
header overhead. The docstring says so plainly.

**Q19. Why is throughput identical-ish across strategies while network
utilisation is exactly identical?**
Because the same amount of traffic is offered in every run and all links stay
up — only the *route* changes, so the network-wide mean `traffic/bandwidth`
(31.32 %) is identical. Route utilisation (43.3 % → 23.2 %) is the meaningful
number: the adaptive strategies move load off the hot links.

---

## D. Machine learning

**Q20. Which model did you use and why a Random Forest?**
Random Forest: trains in under a second on 5 000 rows, needs no feature
scaling, tolerates label noise, reports feature importances that justify the
feature set, and is explainable to an examiner — unlike a deep network. Hyper
parameters: 200 trees, depth 12, `min_samples_leaf = 2`, seed 42.

**Q21. Where did the data come from?**
We generated it: 100 simulated link streams × 50 steps = 5 000 rows. Real
router telemetry is not publicly available. This is stated openly in the
report — no data was downloaded or copied.

**Q22. Isn't a synthetic dataset circular?**
Partly. The labels come from an explicit formula, so 98.5 % accuracy measures
consistency with that formula, not real-world accuracy. We address this in two
ways: the labelling rule is written down and auditable, and the module's value
is demonstrated *end-to-end* by `exp5`, where the prediction changes an actual
routing decision and cuts latency by 64 %. Using public traces is listed as
future work.

**Q23. What are the six features and why each one?**
`latency_ms` (the user-visible symptom), `packet_loss` (buffer overflow /
errors), `bandwidth_utilization` (the direct load measure), `traffic_mbps`
(absolute load), `queue_length` = `congestion² × 60` (buffer depth grows
non-linearly), `previous_level` (congestion persists — the last state is
predictive).

**Q24. Is `previous_level` data leakage?**
No. It is the label of the *previous* observation of the same link — a router
always knows the state it computed last interval. No future information is
used. It is the feature that makes the prediction temporal rather than
per-sample.

**Q25. What does the classifier actually predict?**
One of three classes — Normal / Warning / Congested — for a link, from its
current telemetry. That class is mapped to an additive routing penalty of
0.00 / 0.35 / 0.90.

**Q26. How is the penalty chosen so it does not swamp the cost function?**
The four normalised terms give a link a cost in `[0, 1]`; the penalties are on
the same scale. 0.35 is enough to tip a borderline decision, 0.90 is comparable
to a completely saturated, 100 ms, 20 %-loss link. So the ML signal is
influential but bounded.

**Q27. Why compute penalties for *all* links once per decision instead of
lazily?**
Consistency. If the predictor updated mid-computation the weight function
could return different values for the same link, producing an unstable and
unexplainable decision. Penalties are frozen at the start of each routing
decision.

**Q28. What happens if the model file is deleted?**
`CongestionPredictor.active` becomes `False`, `link_penalty` returns 0.0, and
the system silently degrades to plain Adaptive Routing. Routing never depends
on the ML package being importable.

**Q29. How do you prevent train/test leakage?**
The split is done once, seeded (42), on whole rows; the dataset is generated
deterministically with the same seed. `previous_level` never contains a future
label.

**Q30. What do the feature importances tell you?**
`previous_level` 26 %, `packet_loss` 25 %, `bandwidth_utilization` 22 %,
`latency_ms` 13 %, `queue_length` 11 %, `traffic_mbps` 2 %. Congestion is
persistent and shows up first, then the two direct symptoms — which is exactly
the intuition a network engineer would have.

---

## E. Results

**Q31. State the headline result.**
Against hop-count routing: −57.4 % average latency (354.7 → 151.0 ms),
−95.0 % packet loss (5.83 % → 0.29 %), delivery 94.2 % → 99.7 %,
throughput +43 %. Adding ML: latency −16.9 % further (→ 125.6 ms) and time on a
congested path 26.7 % → 13.3 %.

**Q32. Which single experiment best demonstrates the project?**
`exp2`. The naive route is pinned to a congested backbone at 803 ms with 5.75 %
loss; the adaptive router switches at step 1 to `S→B→D→F→T` and reaches 86 ms
with 0 % loss — a 9.4× improvement from one design change.

**Q33. Which experiment demonstrates the ML module?**
`exp5`, and only `exp5`. A traffic spike hits a single link (`S-A`); the
weighted formula alone leaves the router there at 238 ms, while the ML penalty
triggers a reroute at 86 ms.

**Q34. So the ML only helps in one scenario out of six — is it worth it?**
Yes, and the limitation is reported rather than hidden. The ML module is a
*proactive* signal: it fires precisely when the reactive heuristic is
undecided. That is the intended division of labour in a hybrid design, and the
one firing case is worth 64 % latency in that scenario and 17 % on the
six-scenario average.

**Q35. In `exp5` the ML route has a *higher* current cost (0.608 vs 0.512).
Explain.**
The model is trading a small present cost for a large expected future gain —
the link it is leaving is predicted to worsen. This is the definition of
proactive routing, and it is why the two strategies differ only when a
condition is genuinely borderline.

**Q36. Why is Shortest Path's latency lower than Adaptive's in `exp3` (76 ms vs
86 ms) even though it loses 23.5 % of packets?**
Because average latency is computed only over *delivered* packets: the naive
route's slowest packets are the ones that got dropped, so they never enter the
average. This is survivorship bias. Loss (23.5 % vs 0 %) and delivery rate
(76.5 % vs 100 %) are the honest comparison, and we say so.

**Q37. What happens when the whole neighbourhood is congested (`exp6`)?**
All strategies degrade — baseline 803 ms, both adaptive strategies 369 ms with
1.75 % residual loss. Adaptive routing finds the least-bad path; it cannot
create capacity. This is the honest boundary of the technique.

**Q38. Are your results statistically significant?**
They are deterministic: one seed (42), one fixed topology, 5 steps. That makes
them exactly reproducible but not a statistical sample. Repeating over many
seeds and random topologies is listed as future work — a fair criticism we
accept.

---

## F. Software engineering

**Q39. How is the code organised?**
Five layers — `core` (network model + routing), `ml`, `simulation`
(orchestration), `visualization` (UI), `config` — with enforced dependency
directions: `core` never imports `ml`, `visualization` never imports `core`
directly (it goes through `SimulationEngine`).

**Q40. How do you know the system works?**
162 automated tests: unit tests for topology, conditions, routing, packet
simulation, metrics, scenarios and the ML pipeline, plus headless integration
tests that execute the real `app.py` with Streamlit's `AppTest` and assert that
all six pages render and react to faults.

**Q41. Where are the "magic numbers"?**
Nowhere. Every constant — weights, normalisation bounds, penalties, seeds,
thresholds, packet settings — lives in `config.py`, in dataclasses with
validation (for example the weights must sum to 1.0 or the constructor raises).

**Q42. How reproducible is the experiment?**
Fully. Dataset, model, packet simulation and experiments all use seed 42; a
fresh topology is built for every run. Re-running the commands in the appendix
regenerates every table in the report byte for byte.

**Q43. Why a Streamlit dashboard rather than just scripts?**
Because routing is a *decision over time* and the argument is much more
convincing when you can watch the route flip when a condition is injected. The
dashboard is thin — all logic lives in importable modules, which is why the UI
itself can be tested headlessly.

**Q44. What would you change with more time?**
Learned or swept weights; real traffic traces; 20–100 node topologies; ECMP and
QoS baselines; a closed loop where simulated packet outcomes feed back into the
link state; and deployment of the cost function as an OSPF metric or an SDN
controller application.
