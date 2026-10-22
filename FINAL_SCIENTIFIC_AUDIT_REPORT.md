# Final Scientific Audit & Evaluation Report: ML-Based Predictive TCP Congestion Control

**Project Title:** Machine Learning-Based Predictive TCP Congestion Control Simulation  
**Authors:** BAHAAR (Reg. No: 24BCE1760), PARDHU (Reg. No: 24BCE1754)  
**Evaluation Date:** October 2026  
**Repository:** `https://github.com/pardhuvenkatta/ML-Based-Predictive-TCP-Congestion-Control.git`  

---

## Executive Summary & Honest Conclusion

Through rigorous, multi-scenario evaluation with 20 fixed seeds per scenario, time-ordered train/test partitioning, and strictly sender-observable telemetry (RFC 793/5681 compliant), we investigated whether adding a Machine Learning predictor (Random Forest regressor or classifier) to an ACK-clocked rate pacing controller improves transport layer congestion control. 

**Conclusion:** 
What truly works in this architecture is **ACK-clocked delivery-rate pacing with fast delay-aware expiry**: it stabilizes sending rates at available link capacity, prevents drop-tail queue overflow, and eliminates bufferbloat across clean, high-bandwidth, and moderately congested channels. Conversely, the **machine learning component adds no statistically measurable closed-loop benefit** over non-ML alternatives. In offline evaluations on held-out families, a trivial **Persistence baseline** ($\widehat{\text{ratio}}_{t+5} = \text{ratio}_t$) achieved a lower Mean Absolute Error ($0.0140$) than the Random Forest regressor ($0.0145$). In closed-loop simulations across 7 held-out scenario families, the paired per-seed utility difference between Random Forest and Persistence was $\Delta U_{1.0} = -0.0160 \pm 0.0039$ (95% CI strictly below zero in favor of Persistence), while against simple Linear Extrapolation it showed no meaningful practical distinction. Furthermore, in mixed-protocol competition against aggressive loss-based senders (TCP Reno and TCP CUBIC), ML-TCP suffers from the universal delay-based starvation trap ($JFI \approx 0.51$, goodput collapsing to $< 0.1\text{ pkts/step}$), demonstrating that the learned model does not overcome fundamental transport-layer competition dynamics.

---

## 1. Experimental Setup & Simulation Physics

The simulation models a discrete-event physical network bottleneck traversed by one or more TCP senders:
- **Bottleneck Capacity ($C$):** $3.0\text{ pkts/step}$ default (clean link); dynamic random walks in $[1.0, 9.0]\text{ pkts/step}$.
- **Router Buffer ($B$):** $50\text{ pkts}$ default; extreme shallow buffers at $B = 5\text{ pkts}$ and $B = 3\text{ pkts}$. Drop-tail FIFO queuing discipline.
- **Base Round-Trip Time ($\text{BaseRTT}$):** $50.0\text{ ms}$ propagation delay. Queuing delay adds $1.9\text{ ms}$ per queued packet:
  $$\text{True RTT}_t = \text{BaseRTT} + 1.9 \times \text{Queue}_t$$
- **Telemetry Noise & Spikes:** Gaussian measurement jitter ($\sigma = 1.0\text{ ms}$) plus $1\%$ stochastic latency spikes ($+10\text{ ms}$ to $+35\text{ ms}$).
- **Non-Congestion Packet Drops:** Stochastic Bernoulli packet loss rates of $0.1\%$ and $2.0\%$ representing wireless channel corruption.
- **Staggered Multi-Flow Scaling:** Symmetrical flows ($N=2$ and $N=8$) sharing bottleneck with randomized flow arrival offsets ($t \in [50, 400]$ steps).

### Sender Observability Constraints (Zero Leaks)
The sender operates strictly within the host network stack. It reads **only** local ACK arrival timestamps and sequence acknowledgments:
1. `rtt_ratio`: $\text{RTT}_t / \min_{0 \dots t}(\text{RTT})$
2. `rtt_gradient_1`: $\text{RTT}_t - \text{RTT}_{t-1}$
3. `rtt_gradient_3`: $(\text{RTT}_t - \text{RTT}_{t-3}) / 3$
4. `rtt_gradient_5`: $(\text{RTT}_t - \text{RTT}_{t-5}) / 5$
5. `delivery_rate`: Packets delivered per step estimated via ACK return rate.
6. `send_rate`: Local transmission rate at step $t$.
7. `loss_flag_5`: Local binary indicator of ACK loss in the last 5 steps.

> [!IMPORTANT]
> The sender **never** accesses router internal queue size, buffer capacity, link processing bandwidth, or future simulator events.

---

## 2. Train vs. Held-Out Family Partitioning

To prevent simulator overfitting, we strictly partitioned all scenarios:

| Dataset Family | Purpose | Scenarios Included | Channel Parameters |
| :--- | :--- | :--- | :--- |
| **TRAIN Family** | Model fitting, DAgger retraining, and parameter sweep | `TR1: Clean Baseline`<br>`TR2: Moderate Link` | $C \in [3.0, 5.0]$, $B = 50$, lossless, 1 flow, constant capacity |
| **HELD-OUT Family** | Out-of-distribution evaluation (never seen during training) | `HO1: Variable BW Walk`<br>`HO2: Random Loss 0.1%`<br>`HO3: Random Loss 2.0%`<br>`HO4: Shallow Buffer B=5`<br>`HO5: Ultra-Shallow B=3`<br>`HO6: 2 Flows (Staggered)`<br>`HO7: 8 Flows (Staggered)` | $C \in [1.0, 9.0]$ random walk, $B \in [3, 5]$, random loss $0.1\% - 2.0\%$, dense flow aggregation ($N=2, 8$) |

---

## 3. Phase 1: Baselines & Pacing Mechanism Selection

### A. Multi-Flow BBR Fairness Sanity Check
Evaluating audited BBR (with 10-RTT $BtlBw$ window and $\text{cwnd} = 2.0 \times \text{BDP}$ inflight cap) against delay-based TCP Vegas:
- **2 Flows:** BBR achieved $JFI = 0.668 \pm 0.017$ (10.1 loss) vs. Vegas $JFI = 0.998 \pm 0.001$ (0.0 loss).
- **8 Flows:** BBR achieved $JFI = 0.212 \pm 0.004$ (153.7 loss) vs. Vegas $JFI = 0.993 \pm 0.002$ (0.0 loss).

**Protocol Audit Finding:** The low JFI of BBR ($0.212$) in 8-flow runs is **not a simulator artifact**. It is a documented real-world flaw in canonical BBRv1 (Cardwell et al. 2016; Hock et al. ACM SIGCOMM 2017). When flows arrive with staggered start times, earlier flows that established higher bandwidth estimates dominate the bottleneck buffer. Because BBR only increases rate upon measuring a higher delivery rate, later flows measure truncated delivery rates and remain starved. In contrast, delay-based Vegas measures absolute queuing delay differences, forcing all flows into symmetric equilibrium ($JFI = 0.993$).

### B. No-Model Pacing Comparison on TRAIN Scenarios
We evaluated 3 no-model pacing variants across 20 fixed seeds ($[42, 61]$):
1. **EWMA Pacing:** $\text{est} = 0.85 \times \text{est} + 0.15 \times \text{acked}$.
2. **MaxFilter-20:** Max delivery rate over a 20-step rolling window.
3. **MaxFilter-FastExpiry:** Max delivery rate over a 10-step window, with immediate flush/reset to current delivery rate upon delay inflation ($\text{rtt\_ratio} > 1.15, \text{grad}_1 > 0$) or packet loss.

| Pacing Variant | TRAIN Goodput (pkts/step) | TRAIN Loss | TRAIN Avg RTT (ms) | HELD-OUT S2 Goodput / Loss | HELD-OUT S8 Goodput / Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EWMA** | $3.9324 \pm 0.0021$ | 0.0 | 59.4 | 3.8450 / 0.3 pkts | 5.8769 / 0.7 pkts |
| **MaxFilter-20** | **$3.9476 \pm 0.0014$** | **0.0** | 59.5 | 3.8137 / 17.4 pkts | 5.8860 / 3.3 pkts |
| **MaxFilter-FastExpiry** | **$3.9476 \pm 0.0014$** | **0.0** | 59.5 | **3.8584 / 0.4 pkts** | **5.8934 / 0.5 pkts** |

**Selection:** `MaxFilter-FastExpiry` was selected as the winning No-Model Baseline: it matches the highest goodput on TRAIN ($3.9476$), while its fast expiry prevents holding stale capacity estimates during link collapse, reducing losses in S2 by **97.7%** relative to standard MaxFilter-20.

---

## 4. Phase 2: Closed-Loop Predictor Ablation & Gate 1 Decision

### A. Offline Predictor Quality on Held-Out Test Traces
Evaluating future $t+5$ RTT ratio prediction ($\frac{\text{RTT}_{t+5}}{\min \text{RTT}}$) on time-ordered held-out traces:

| Predictor Method | Test MAE | Test $R^2$ | Offline Winner |
| :--- | :---: | :---: | :--- |
| **Persistence Baseline** ($\hat{y}_{t+5} = y_t$) | **0.0140** | **-0.3070** | **Lowest MAE (Best Offline Predictor)** |
| **Random Forest Regressor** | 0.0145 | -0.4352 | Second Best (Overfitting to TRAIN dynamics) |
| **Linear Extrapolation** ($\hat{y}_{t+5} = y_t + 5 \cdot \frac{g_1}{\text{min\_RTT}}$) | 0.0391 | -20.8696 | Third Best (Amplifies high-frequency noise) |

### B. Closed-Loop Performance on Held-Out Families (20 Seeds Mean ± 95% CI)
Controllers use identical architectures: if $\hat{y}_{t+5} \ge \text{threshold}$ and $\text{gradient}_1 > 0$, cut rate by factor; else pace with `MaxFilter-FastExpiry`.

| Scenario | No Predictor (FastExpiry Baseline) | Persistence ($\text{th}=1.05, 0.9$) | Linear Extrapolation ($\text{th}=1.20, 0.9$) | Random Forest Regressor ($\text{th}=1.05, 0.9$) |
| :--- | :---: | :---: | :---: | :---: |
| **HO1: Variable BW Walk** | $3.8584 \pm 0.6062$ (102.4ms) | $3.6676 \pm 0.5728$ (50.9ms) | $3.5351 \pm 0.5192$ (51.3ms) | $3.5358 \pm 0.5386$ (50.6ms) |
| **HO2: Random Loss 0.1%** | $2.9581 \pm 0.0053$ (59.1ms) | $2.8940 \pm 0.0090$ (50.7ms) | $2.8669 \pm 0.0115$ (50.7ms) | $2.8486 \pm 0.0102$ (50.4ms) |
| **HO3: Random Loss 2.0%** | $2.2509 \pm 0.0720$ (52.7ms) | $2.0290 \pm 0.0696$ (50.2ms) | $1.8895 \pm 0.0760$ (50.2ms) | $1.8976 \pm 0.0710$ (50.1ms) |
| **HO4: Shallow Buf B=5** | $2.6933 \pm 0.0023$ (50.8ms) | **$2.9303 \pm 0.0027$ (50.7ms)** | $2.8862 \pm 0.0060$ (50.6ms) | $2.8959 \pm 0.0035$ (50.4ms) |
| **HO5: Ultra-Shallow B=3** | $2.4661 \pm 0.0019$ (50.2ms) | $2.4565 \pm 0.0032$ (50.2ms) | $2.4380 \pm 0.0056$ (50.1ms) | $2.4405 \pm 0.0051$ (50.1ms) |
| **HO6: Multi-Flow 2 Flows** | $3.9537 \pm 0.0015$ (70.8ms) | $3.9006 \pm 0.0033$ (51.0ms) | $3.8915 \pm 0.0055$ (51.2ms) | $3.8631 \pm 0.0048$ (50.6ms) |
| **HO7: Multi-Flow 8 Flows** | $5.8934 \pm 0.0070$ (104.6ms) | $5.8635 \pm 0.0091$ (52.3ms) | $5.8693 \pm 0.0101$ (53.9ms) | $5.8419 \pm 0.0098$ (51.9ms) |

### C. Paired Statistical Differences in Objective Utility ($U_{1.0} = \log(\text{Goodput}) - 1.0 \times \log(\text{RTT})$)
| Comparison | Mean Difference ($\Delta U_{1.0}$) | 95% Confidence Interval | Statistically Significant? | Winner |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest vs. Linear Extrapolation** | $+0.0089$ | $[+0.0067, +0.0111]$ | Yes (*) | Random Forest (+0.89% gain) |
| **Random Forest vs. Persistence** | **$-0.0160$** | **$[-0.0199, -0.0121]$** | **Yes (*)** | **Persistence Wins (RF is strictly worse)** |

> [!CAUTION]
> ### GATE 1 VERDICT: ML PREDICTOR ADDS NO MEASURABLE BENEFIT
> Across all held-out scenarios, the Random Forest closed-loop objective score does **not** beat simple Persistence. Persistence achieves higher goodput in 7 out of 7 held-out scenarios while maintaining identical queue latency. Per the project protocol, **Phases 3, 4, and 5 are skipped**, proceeding directly to Phase 6.

---

## 5. Phase 6: Fairness and Competition Benchmarks

### A. Symmetrical Multi-Flow Fairness (2 and 8 Flows)
Bottleneck shared symmetrically by homogeneous flows across 20 seeds:

| Protocol Candidate | 2-Flow Goodput | 2-Flow Loss | 2-Flow JFI | 8-Flow Goodput | 8-Flow Loss | 8-Flow JFI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ML-TCP (RF)** | 3.8631 | **0.0 pkts** | **$0.998 \pm 0.001$** | 5.8419 | **0.0 pkts** | **$0.996 \pm 0.001$** |
| **Persistence (Best Non-ML)** | 3.9006 | **0.0 pkts** | **$0.998 \pm 0.001$** | 5.8635 | **0.0 pkts** | **$0.996 \pm 0.001$** |
| **Reference Vegas** | **3.9691** | **0.0 pkts** | **$0.998 \pm 0.001$** | **5.9208** | **0.0 pkts** | **$0.993 \pm 0.002$** |
| **Audited BBR** | 3.9708 | 10.1 pkts | $0.668 \pm 0.017$ | 5.5899 | 153.7 pkts | $0.212 \pm 0.004$ |
| **TCP Reno** | 3.6082 | 137.6 pkts | $0.998 \pm 0.001$ | 5.8612 | 23.4 pkts | $0.994 \pm 0.001$ |
| **TCP CUBIC** | 3.9180 | 25.4 pkts | $0.967 \pm 0.010$ | 5.8253 | 50.4 pkts | $0.960 \pm 0.005$ |

### B. Mixed-Protocol Competition Against Loss-Based Senders
Sharing a $6.0\text{ pkts/step}$ bottleneck with 1 competing flow over 20 seeds:

| Competition Pairing | Candidate Goodput | Opponent Goodput | Aggregate Loss | Jain's Fairness Index | Competition Outcome |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ML-TCP (RF) vs. Reno** | **$0.0882 \pm 0.0011$** | **$5.8778 \pm 0.0006$** | 37.1 pkts | **$0.515 \pm 0.000$** | **Severe Starvation of ML-TCP** |
| **Persistence vs. Reno** | **$0.0895 \pm 0.0010$** | **$5.8778 \pm 0.0006$** | 36.8 pkts | **$0.515 \pm 0.000$** | **Severe Starvation of Persistence** |
| **Reference Vegas vs. Reno** | **$0.4663 \pm 0.0010$** | **$5.5061 \pm 0.0010$** | 33.3 pkts | **$0.584 \pm 0.000$** | **Severe Starvation of Vegas** |
| **Audited BBR vs. Reno** | **$1.8828 \pm 0.0110$** | **$4.1142 \pm 0.0110$** | 73.3 pkts | **$0.878 \pm 0.002$** | **Fair Sharing (BBR resists)** |
| **ML-TCP (RF) vs. CUBIC** | **$0.0464 \pm 0.0004$** | **$5.9330 \pm 0.0002$** | 30.7 pkts | **$0.508 \pm 0.000$** | **Near-Total Starvation of ML-TCP** |
| **Persistence vs. CUBIC** | **$0.0470 \pm 0.0003$** | **$5.9329 \pm 0.0002$** | 31.0 pkts | **$0.508 \pm 0.000$** | **Near-Total Starvation of Persistence** |
| **Reference Vegas vs. CUBIC** | **$0.1509 \pm 0.0003$** | **$5.8315 \pm 0.0003$** | 20.0 pkts | **$0.526 \pm 0.000$** | **Severe Starvation of Vegas** |
| **Audited BBR vs. CUBIC** | **$0.6849 \pm 0.0086$** | **$5.3125 \pm 0.0086$** | 31.5 pkts | **$0.627 \pm 0.002$** | **Substantial Starvation of BBR** |

#### Why Does Starvation Occur?
Starvation is **not specific to ML**; it is an inherent property of **delay-based congestion control** when competing with loss-based algorithms (Mo et al. 1999). Reno and CUBIC continuously inflate their window until the physical buffer is full. Because ML-TCP, Persistence, and Vegas interpret queuing delay as impending bottleneck collapse, they proactively throttle down to their minimum rate floor ($0.02 - 0.20 \cdot \text{delivery\_rate}$). BBR partially resists Reno ($JFI = 0.878$) only because it enforces an inflight cap based on maximum measured delivery rate rather than delay backoff.

---

## 6. Phase 7: Ablations (Features, Horizons & Thresholds)

### A. Leave-One-Feature-Out Ablation (Evaluated on S2 Random Walk)
| Feature Removed | Goodput (pkts/step) | Avg RTT (ms) | $\Delta U_{1.0}$ vs. Full Model | Operational Impact |
| :--- | :---: | :---: | :---: | :--- |
| **None (Full 7 Features)** | 3.5358 | 50.6 | $+0.0000$ (Ref) | Reference Baseline |
| **`rtt_ratio`** | **1.8899** | **54.7** | **$-0.6437 \pm 0.1117$ (*)** | **Critical: Controller Collapses** |
| **`rtt_gradient_1`** | 3.5558 | 50.6 | $+0.0044 \pm 0.0017$ (*) | Negligible / Redundant |
| **`rtt_gradient_3`** | 3.5251 | 50.6 | $-0.0019 \pm 0.0019$ (ns) | Negligible / Redundant |
| **`rtt_gradient_5`** | 3.5318 | 50.6 | $-0.0008 \pm 0.0009$ (ns) | Negligible / Redundant |
| **`delivery_rate`** | 3.5259 | 50.6 | $-0.0016 \pm 0.0019$ (ns) | Negligible / Redundant |
| **`send_rate`** | 3.5259 | 50.6 | $-0.0016 \pm 0.0018$ (ns) | Negligible / Redundant |
| **`loss_flag_5`** | 3.5258 | 50.6 | $-0.0017 \pm 0.0018$ (ns) | Negligible / Redundant |

**Finding:** The feature set is dominated by **`rtt_ratio`**. Removing `rtt_ratio` cuts goodput by **46.5%** and destroys control stability. The other 6 features (gradients, delivery rate, send rate, loss flag) provide statistically negligible utility differences ($|\Delta U_{1.0}| < 0.005$).

### B. Prediction Horizon Sensitivity
| Horizon ($H$) | Goodput (pkts/step) | Loss | Avg RTT (ms) | $U_{1.0}$ Score | Effect |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **$H = 3$** | **2.9174** | **0.0** | 50.4 | **-2.8826** | **Optimal:** Tight reactive pacing without over-throttling. |
| **$H = 5$** | 1.7551 | 0.0 | 50.1 | -3.3549 | Moderate over-throttling on transient noise. |
| **$H = 10$** | 0.9862 | 0.0 | 50.0 | -3.9267 | Severe over-throttling: predicts distant delay spikes and collapses rate. |

---

## 7. Corrections to the Original Results

| Aspect | Initial Pre-Audit Setup | Audited Honest Setup | Scientific Impact of Correction |
| :--- | :--- | :--- | :--- |
| **Router Queue Access** | Cheated: Read `Queue_Size` directly from router state | **Strictly Purged:** Only sender-observable ACK timings (`rtt_ratio`, gradients) | Real TCP senders cannot observe buffer size. Honest version works across arbitrary unknown buffer sizes. |
| **Capacity Hardcoding** | Hardcoded capacity constant `3.02` in rate adjustments | **Removed:** Adaptive ACK-based delivery rate pacing | Initial model only worked at 3 pkts/step. Honest version dynamically tracks links from 1 to 9 pkts/step. |
| **AIMD Baseline** | Artificially weak AIMD that suffered 103 packet losses | **Standard Reno & CUBIC:** Full RFC 5681/8312 baselines | Initial claim of "beating AIMD" was an artifact of misconfigured baseline parameters. |
| **Data Partitioning** | Random row-shuffled train/test split (temporal leakage) | **Time-Ordered Split & Held-Out Scenario Families** | Row shuffling leaked future packet states to the past. Honest split tests true generalization. |
| **Original Claims** | Claimed ML model achieved 0 loss through superior AI | **Isolated via Ablation:** ACK delivery pacing achieved 0 loss, **not** the ML model | Senders with the ML model disabled achieve identical 0 loss and line-rate goodput. |

---

## 8. Where ML-TCP Loses

1. **Under Wireless Non-Congestion Loss (S4: 2.0% Loss):**
   - TCP Reno achieves **$2.4855\text{ pkts/step}$**; BBR achieves **$2.7964\text{ pkts/step}$**.
   - ML-TCP achieves only **$1.8976 - 2.1605\text{ pkts/step}$** because it backs off on packet losses that are not caused by queue overflow.
2. **Under Ultra-Shallow Buffers (S6: Buffer $B=3$ pkts):**
   - TCP CUBIC achieves **$2.7006\text{ pkts/step}$** with $0.2$ drops.
   - ML-TCP achieves **$2.4405\text{ pkts/step}$** with $1.7$ drops. In micro-buffers, queue build-up occurs faster than the multi-step gradient history can detect.
3. **In Direct Competition Against TCP Reno and CUBIC:**
   - Reno takes **$5.88\text{ pkts/step}$** ($98\%$ share).
   - ML-TCP is starved at **$0.088\text{ pkts/step}$** ($1.5\%$ share).
4. **Against Non-ML Persistence in Clean Channels:**
   - Persistence achieves **$3.6676\text{ pkts/step}$** on S2 vs. Random Forest's **$3.5358\text{ pkts/step}$**, with equivalent latency ($50.6 - 50.9\text{ ms}$).

---

## 9. Deliverables & Reproduction Guide

### Publication-Grade Deliverables
- [`time_series_3_scenarios.png`](file:///c:/Users/pardh/OneDrive/Desktop/lp/time_series_3_scenarios.png): Time-domain traces of sending rate, queue length, and RTT across Clean (S1), Random Walk (S2), and Wireless Loss (S4) scenarios.
- [`goodput_rtt_frontier_comparison.png`](file:///c:/Users/pardh/OneDrive/Desktop/lp/goodput_rtt_frontier_comparison.png): Pareto frontier trade-offs across controllers.
- [`scenario_benchmarks_barchart.png`](file:///c:/Users/pardh/OneDrive/Desktop/lp/scenario_benchmarks_barchart.png): Goodput and RTT bar charts with 95% CI error bars across all held-out scenarios.
- [`fairness_competition_chart.png`](file:///c:/Users/pardh/OneDrive/Desktop/lp/fairness_competition_chart.png): Jain's Fairness Index in symmetrical multi-flow and mixed competition.
- [`pareto_frontier_all_24_rules.png`](file:///c:/Users/pardh/OneDrive/Desktop/lp/pareto_frontier_all_24_rules.png): 24-configuration Rule sweep vs. Baselines.

### Single-Command Reproduction
Install dependencies:
```bash
pip install -r requirements.txt
```
Run full reproduction suite with fixed seeds (`seeds.json`):
```bash
python reproduce_all.py
```
