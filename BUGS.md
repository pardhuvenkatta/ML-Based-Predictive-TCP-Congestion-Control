# ML-TCP Code Audit Bug Tracker (BUGS.md)

This document tracks every bug identified, root cause analyzed, fix applied, test added, severity assigned, and affected benchmark results audited across `src/sim.py`, `src/controllers.py`, `src/models.py`, `src/analysis.py`, `src/experiments.py`, `reproduce_all.py`, and `verify_consolidation.py`.

---

## Bug Inventory

| Bug ID | File:Line | Symptom | Root Cause | Fix | Test Added | Severity | Affected Results |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BUG-01** | `src/sim.py:58` | Predictive agent starts with oracle capacity knowledge | `agent.cached_pred_cap = float(fixed_capacity)` was injected before simulation loop | Removed oracle assignment; set uncalibrated default `1.05` | `tests/test_observability.py::test_no_oracle_leakage_in_sim` | Critical | Phase 2 initial capacity steps |
| **BUG-02** | `src/sim.py:175` | Initial sending rate in trace simulation set to exact ground-truth trace capacity | `agent.rate = float(cap_trace[0])` leaked trace step 0 capacity to agent | Initialized probe rate to `1.05` pkts/step | `tests/test_observability.py::test_agent_initialization_no_oracle` | Major | Phase 2 trace evaluation |
| **BUG-03** | `src/sim.py:181` | Predictive capacity cache in trace simulation initialized with ground-truth trace capacity | `agent.cached_pred_cap = float(cap_trace[0])` leaked oracle trace capacity | Initialized cache to `1.05` pkts/step | `tests/test_observability.py::test_agent_initialization_no_oracle` | Major | Phase 2 trace evaluation |
| **BUG-04** | `src/sim.py:341` | In multi-flow simulation, every flow agent was initialized with oracle fair-share | `a.rate = bottleneck_cap / n_flows` gave flows oracle knowledge of global link and flow count | Initialized each flow rate to `1.05` pkts/step | `tests/test_sim.py::test_multi_flow_fairness` | Major | Multi-flow initial convergence |
| **BUG-05** | `src/controllers.py:120-121` | Reno pacing rate hardcoded 50 ms path delay constant | `(self.cwnd / max(10.0, curr_rtt)) * 50.0` assumed static 50 ms base RTT | Scaled rate with measured `base_rtt_est = self.min_rtt if self.min_rtt is not None else 50.0` | `tests/test_controllers.py::test_rtt_scale_invariance` | Major | Reno baseline in long-RTT / varied-RTT suites |
| **BUG-06** | `src/controllers.py:146-147` | CUBIC pacing rate hardcoded 50 ms path delay constant | `(self.cwnd / max(10.0, curr_rtt)) * 50.0` assumed static 50 ms base RTT | Scaled rate with measured `base_rtt_est = self.min_rtt if self.min_rtt is not None else 50.0` | `tests/test_controllers.py::test_rtt_scale_invariance` | Major | CUBIC baseline in long-RTT / varied-RTT suites |
| **BUG-07** | `src/controllers.py:282-285` | `CapacityAgentBase` initialized EWMA and Holt estimators with link capacity constant | Hardcoded `self.ewma_cap = 3.0; self.cached_pred_cap = 3.0` (canonical link capacity) | Changed initial estimators to probe rate `1.05` | `tests/test_observability.py::test_no_oracle_constants_in_controllers` | Major | Capacity forecast warmup |
| **BUG-08** | `src/controllers.py:322` | Persistence capacity forecast fell back to 3.0 pkts/step on empty delivery history | Hardcoded `3.0` constant | Changed fallback to sender-observable `self.est_delivery_rate` | `tests/test_controllers.py::test_edge_cases` | Minor | Persistence forecast step 0 |
| **BUG-09** | `src/controllers.py:220-223` | CopaAgent target rate diverged to hundreds of pkts/step | Millisecond-to-second mismatch: ms divided by 1000 produced minute `q_delay` (~0.005) causing `1.0 / (delta * q_delay) = 400` pkts/step | Used normalized queue delay ratio `(curr_rtt - min_rtt) / min_rtt` and target rate formula `lambda = est_delivery_rate / (1.0 + delta * q_delay_ratio)` | `tests/test_controllers.py::test_copa_queue_delay_control` | Critical | Phase 3 Opponent Competition against Copa |
| **BUG-10** | `src/experiments.py:220` | `run_master_scorecard` loaded pre-saved CSV instead of executing live simulations | Scorecard runner bypassed simulator execution and loaded `results/results_summary.csv` | Replaced with dynamic simulation synthesis running all 14 scenarios live across all phases | `tests/test_reproducibility.py::test_reproduce_scorecard_live` | Critical | Master Scorecard reproducibility |
| **BUG-11** | `src/models.py:17, 73, 126` | Mutable default argument `seeds=[42, 43, 44]` in model training functions | Python mutable list default persists state across calls if mutated | Changed signature to `seeds=None` and initialized `[42, 43, 44]` inside function body | `tests/test_ml.py::test_model_training_mutable_defaults` | Minor | None (defensive hygiene) |
| **BUG-12** | `src/controllers.py:347` | `CapacityAgentBase.update_rate()` crashed with `IndexError` on uninitialized ACK history | Evaluated `self.rtt_history[-1]` without checking if history is empty before step 0 ACK | Guarded with `self.rtt_history[-1] if self.rtt_history else (self.min_rtt or 50.0)` | `tests/test_controllers.py::test_edge_cases_and_robustness` | Major | Step 0 rate actuation |
| **BUG-13** | `src/controllers.py:179` | `ReferenceVegasAgent` queue backlog calculation compared rate difference against packet limits | Unit mismatch: `diff = rate * (1 - min_rtt/rtt)` is bounded by `< 3.0` for all RTT, so `diff > beta (3.0)` was dead code | Replaced with physical queue backlog estimator `diff = rate * max(0, curr_rtt - min_rtt) / min_rtt * 5.0` | `tests/test_controllers.py::test_controller_dynamics_reno_vs_delay_based` | Critical | TCP Vegas baseline evaluation |
| **BUG-14** | `src/controllers.py:100, 310` | OpponentAgentBase had 86.2% false-positive competitive mode triggers when running alone | Lacked queue drain headroom when `rtt_ratio > 1.12`, pacing at `100% * target` and locking in self-induced standing queue | Added 6% queue drain headroom (`0.94 * target`) when queue inflated; FPR alone dropped to 0.0% | `tests/test_controllers.py::test_competitive_mode_alone_fpr` | Critical | Phase 3 Opponent Competition FPR |
| **BUG-15** | `src/controllers.py:80, 100` | Reference controller underperformed on short-RTT links (20 ms) | Fixed `1.12` ratio threshold corresponds to only 2.4 ms delay at 20 ms RTT (1 packet adds 10% delay), throttling sender prematurely | Scaled thresholds adaptively with measured RTT: `flush_ratio = 1.0 + max(0.12, 3.5 / base_rtt)` | `tests/test_controllers.py::test_rtt_scale_invariance` | Major | RTT scale-invariance suite (20 ms) |
| **BUG-16** | `src/models.py:46-52` | `train_lofo_loss_models()` crashed with `ValueError: This solver needs samples of at least 2 classes` | Paced sender never exceeded buffer capacity 30.0 with cap=1.2, producing 0 congestion losses (`y=1`) | Scaled to `cap=0.8` drop and `buffer_capacity=10.0` during loss data collection | `tests/test_ml.py::test_lofo_loss_model_separation` | Critical | LOFO Loss Classifier training |

---

## Numerical Impact on Results

Before-and-after verification of affected benchmark scenarios:

| Scenario / Metric | Pre-Audit Baseline | Post-Fix Audited | Delta / Impact | Scientific Claim Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Copa Competition (Phase 3)**: Host Goodput | 1.84 pkts/step | 1.82 pkts/step | -1.1% | None; Copa now behaves as a stable delay-based controller with bounded queue |
| **Copa Competition**: Copa Opponent Goodput | 0.98 pkts/step | 1.01 pkts/step | +3.0% | Eliminates artificial rate explosion in opponent Copa |
| **Reno Long-RTT (300 ms)**: Reno Goodput | 0.72 pkts/step | 0.89 pkts/step | +23.6% | Reno achieves expected bandwidth-delay product scaling on long RTTs |
| **CUBIC Long-RTT (300 ms)**: CUBIC Goodput | 0.94 pkts/step | 1.18 pkts/step | +25.5% | CUBIC achieves proper RTT-adaptive window scaling |
| **Multi-Flow 8-Flow JFI**: Initial transient | 0.999 (artificial) | 0.984 (realistic) | -1.5% | Flows dynamically converge to fair-share instead of starting at oracle fair-share |
| **Opponent Alone**: Competitive FPR | 86.2% (spurious triggers) | 0.0% (clean isolation) | -86.2% | Resolves false-positive trigger defect; competitive mode activates only against genuine opponents |
| **Short-RTT (20 ms)**: Reference Goodput | 2.02 pkts/step | 2.96 pkts/step | +46.5% | Full link utilization achieved on short RTT paths |
| **LOFO Loss Training**: Binary Class Balance | 100% Class 0 (Crashed) | 82% Class 0 / 18% Class 1 | Fixed Crash | Enables valid classification training and benchmarking |
| **Master Scorecard**: Execution Mode | Static CSV read | 100% Live compute | Exact simulation | Eliminates static CSV dependency; full mathematical reproducibility |
