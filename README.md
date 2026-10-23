# Scientific Evaluation & Audit of ML-Based Predictive TCP Congestion Control

**Project Title:** Comprehensive Scientific Evaluation of Machine Learning-Based Predictive TCP Congestion Control  
**Authors:** BAHAAR (Reg. No: 24BCE1760), PARDHU (Reg. No: 24BCE1754)  
**Course:** Computer Networks  
**Repository:** `https://github.com/pardhuvenkatta/ML-Based-Predictive-TCP-Congestion-Control.git`  
**Master Single-Command Runner:** `python reproduce_all.py`

---

## 📌 Executive Summary & Abstract

Machine Learning (ML) has frequently been proposed to revolutionize transport-layer congestion control by proactively predicting bufferbloat and packet drops. In this project, we conducted a rigorous, multi-scenario empirical evaluation across 20 pseudo-random seeds per scenario (Seeds 42–61), strict time-ordered train/test partitioning, and strictly sender-observable telemetry (RFC 793/5681 compliant, zero router or queue state leaks).

### Core Scientific Findings
1. **Pacing Dominance:** The primary performance driver of this architecture is **ACK-clocked delivery-rate pacing**: it stabilizes sending rates at available bottleneck capacity, prevents drop-tail queue overflow, and eliminates bufferbloat across clean, high-bandwidth, and moderately congested channels.
2. **Predictor Ablation & Honest Rank:** In closed-loop predictive control on the baseline sweep grid, the Random Forest regressor statistically outperformed Linear Extrapolation ($\Delta U_{1.0} = +0.0089$, 95% CI $[+0.0067, +0.0111]$), but was strictly inferior to a zero-parameter **Persistence baseline** ($\Delta U_{1.0} = -0.0160$, 95% CI $[-0.0199, -0.0121]$).
3. **Where ML Genuinely Wins (Loss-Type Classification):** When used to classify whether an ACK loss is caused by buffer overflow or random wireless channel corruption (suppressing backoff on random loss), a Random Forest classifier **consistently outperformed classical threshold rules** (`rtt_ratio < 1.05`):
   - At 2.0% random loss: RF Classifier achieved **2.7965 pkts/step** (93.2% capacity) vs Non-ML Rule's **2.6792 pkts/step** (89.3%) and Standard Backoff's **2.5233 pkts/step** (84.1%).
   - Under Gilbert-Elliott bursty loss: RF Classifier achieved **2.8438 pkts/step** (94.8% capacity) vs Non-ML Rule's **2.7503 pkts/step** (91.7%) and Standard's **2.6596 pkts/step** (88.7%).
4. **Deep Reinforcement Learning (Gate F Evaluation):** Continuous PPO and SAC agents were trained in a Gymnasium environment across 12 reward formulations $\times$ 3 seeds with domain randomization and a Reno/CUBIC competition curriculum. Both failed Gate F ($\Delta U_{1.0} = -1.7924 \pm 0.3577$ for PPO; $\Delta U_{1.0} = -0.3539 \pm 0.1879$ for SAC), suffering catastrophic policy collapse under long propagation delays (Base RTT = 200ms) and multi-flow scaling.
5. **Copa/Nimbus Competitive-Mode Switching:** Detecting persistent standing queues (`rtt_ratio > 1.20` for $>60$ steps) relieved loss-induced starvation against TCP CUBIC at $B=50$, surging goodput from **0.1595 to 4.7736 pkts/step (+2893.5% gain)** with an insignificant latency cost of only $+6.3\text{ ms}$ when operating alone.

---

## 🔍 Leakage Audit: Old Numbers vs. Honest Numbers

In preliminary prototype implementations, models mistakenly utilized internal router state variables (`Queue_Size`), achieving artificial "perfection":

| Metric / Dimension | Unaudited Prototype (Cheating) | Audited Scientific Implementation (Honest) |
| :--- | :--- | :--- |
| **Telemetry Source** | Router internal state (`Queue_Size`) | Strictly host-side ACK telemetry (RFC 793/5681) |
| **Input Features** | `[RTT, RTT_Change, Queue_Size, Sending_Rate]` | `[rtt_ratio, grad_1, grad_3, grad_5, delivery_rate, send_rate, loss_flag]` |
| **Evaluation Seeds** | 1 seed (Fixed run) | **20 Pseudo-Random Seeds (42–61) with 95% CIs** |
| **Train/Test Split** | Random shuffle (temporal data leakage) | **Strict Time-Ordered Split & Held-Out Scenarios** |
| **Predictor Finding** | "RF achieves 100% loss-free control" | Pacing drives loss reduction; **Persistence beats RF** ($\Delta U = -0.0160$) |
| **Deep RL Verdict** | Assumed optimal | **Gate F Rejected:** PPO and SAC fail to beat lightweight hybrid rule |

---

## 🚀 Reproduction Suite & Quick Start

### 1. Installation
Ensure Python 3.8+ is installed. Install all project dependencies:
```bash
pip install -r requirements.txt
```

### 2. Single-Command End-to-End Reproduction
Execute the master reproduction runner to regenerate every CSV, table, and publication figure:
```bash
python reproduce_all.py
```

### Expected Runtime Breakdown (Standard 8-Core Desktop/Laptop)
- **Step 1 (Supervised Models & Loss Classification):** ~45 seconds
- **Step 2 (RL Controllers PPO & SAC, Distillation & Gate F):** ~90 seconds
- **Step 3 (Realistic Robustness, 16-Flow Scaling & Competition):** ~75 seconds
- **Step 4 (Final Robustness Deliverables & Plots):** ~30 seconds
- **Total Pipeline Execution Time:** **~3.5 – 4.5 minutes**

---

## 📊 Summary of Benchmark Results

### A. Supervised Offline & Closed-Loop Performance
| Model | Target Variable | Offline MAE | Offline $R^2$ | Inference Time | Model Size | Closed-Loop HO1 Utility ($U_{1.0}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | `rtt_ratio (t+5)` | **0.0142** | **0.9986** | **0.02 $\mu s$** | **0.0 KB** | **-2.6923** (0.0 drops) |
| **Linear Extrapolation** | `rtt_ratio (t+5)` | 0.0213 | 0.9947 | 0.05 $\mu s$ | 0.0 KB | -2.7213 (0.0 drops) |
| **Random Forest** | `rtt_ratio (t+5)` | 0.0169 | 0.9985 | 2.85 $\mu s$ | 3,547.0 KB | -2.7143 (0.0 drops) |
| **LightGBM** | `rtt_ratio (t+5)` | 0.0162 | 0.9986 | 1.10 $\mu s$ | 286.6 KB | -2.7077 (0.0 drops) |
| **Small PyTorch MLP** | `rtt_ratio (t+5)` | 0.0176 | 0.9987 | 0.25 $\mu s$ | 13.3 KB | -2.7119 (0.0 drops) |
| **PyTorch GRU (8-step)** | `rtt_ratio (t+5)` | 0.0623 | 0.9748 | 4.51 $\mu s$ | 17.7 KB | -2.7038 (0.0 drops) |
| **PyTorch 1D-CNN (8-step)**| `rtt_ratio (t+5)` | 0.0251 | 0.9981 | 1.58 $\mu s$ | 10.3 KB | -2.7150 (0.0 drops) |

### B. Reinforcement Learning Gate F Summary
- **PPO Overall Difference vs. Best Non-ML Rule:** $\Delta U_{1.0} = -1.7924 \pm 0.3577$ (Statistically significant degradation; Gate F NOT PASSED)
- **SAC Overall Difference vs. Best Non-ML Rule:** $\Delta U_{1.0} = -0.3539 \pm 0.1879$ (Statistically significant degradation; Gate F NOT PASSED)
- **Gate F Verdict:** **NO MEASURABLE LEARNED BENEFIT** for end-to-end RL rate control over the lightweight, delay-aware hybrid pacing rule.

### C. Multi-Flow Scalability up to 16 Concurrent Flows
- **Equal Starts ($N=16$):**
  * Winning Hybrid Rule: **12.0000 pkts/step** (100% capacity), 0.0 loss, $\text{JFI} = 1.000 \pm 0.000$.
  * Reference Vegas: **11.9999 pkts/step**, 0.0 loss, $\text{JFI} = 1.000 \pm 0.000$.
  * Audited BBR: 11.8249 pkts/step, 65.6 loss, $\text{JFI} = 0.208 \pm 0.118$.
  * TCP Reno: Collapses to 2.8191 pkts/step with **3,486 packet drops**.
- **Staggered Starts ($N=16$):**
  * Winning Hybrid Rule: **11.6881 pkts/step**, 0.2 loss, $\text{JFI} = 0.795 \pm 0.015$.
  * Audited BBR: 10.9118 pkts/step, 407.8 loss, $\text{JFI} = 0.117 \pm 0.002$.

---

## 📚 Comparison with Published Literature

1. **Remy (Winstein & Balakrishnan, SIGCOMM 2013):** Lookup-table policies optimized under known prior distributions. Our zero-parameter Persistence baseline and Hybrid Pacing achieve comparable optimality without requiring offline traffic priors.
2. **PCC Vivace (Dong et al., NSDI 2018):** Online rate pacing driven by empirical delivery rate and latency gradients eliminates bufferbloat, confirming that rate pacing dominates black-box regression.
3. **Aurora (Jay et al., ICML 2019):** Deep RL (PPO) applied directly to congestion control suffers severe performance collapse on out-of-distribution RTTs and multi-flow competition, matching our empirical findings.
4. **Orca (Foo et al., SIGCOMM 2020):** Confirms why a hybrid architecture is necessary: fast per-packet dynamics must be governed by classical pacing, while learning operates on larger timescales.
5. **Genet (Zheng et al., SIGCOMM 2022):** Deep RL policies overfit simulation dynamics and fail on micro-buffers ($B=5$), where simple persistence generalizes flawlessly.

---

## 📁 Repository Structure & Artifacts

```
├── reproduce_all.py                            # Master single-command reproduction runner
├── seeds.json                                  # 20 fixed pseudo-random seeds (42-61)
├── requirements.txt                            # Python dependencies
├── FINAL_SCIENTIFIC_AUDIT_REPORT.md            # Comprehensive scientific report
│
├── step1_supervised_and_loss_classification.py # Step 1: RF, LightGBM, MLP, GRU, CNN & Loss Classifier
├── step2_rl_controllers_ppo_sac.py             # Step 2: Gymnasium TCP env, PPO/SAC sweep, distillation
├── step3_realistic_robustness_and_competition.py# Step 3: Realistic channels, 16 flows, Copa competition
├── comprehensive_final_robustness.py           # Step 4: Mahimahi trace, RTTxBuffer factorial, stability
│
├── supervised_offline_metrics.csv              # Offline MAE, R2, latency, footprint
├── loss_classifier_evaluation.csv              # Random & bursty loss classification results
├── supervised_closed_loop_held_out.csv         # Closed-loop supervised held-out results
├── rl_reward_sweep_training.csv                # 12-setting RL reward sweep results
├── rl_distillation_comparison.csv              # Decision tree and MLP distillation metrics
├── rl_held_out_comparison.csv                  # Complete RL held-out benchmark
├── rl_paired_differences.csv                   # Paired per-seed differences for Gate F
├── realistic_conditions_benchmark.csv          # Cellular, ACK compression, AQM benchmark
├── multi_flow_scaling_16flows.csv              # 2, 4, 8, 16 flows equal vs. staggered starts
├── competitive_mode_switching.csv              # Starvation relief vs Reno/CUBIC
├── factorial_rtt_buffer_benchmark.csv          # Full factorial Base RTT x Buffer benchmark
├── mahimahi_trace_benchmark.csv                # Public Mahimahi LTE/5G trace replay
├── stability_and_recovery_metrics.csv          # Settling time, capacity & burst recovery
├── overhead_and_kernel_feasibility.csv         # Kernel/userspace feasibility matrix
│
├── time_series_4_scenarios.png                 # Dynamic rate, queue, and RTT tracking
├── pareto_frontier_all_scenarios.png           # Goodput vs. RTT Pareto frontiers
├── benchmark_barchart_with_ci.png              # Benchmark goodput with 95% CIs
├── multi_flow_competition_chart.png            # Multi-flow JFI scalability (2-16 flows)
├── rl_training_curves.png                      # RL validation reward sweep curves
└── sensitivity_sweep_invariance.png            # +/-50% simulator parameter rank invariance
```

---

## 🎯 Final Scientific Verdict

In network transport protocols, domain-specific physical principles (ACK clocking, Little's Law, and delay gradients) should govern fast timescale rate adjustments, while Machine Learning provides genuine, demonstrable value on orthogonal pattern-classification sub-tasks such as distinguishing wireless corruption loss from queue overflow.
