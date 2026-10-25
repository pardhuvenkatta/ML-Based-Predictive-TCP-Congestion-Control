# Scientific Evaluation & Audit of ML-Based Predictive TCP Congestion Control

**Project Title:** Comprehensive Scientific Evaluation and Architectural Consolidation of Machine Learning-Based Predictive TCP Congestion Control  
**Version:** v2.1.0 (Audited Scale-Invariant Coincident-Rise Release)  
**License:** MIT Open-Source License  
**Master CLI Runner:** `python reproduce_all.py`  
**Full Scientific Report & Hardware Guide:** [REPORT.md](REPORT.md)  
**Consolidation & Schema Changelog:** [CHANGELOG.md](CHANGELOG.md)  

---

## 📌 Executive Summary & Abstract

Machine Learning (ML) has frequently been proposed to revolutionize transport-layer congestion control by proactively predicting bufferbloat and packet drops. In this project, we conducted a rigorous, multi-scenario empirical evaluation across 20 pseudo-random seeds per scenario (Seeds 42–61), strict time-ordered train/test partitioning, and strictly sender-observable telemetry (RFC 793/5681 compliant, zero router or queue state leaks).

### Core Scientific Findings
1. **Pacing Dominance & Scale-Invariance:** Pacing proportional to estimated bottleneck delivery rate combined with scale-invariant coincident-rise throttling ($RTT / RTT_{min} > 1.15$ AND $\nabla RTT > 0$) eliminates up to 98% of bufferbloat drops without requiring neural inference on static channels.
2. **Where ML Genuinely Wins (Dynamic Channels):** Machine learning models (Random Forest, GRU, and 1D-CNN capacity forecasters) achieve statistically significant and practically meaningful wins ($> 1\%$ utility gain at Kleinrock weights $d \in \{0.5, 1.0, 2.0\}$ with 95% CIs excluding zero) on non-stationary channels with rapid capacity transitions (deep Markov fades, Mahimahi cellular traces, and step drops).
3. **Where Simple Heuristics Suffice:** In steady-state loss discrimination, matched-budget non-ML rules perform equivalently to multi-parameter classifiers, confirming that sender-observable telemetry features must be paired with genuine non-stationarity to justify neural inference overhead.
4. **Master Audit Scorecard:** Across 14 rigorously audited scenarios evaluated with leave-one-family-out (LOFO) testing, 12 scenarios favor ML or achieve a statistically validated tie, with 0 scenarios where non-ML outperforms ML under practical significance criteria.

---

## 📁 Repository Structure

The repository is consolidated into a minimal, reproducible architecture:

```text
.
├── README.md               # Project overview, quickstart, and core results
├── REPORT.md               # Master technical report (Scientific Audit + Hardware Guide)
├── CHANGELOG.md            # Schema transformation and dataset consolidation changelog
├── LICENSE                 # MIT Open-Source License
├── requirements.txt        # Exact Python dependencies
├── seeds.json              # Fixed random seeds (benchmark 42-61, Copa 42-81, training)
├── reproduce_all.py        # Master command-line entry point for reproduction
├── src/                    # Consolidated modular source code
│   ├── __init__.py         # Package initialization
│   ├── sim.py              # Unified discrete-event network simulator
│   ├── controllers.py      # Canonical controller library (heuristics, rules, ML)
│   ├── models.py           # Neural network & LOFO model architectures
│   ├── experiments.py      # Modular experiment runners matching all audit phases
│   └── analysis.py         # Statistical analysis, U_d utility, t-tests, Pareto frontiers
├── results/                # Complete consolidated experimental telemetry
│   ├── results_all.csv.gz  # Long-format raw per-seed runs and traces (6,800 rows)
│   ├── results_summary.csv # Consolidated summary & scorecard tables (1,499 rows, 73 tables)
│   └── models.zip          # Serialized neural and tree model weights
├── figures/                # Presentation and audit visualizations
│   ├── figures.pdf         # Multi-page compilation of all 31 experimental figures
│   ├── pareto_frontier_audit_3domains.png
│   ├── architecture_diagram.png
│   ├── comparison_dashboard.png
│   └── benchmark_barchart_with_ci.png
└── archive/
    └── old_files.zip       # Full byte-for-byte pre-consolidation archive (169 files)
```

---

## 🚀 Reproduction Suite & Quick Start

### 1. Installation
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Single-Command Experiment Reproduction
All experiments are selectable via `reproduce_all.py`:

```bash
# 1. Verify 50 ms baseline regression suite (< 2.0% tolerance check)
python reproduce_all.py --experiment regression_50ms

# 2. Run Leave-One-Family-Out (LOFO) scientific audit
python reproduce_all.py --experiment lofo_audit

# 3. Generate Goodput vs RTT Pareto frontiers
python reproduce_all.py --experiment pareto_frontiers

# 4. Evaluate factorial RTT grid (20-300 ms, buffers 5-50 pkts)
python reproduce_all.py --experiment long_rtt_grid

# 5. Compile 14-scenario master scorecard
python reproduce_all.py --experiment master_scorecard

# 6. Run 40-seed statistical evaluation against Copa cross-traffic
python reproduce_all.py --experiment copa_robustness

# 7. Evaluate multi-flow scalability (2 to 16 flows)
python reproduce_all.py --experiment multi_flow_scaling

# 8. Evaluate coexistence and fairness against BBR
python reproduce_all.py --experiment bbr_competition

# 9. Reproduce the complete benchmark suite end-to-end
python reproduce_all.py --experiment all

# Quick smoke-test mode across all experiments:
python reproduce_all.py --all --quick
```

---

## 📊 Summary of Benchmark Results

### A. Master 14-Scenario Scorecard ($N=20$ Seeds, Kleinrock $U_d$)
Utility function: $U_d = \ln(\text{goodput}) - d \cdot \ln(\text{RTT})$ for $d \in \{0.5, 1.0, 2.0\}$.

| Domain | Scenario | Evaluated Model | Model GP | Model RTT | Tuned GP | Tuned RTT | $\Delta U_{1.0}$ | 95% CI ($U_{1.0}$) | $p$-value | Scientific Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Loss** | L1: 1% Uniform Loss | RF Loss Classifier | 2.923 | 52.4 ms | 2.915 | 52.4 ms | +0.0028 | [+0.002, +0.003] | $< 10^{-4}$ | Statistically Outperforms |
| **Loss** | L2: 2% Uniform Loss | RF Loss Classifier | 2.857 | 52.6 ms | 2.846 | 52.6 ms | +0.0039 | [+0.003, +0.005] | $< 10^{-4}$ | Statistically Outperforms |
| **Loss** | L3: Burst GE Fading | RF Loss Classifier | 2.906 | 52.8 ms | 2.898 | 52.8 ms | +0.0028 | [+0.002, +0.004] | $< 10^{-4}$ | Statistically Outperforms |
| **Loss** | L4: Fast Rayleigh Fading| RF Loss Classifier | 2.880 | 52.7 ms | 2.871 | 52.7 ms | +0.0031 | [+0.002, +0.004] | $< 10^{-4}$ | Statistically Outperforms |
| **Loss** | L5: Deep Ricean Fading | RF Loss Classifier | 2.894 | 52.7 ms | 2.885 | 52.7 ms | +0.0031 | [+0.002, +0.004] | $< 10^{-4}$ | Statistically Outperforms |
| **Capacity** | D1: Sudden Step Drop | RF Capacity Forecaster| 3.614 | 79.5 ms | 3.528 | 84.1 ms | +0.0815 | [+0.058, +0.105] | $< 10^{-6}$ | **Statistically Outperforms (Large Gain)** |
| **Capacity** | D2: Sinusoidal Walk | RF Capacity Forecaster| 2.977 | 64.2 ms | 2.921 | 67.8 ms | +0.0747 | [+0.051, +0.098] | $< 10^{-6}$ | **Statistically Outperforms (Large Gain)** |
| **Capacity** | D3: Markov Fades | RF Capacity Forecaster| 3.195 | 71.3 ms | 3.136 | 74.8 ms | +0.0673 | [+0.046, +0.089] | $< 10^{-6}$ | **Statistically Outperforms (Large Gain)** |
| **Capacity** | D4: Cellular Trace | RF Capacity Forecaster| 3.421 | 75.1 ms | 3.354 | 78.9 ms | +0.0694 | [+0.048, +0.091] | $< 10^{-6}$ | **Statistically Outperforms (Large Gain)** |
| **Opponent** | C1: Flow Alone | Opponent Classifier | 2.973 | 59.1 ms | 2.969 | 59.2 ms | +0.0040 | [+0.003, +0.005] | $< 10^{-4}$ | Statistically Outperforms |
| **Opponent** | C2: vs Loss-Based (Reno)| Opponent Classifier | 2.812 | 78.4 ms | 2.801 | 78.6 ms | +0.0064 | [+0.004, +0.008] | $< 10^{-4}$ | Statistically Outperforms |
| **Opponent** | C3: vs Loss-Based (CUBIC)| Opponent Classifier| 2.784 | 81.2 ms | 2.772 | 81.5 ms | +0.0080 | [+0.006, +0.010] | $< 10^{-4}$ | Statistically Outperforms |
| **Opponent** | C4: vs Rate-Based (BBR) | Opponent Classifier | 2.914 | 72.1 ms | 2.903 | 72.3 ms | +0.0066 | [+0.004, +0.009] | $< 10^{-4}$ | Statistically Outperforms |
| **Opponent** | C5: vs Delay-Based (Copa)| Opponent Classifier| 2.980 | 58.6 ms | 2.977 | 58.7 ms | +0.0027 | [+0.002, +0.004] | $< 10^{-4}$ | Statistically Outperforms |

### B. Multi-Flow Scalability & Fairness
Across concurrent flow scaling tests from 2 to 16 flows, the reference controller maintains near-perfect Jain's Fairness Index:
- **2 Flows:** JFI = 0.9997
- **4 Flows:** JFI = 0.9994
- **8 Flows:** JFI = 0.9994
- **16 Flows:** JFI = 0.9990

---

## 🔬 Key Visualizations

The primary visualizations are available in `figures/`:
- **`figures/pareto_frontier_audit_3domains.png`**: Empirical Pareto frontier showing goodput vs. RTT trade-offs across loss, capacity, and competition domains.
- **`figures/architecture_diagram.png`**: Full system architecture of the telemetry acquisition, feature extractor, ML decision engine, and kernel pacing actuator.
- **`figures/comparison_dashboard.png`**: Multi-panel comparative dashboard illustrating AIMD reactive loss vs. ML predictive control.
- **`figures/benchmark_barchart_with_ci.png`**: High-resolution benchmark comparison with 95% confidence intervals.
- **`figures/figures.pdf`**: Complete 31-page compilation of all experimental figures.

---

## 🛠️ Hardware Implementation

Complete instructions for hardware validation and testbed reproduction are documented in [REPORT.md (Section 2)](REPORT.md#section-2-hardware-implementation--demonstration-guide):
1. **Physical IoT Telemetry & Queue Monitor:** ESP32/Arduino microcontroller displaying queue size, latency, and predictive throttles via I2C OLED and alerting via buzzer (`hardware_monitor.ino`).
2. **Raspberry Pi Linux Dual-Homed Router:** Hardware router with Linux `tc-tbf` bottleneck queue for end-to-end hardware validation.
3. **OpenFlow SDN & SmartNIC In-Network Acceleration:** Line-rate eBPF / XDP deployment considerations.

---

## 📄 License & Attribution

This project is licensed under the MIT License. See [LICENSE](LICENSE) for full details.
For questions, citations, or complete audit reproductions, refer to [REPORT.md](REPORT.md).
