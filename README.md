# ML-Based Predictive TCP Congestion Control
### Enhancing Network Reliability through Machine Learning-Driven Proactive Congestion Management

**Course:** Computer Networks  
**Team Members:**  
- **BAHAAR** (Reg. No: 24BCE1760)  
- **PARDHU** (Reg. No: 24BCE1754)  

---

## 📌 Problem Statement & Objective
Traditional TCP congestion control algorithms (such as Reno, Cubic, and standard AIMD) rely on **reactive feedback**. They only react *after* network congestion has already resulted in packet loss, timeout spikes, or bufferbloat. This delayed detection causes severe throughput oscillations, packet drops, and bandwidth underutilization.

**Our Objective:**  
Develop a **proactive machine learning framework** that continuously monitors network telemetry (`RTT`, `RTT_Change`, `Queue_Size`, `Sending_Rate`) to predict router queue congestion *before* packet loss occurs and adaptively throttles the transmission rate by 30%, maintaining zero packet loss and 99.9% link saturation.

---

## 🚀 Quick Start Guide

### 1. Requirements
Ensure Python 3.8+ is installed with the necessary dependencies:
```bash
pip install pandas scikit-learn matplotlib numpy joblib
```

### 2. How to Run: Half Implementation vs. Full Implementation

| Milestone | What It Does | Command to Run |
| :--- | :--- | :--- |
| **Half Implementation** *(Baseline)* | Runs traditional reactive TCP (AIMD), logs baseline metrics (Loss: 103 packets, Throughput: 2.9878), and generates the 5,000-row `network_data.csv` dataset. | `python network_simulation.py`<br>*or* `python main.py --mode half` |
| **Train ML Model** | Trains the Random Forest classifier on telemetry data, benchmarks against Decision Tree & Logistic Regression (Slide 6 table), and saves `rf_congestion_model.pkl`. | `python train_model.py`<br>*or* `python main.py --mode train` |
| **Full Implementation** *(ML-TCP)* | Integrates the trained ML model in a closed loop, proactively reducing rate by 30% when queue $> 80\%$. Achieves **0 packet loss** and **2.9850 throughput**! | `python ml_tcp_simulation.py`<br>*or* `python main.py --mode full` |
| **Threshold Benchmarks** | Tests the 4 experimental thresholds from Slide 4 (90%/30%, 90%/40%, 80%/30%, 70%/30%). | `python ml_tcp_simulation.py`<br>*or* `python main.py --mode benchmark` |
| **Generate Comparisons** | Produces side-by-side multi-panel dashboards comparing AIMD vs ML-TCP. | `python compare_results.py`<br>*or* `python main.py --mode compare` |
| **Complete Pipeline** | Executes all modules end-to-end automatically. | `python main.py --mode all` |

---

## 📊 Summary of Results & Benchmarks

### A. Performance Comparison: Baseline vs Proactive ML-TCP
| Metric | Traditional Reactive AIMD (Half Implementation) | Proactive ML-TCP (Full Implementation) | Improvement |
| :--- | :--- | :--- | :--- |
| **Congestion Response** | Reactive (after buffer drops packets) | Proactive (predicts queue overflow) | Intelligent Headroom Control |
| **Total Packet Loss** | **103 Packets** | **0 Packets** | **100% Elimination of Loss** |
| **Raw Link Throughput** | 2.9878 pkts/step | 2.9926 pkts/step | Full Link Saturation |
| **Effective Useful Goodput** | **2.8249 pkts/step** *(penalized by retransmissions & stalls)* | **2.9926 pkts/step** *(100% useful data delivery)* | **+5.9% Higher Effective Goodput!** |
| **Sending Rate Dynamic Range** | 1.79 – 5.15 pkts/step (Violent Sawtooth) | 2.92 – 3.05 pkts/step (Ultra-Stable) | Smooth, predictable transmission |
| **Buffer Utilization** | Buffer bloat (Hits 50-packet ceiling) | Safe buffer headroom ($\le 40$ pkts) | Prevents Buffer Overflow |
| **RTT Stability** | Violent oscillation (118 ms – 145 ms) | Bounded, stable RTT ($\sim 122$ ms peak) | -96.6% Lower Jitter (0.63 ms) |

> **💡 Understanding the Throughput & Graph Difference:**
> 1. **Why was raw throughput close at first glance?** The router bottleneck is physically capped at 3.0 packets/step. Both protocols saturate the link capacity. However, Traditional AIMD wastes bandwidth **retransmitting 103 dropped packets** and suffers recovery pipeline stalls. ML-TCP achieves **+5.9% higher effective goodput** with zero wasted bandwidth!
> 2. **Instantaneous Rate Difference:** In `throughput_comparison.png`, Traditional TCP's sending rate swings wildly between 1.8 and 5.2 pkts/step in a chaotic sawtooth, while ML-TCP is a solid, clean, stable green line right at optimal capacity.
> 3. **Slide 4 Sensitivity Proof:** As demonstrated in `threshold_sensitivity_chart.png`, tuning the ML threshold directly impacts throughput (2.9910 for 90% $\to$ 2.8120 for 90%/40% cut $\to$ 2.9850 for 80% $\to$ 2.6500 for 70%).

---

### B. Adaptive Rate Control Threshold Benchmark (Slide 4 Table Validation)
| Threshold % (Packet Limit) | Reduction % | Total Packet Loss | Throughput (Packets/Step) | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **90% (45 Packets)** | 30% | ~18 packets | 2.9910 | **Failed:** Buffer headroom too small; feedback lag causes queue overflow during traffic spikes. |
| **90% (45 Packets)** | 40% | ~8 packets | 2.8120 | **Failed:** Reduces packet loss slightly, but throughput drops significantly due to over-throttling. |
| **80% (40 Packets)** | **30%** | **0 packets** | **2.9850** | **Optimal: Zero packet loss while maintaining full link capacity.** |
| **70% (35 Packets)** | 30% | 0 packets | 2.6500 | **Sub-optimal:** Zero loss, but triggers false alarms early, lowering throughput needlessly. |

### C. ML Algorithm Evaluation (Slide 6 Comparison Table)
- **Random Forest Classifier (Selected Optimal Model):**
  - **Detection Accuracy:** $100.00\%$
  - **Precision / Recall / F1:** $1.00 / 1.00 / 1.00$
  - **Inference Latency:** $\sim 34\ \mu s/\text{packet}$ (ultra-fast for microsecond networking loops)
  - **Feature Importance:** RTT ($51.59\%$), Queue Size ($44.51\%$), Sending Rate ($3.20\%$), RTT Change ($0.71\%$)

---

## 🔌 Hardware Implementation Possibilities

Detailed hardware blueprints, schematics, and testbed configurations are provided in [HARDWARE_IMPLEMENTATION_GUIDE.md](HARDWARE_IMPLEMENTATION_GUIDE.md):

1. **Option 1: Physical IoT Telemetry & Router Monitor (ESP32 / Arduino + 0.96" OLED + LEDs + Buzzer)**
   - Displays real-time queue fill progress bar, RTT, and sounds an audio-visual alarm during ML proactive rate throttling.
   - Code: `hardware_monitor.ino` & `hardware_serial_bridge.py`.
   - **Recommended for live project exhibition and viva.**
2. **Option 2: Raspberry Pi / Linux Router Testbed**
   - Raspberry Pi acting as an intermediate dual-homed router using Linux Traffic Control (`tc` / `netem`) enforcing a 50-packet bottleneck buffer.
   - Real sockets query kernel telemetry via `getsockopt(TCP_INFO)` to pace transmission in real-time.
3. **Option 3: SDN OpenFlow Hardware Switch**
   - Ryu OpenFlow controller querying switch port queue counters and injecting dynamic meter rate mods.
4. **Option 4: Line-Rate In-Network Acceleration (eBPF / XDP & SmartNIC / FPGA)**
   - Executing Random Forest decision trees inside kernel packet hooks in $<50\text{ ns}$ to set TCP ECN bits.

---

## 📁 Repository Structure
```
├── network_simulation.py        # Baseline reactive AIMD simulator (Half Implementation)
├── network_data.csv             # 5,000-row telemetry dataset generated from baseline
├── train_model.py               # Machine learning training & latency benchmark script
├── rf_congestion_model.pkl      # Serialized Random Forest classifier
├── ml_tcp_simulation.py         # Full proactive ML-TCP simulation & benchmark engine
├── compare_results.py           # Multi-panel comparison dashboard generator
├── main.py                      # Unified CLI launcher with interactive menu
├── hardware_monitor.ino         # Arduino / ESP32 hardware OLED & LED monitor sketch
├── hardware_serial_bridge.py    # Python-to-microcontroller serial telemetry bridge
├── HARDWARE_IMPLEMENTATION_GUIDE.md # Comprehensive 4-architecture hardware guide
├── rtt_plot.png                 # Baseline RTT oscillation plot (Slide 9)
├── ml_tcp_rtt_plot.png          # ML-TCP stabilized RTT dual-comparison plot
├── throughput_comparison.png    # Dedicated Throughput & Goodput comparative plot
├── comparison_dashboard.png     # 4-panel comprehensive comparative dashboard
├── threshold_sensitivity_chart.png # Slide 4 benchmark sensitivity bar chart
└── README.md                    # Project documentation
```
