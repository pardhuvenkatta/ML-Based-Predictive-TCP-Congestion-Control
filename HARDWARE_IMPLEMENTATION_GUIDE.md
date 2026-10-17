# Hardware Implementation Guide: ML-Based Predictive TCP Congestion Control

**Project Title:** ML-Based Predictive TCP Congestion Control: Enhancing Network Reliability through Machine Learning-Driven Proactive Congestion Management  
**Team Members:**  
- **BAHAAR** (Reg. No: 24BCE1760)  
- **PARDHU** (Reg. No: 24BCE1754)  

---

## 1. Executive Summary & Hardware Viability

In traditional computer network deployments, TCP congestion control runs either inside the operating system's kernel stack (sender-driven) or on intermediate routing hardware (router-driven Active Queue Management like RED/CoDel).

When transitioning our simulation into physical hardware, **four viable hardware architectures** can be built depending on lab equipment and demonstration goals:

| Architecture | Hardware Components Required | Complexity | Best Suited For |
| :--- | :--- | :--- | :--- |
| **Option 1: IoT Telemetry & Physical Queue Monitor** | ESP32 / Arduino Uno, 0.96" I2C OLED, 3 LEDs, Buzzer | **Low (Recommended for Viva/Expo)** | Physical table demonstration during project review / evaluation |
| **Option 2: Raspberry Pi Linux Router Testbed** | Raspberry Pi 4/5 (or mini-PC) with 2 Ethernet ports, 2 laptops | **Medium** | Real-world IP packet forwarding with physical bottleneck link |
| **Option 3: SDN Hardware Switch Testbed** | OpenFlow Hardware Switch / Open vSwitch on Raspberry Pi | **High** | Software-Defined Networking (SDN) research demonstrations |
| **Option 4: Line-Rate SmartNIC / NetFPGA** | AMD Xilinx NetFPGA SUME / SmartNIC, PCIe Host | **Very High** | High-speed data center line-rate packet processing (10Gbps+) |

---

## 2. Option 1: Physical IoT Telemetry & Queue Monitor (Recommended for Project Review)

### A. Overview
This setup connects an **ESP32 or Arduino Uno/Nano** to your computer via USB Serial. As the simulation or socket sender runs, it continuously streams real-time telemetry frames (`RTT`, `Queue Occupancy`, `Sending Rate`, `ML Throttle Trigger`) to the microcontroller.

The hardware device visually proves proactive buffer management to the review panel:
- **0.96" I2C OLED Screen:** Displays live RTT (ms), Sending Rate, and an animated queue fill progress bar (0–50 packets).
- **Green LED:** Router buffer is safe ($< 70\%$ capacity).
- **Yellow LED:** Caution zone ($70\% - 80\%$ capacity).
- **Red LED + Audio Buzzer:** Flashes and chirps the exact millisecond our Random Forest model predicts imminent congestion ($> 80\%$ / 40 packets) and executes the proactive 30% rate reduction!

```
+-------------------------------------------------------------+
|                     PC / Laptop                             |
|  [network_simulation.py] or [ml_tcp_simulation.py]          |
|                          |                                  |
|            USB Serial (115200 Baud, COM3)                   |
+--------------------------|----------------------------------+
                           v
+-------------------------------------------------------------+
|                 ESP32 / Arduino Node                        |
|                                                             |
|   +--------------------+     +---------------------------+  |
|   | 0.96" OLED (I2C)   |     | Status LEDs & Buzzer      |  |
|   |  RTT: 111.5 ms     |     |  [GREEN]  < 70% Safe      |  |
|   |  Queue: [====   ]  |     |  [YELLOW] 70-80% Caution  |  |
|   |  Rate: 2.98 pkts/s |     |  [RED]    > 80% Throttle  |  |
|   +--------------------+     |  [BUZZER] Alert Chirp     |  |
|                              +---------------------------+  |
+-------------------------------------------------------------+
```

### B. Bill of Materials (BOM)
1. **ESP32 DevKit V1** or **Arduino Uno / Nano** (~$4 - $6)
2. **0.96 inch I2C OLED Display (SSD1306, 128x64)** (~$2 - $3)
3. **3x 5mm LEDs** (Green, Yellow, Red) + 3x $220\ \Omega$ Resistors
4. **1x 5V Passive/Active Buzzer**
5. **Breadboard & Jumper Wires**
6. **Micro-USB Cable**

### C. Wiring Diagram & Pin Connections (ESP32)

| Component | Component Pin | ESP32 Pin | Purpose |
| :--- | :--- | :--- | :--- |
| **OLED Display** | VCC | `3.3V` | Power supply |
| | GND | `GND` | Ground |
| | SCL | `GPIO 22` | I2C Clock |
| | SDA | `GPIO 21` | I2C Data |
| **Green LED** | Anode (+) via $220\Omega$ | `GPIO 12` | Buffer safe indicator ($< 35$ packets) |
| **Yellow LED** | Anode (+) via $220\Omega$ | `GPIO 14` | Approaching threshold ($35 - 40$ packets) |
| **Red LED** | Anode (+) via $220\Omega$ | `GPIO 27` | Congestion predicted ($> 40$ packets) |
| **Buzzer** | Positive (+) | `GPIO 26` | Proactive throttle alert sound |
| **All Grounds**| Cathodes (-) | `GND` | Common ground rail |

*(Note: For Arduino Uno, use `A4` for SDA, `A5` for SCL, and digital pins `8`, `9`, `10`, `11` for LEDs/Buzzer).*

### D. Software Code Files Provided in Workspace
- **Firmware:** `hardware_monitor.ino` (Flash this to ESP32 using Arduino IDE).
- **Serial Bridge:** `hardware_serial_bridge.py` (Streams live telemetry from Python to COM port).

### E. How to Run:
```bash
# Flash hardware_monitor.ino to ESP32/Arduino via Arduino IDE
# Run the Python hardware bridge:
python hardware_serial_bridge.py COM3
```

---

## 3. Option 2: Raspberry Pi / Linux Dual-Homed Router Testbed

### A. Network Topology
To test ML-TCP against real physical IP packets, configure a 3-device physical network:

```
[ Laptop 1: TCP Sender ]           [ Raspberry Pi 4: Bottleneck Router ]          [ Laptop 2: TCP Receiver ]
       (Client)                                   (Router)                                  (Server)
  IP: 192.168.10.2   ===========>        eth0: 192.168.10.1 (Ingress)        ===========>  IP: 192.168.20.2
  Runs ML-TCP Socket                     eth1: 192.168.20.1 (Egress)                       Runs iperf3 / Sink
                                         Linux `tc` enforces 50-pkt buffer
```

### B. Configuring the Raspberry Pi Router Buffer
On the Raspberry Pi, enable IP forwarding and configure Linux Traffic Control (`tc`) to enforce the exact bottleneck parameters from our project ($50\text{ packets}$ buffer limit, $10\text{ Mbps}$ bandwidth):

```bash
# 1. Enable IPv4 packet routing on Raspberry Pi
sudo sysctl -w net.ipv4.ip_forward=1

# 2. Clear existing queueing disciplines on egress interface (eth1)
sudo tc qdisc del dev eth1 root 2>/dev/null

# 3. Create a Token Bucket Filter (TBF) bottleneck queue with exactly 50 packets limit
# This mirrors our simulation: capacity = 50 packets!
sudo tc qdisc add dev eth1 root handle 1: tbf \
    rate 10mbit \
    burst 15k \
    limit 50p

# 4. (Optional) Inject 50ms base propagation delay to match base_rtt = 50ms:
sudo tc qdisc add dev eth1 parent 1: handle 10: netem delay 50ms
```

### C. Extracting Live Kernel Telemetry on Sender
On the sender laptop, real TCP socket telemetry is extracted directly from the Linux TCP kernel via the `TCP_INFO` socket option:

```python
import socket
import struct

# Create TCP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("192.168.20.2", 5001))

# Extract real-time kernel telemetry (RFC standard TCP_INFO struct)
tcp_info = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 104)
# Unpack RTT (tcpi_rtt), RTT variation, and congestion window (tcpi_snd_cwnd)
tcpi_rtt, tcpi_rttvar, tcpi_snd_cwnd, tcpi_unacked = struct.unpack(
    "IIII", tcp_info[28:44]
)

# Convert to simulation features:
# RTT in ms = tcpi_rtt / 1000.0
# Queue estimate = tcpi_unacked (packets in flight)
# Feed directly into rf_congestion_model.pkl!
```

---

## 4. Option 3: Software-Defined Networking (SDN) OpenFlow Switch

In modern enterprise networks, routers are separated into a **Control Plane** (central controller) and **Data Plane** (hardware switches):
1. **Hardware Switch:** Raspberry Pi running Open vSwitch (OVS) with physical Ethernet ports.
2. **Controller (Ryu / POX):** A Python-based OpenFlow controller periodically polls port queue telemetry using `OFPQueueStatsRequest`.
3. **ML Closed Loop:**
   - When port statistics indicate queue depth $> 80\%$, the controller injects an OpenFlow Meter rule (`OFPFlowMod` with `OFP_METER`) that clamps the ingress rate by 30%.
   - Prevents buffer drops across all physical ports connected to the switch!

---

## 5. Option 4: Line-Rate In-Network Acceleration (eBPF / XDP & SmartNIC)

For high-performance data center networks ($10\text{ Gbps} - 100\text{ Gbps}$), sending packets to Python in user space creates latency bottlenecks.

### How ML-TCP Runs at Line Rate:
1. **eBPF / XDP (eXpress Data Path):**
   - The Random Forest decision rules are compiled into native eBPF bytecode.
   - The eBPF program hooks into the Network Interface Card (NIC) driver.
   - For every incoming Ethernet frame, eBPF extracts the IP/TCP header in $< 50\text{ nanoseconds}$, runs tree branch evaluation, and sets the **Explicit Congestion Notification (ECN)** bits (`CE = 11`) in the IP header when queue occupancy $> 80\%$.
2. **SmartNIC / NetFPGA:**
   - Using Xilinx Vivado HLS (High-Level Synthesis), the decision tree logic is synthesized directly into FPGA lookup tables (LUTs), making throttling decisions in hardware clock cycles without CPU intervention.

---

## 6. Viva / Project Review Q&A: Hardware Possibilities

When your project guide or examiner asks: *"How can this software simulation be realized in actual hardware?"*, you can confidently present this structured answer:

> **Answer:**  
> *"Our project can be implemented in hardware across three tiers:*  
> 1. *For physical demonstration and visualization, we built an **IoT hardware telemetry node using an ESP32 with an OLED display and RGB indicators**. It connects over serial to visually track router queue depth and sounds an alarm when the ML model triggers a 30% proactive rate reduction.*  
> 2. *For a real networking testbed, we can use a **Raspberry Pi 4 configured as an intermediate Linux router**. Using Linux Traffic Control (`tc` and `netem`), we create a physical bottleneck buffer of 50 packets and a 50ms delay. The sender laptop queries socket stats via `getsockopt(TCP_INFO)` and runs our Random Forest model to pace packets.*  
> 3. *For industrial high-speed networks (10Gbps+), our Random Forest model can be compiled into **eBPF/XDP kernel bytecode or FPGA SmartNICs**, marking TCP ECN bits at hardware line-rate with sub-microsecond latency."*
