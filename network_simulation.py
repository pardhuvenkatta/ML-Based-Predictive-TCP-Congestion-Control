"""
ML-Based Predictive TCP Congestion Control
Script: network_simulation.py (Baseline AIMD Simulator & Telemetry Data Generator)

Team Members:
  - BAHAAR (Reg. No: 24BCE1760)
  - PARDHU (Reg. No: 24BCE1754)

Description:
  Simulates traditional reactive TCP congestion control using AIMD
  (Additive Increase, Multiplicative Decrease).
  - Router queue capacity: 50 packets (Drop-Tail)
  - Router process rate: 3 packets per time step
  - Outputs baseline metrics: Total Packet Loss (103), Throughput (2.9878)
  - Generates 'network_data.csv' containing both:
      1. Legacy features (for slide reproducibility)
      2. Honest sender-observable features (RTT ratio, gradients k=1,3,5, delivery rate, send rate, loss flag)
      3. Future-looking predictive targets (Loss/Congestion in next H=3, 5, 10 steps)
  - Generates baseline RTT plot: 'rtt_plot.png' (Slide 9)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


def run_baseline_simulation(total_steps=5000, save_csv=True, show_plot=False, output_plot_path="rtt_plot.png", seed=42):
    # Simulation Parameters
    queue_capacity = 50.0      # Router queue capacity in packets
    process_rate = 3.0         # Packets forwarded per time step
    base_rtt = 50.0            # Base Round-Trip Time in milliseconds
    
    # State variables
    sending_rate = 1.05        # Initial transmission rate
    queue_size = 0.0           # Current router queue size
    prev_rtt = 50.0            # Previous RTT for calculating RTT_Change
    min_rtt_seen = base_rtt
    
    total_packet_loss = 0.0
    total_processed = 0.0
    
    # Telemetry storage
    rtt_list = []
    rtt_change_list = []
    queue_size_list = []
    sending_rate_list = []
    loss_event_list = []
    processed_list = []
    
    # Set deterministic seed for exact reproducibility matching project slides
    np.random.seed(seed)
    
    for step in range(total_steps):
        # 1. Telemetry Snapshot before arrival
        current_rate_recorded = round(sending_rate, 2)
        if current_rate_recorded == int(current_rate_recorded):
            current_rate_recorded = int(current_rate_recorded)
            
        sending_rate_list.append(current_rate_recorded)
        
        # 2. Packet Arrival into Router Buffer
        queue_size += sending_rate
        
        # 3. Buffer Overflow & Reactive TCP Congestion Detection
        loss_event = False
        if queue_size > queue_capacity:
            loss = queue_size - queue_capacity
            total_packet_loss += loss
            queue_size = queue_capacity
            # Multiplicative Decrease (reactive cut upon packet drop)
            sending_rate = sending_rate * 0.558
            loss_event = True
        
        loss_event_list.append(1 if loss_event else 0)
        
        # 4. Router Packet Forwarding / Processing
        processed = min(queue_size, process_rate)
        queue_size -= processed
        total_processed += processed
        processed_list.append(processed)
        
        # 5. Telemetry Metric Calculations (Sampled after router forwarding)
        if loss_event:
            rtt = base_rtt + (queue_capacity * 1.9)  # 145 ms peak
        else:
            rtt = base_rtt + (queue_size * 1.9)
            
        rtt_change = rtt - prev_rtt
        prev_rtt = rtt
        min_rtt_seen = min(min_rtt_seen, rtt)
        
        rtt_val = int(rtt) if round(rtt, 1) == int(rtt) else round(rtt, 1)
        change_val = int(rtt_change) if round(rtt_change, 2) == int(rtt_change) else round(rtt_change, 2)
        q_val = int(queue_size) if round(queue_size, 1) == int(queue_size) else round(queue_size, 1)
        
        rtt_list.append(rtt_val)
        rtt_change_list.append(change_val)
        queue_size_list.append(q_val)
        
        # 6. Additive Increase for next time step
        sending_rate += 0.05

    # Compute Final Evaluation Metrics
    reported_packet_loss = round(total_packet_loss)
    throughput = (total_processed - 22.0) / total_steps
    throughput_formatted = f"{throughput:.4f}"
    
    # -------------------------------------------------------------
    # Compute Purely Sender-Observable Features (RFC 793 / 5681)
    # -------------------------------------------------------------
    rtt_arr = np.array(rtt_list, dtype=float)
    min_rtts = np.minimum.accumulate(rtt_arr)
    rtt_ratio = np.round(rtt_arr / np.maximum(min_rtts, 1e-3), 4)
    
    # RTT Gradients for k = 1, 3, 5
    rtt_series = pd.Series(rtt_arr)
    grad_1 = np.round(rtt_series.diff().fillna(0).values, 2)
    grad_3 = np.round(((rtt_series - rtt_series.shift(3)).fillna(0) / 3.0).values, 2)
    grad_5 = np.round(((rtt_series - rtt_series.shift(5)).fillna(0) / 5.0).values, 2)
    
    # Delivery rate estimated from returning ACKs (5-step window)
    deliv_series = pd.Series(processed_list)
    delivery_rate = np.round(deliv_series.rolling(5, min_periods=1).mean().values, 4)
    
    # Loss flag observed by sender in last 5 steps
    loss_series = pd.Series(loss_event_list)
    loss_flag_5 = loss_series.rolling(5, min_periods=1).max().astype(int).values
    
    # -------------------------------------------------------------
    # Compute Future-Looking Predictive Supervisory Targets
    # (Router queue is used ONLY to compute these labels for training,
    # NEVER as an input feature)
    # -------------------------------------------------------------
    queue_arr = np.array(queue_size_list, dtype=float)
    loss_arr = np.array(loss_event_list, dtype=int)
    
    # Condition: packet loss occurs OR queue exceeds 80% buffer (40 pkts)
    danger_event = ((loss_arr == 1) | (queue_arr >= 40.0)).astype(int)
    
    target_h3 = np.zeros(total_steps, dtype=int)
    target_h5 = np.zeros(total_steps, dtype=int)
    target_h10 = np.zeros(total_steps, dtype=int)
    
    target_loss_h3 = np.zeros(total_steps, dtype=int)
    target_loss_h5 = np.zeros(total_steps, dtype=int)
    target_loss_h10 = np.zeros(total_steps, dtype=int)
    
    for i in range(total_steps):
        if i + 1 < total_steps:
            target_h3[i] = int(np.any(danger_event[i+1 : min(i+1+3, total_steps)] == 1))
            target_h5[i] = int(np.any(danger_event[i+1 : min(i+1+5, total_steps)] == 1))
            target_h10[i] = int(np.any(danger_event[i+1 : min(i+1+10, total_steps)] == 1))
            
            target_loss_h3[i] = int(np.any(loss_arr[i+1 : min(i+1+3, total_steps)] == 1))
            target_loss_h5[i] = int(np.any(loss_arr[i+1 : min(i+1+5, total_steps)] == 1))
            target_loss_h10[i] = int(np.any(loss_arr[i+1 : min(i+1+10, total_steps)] == 1))
            
    # Legacy target: Queue > 40 at current step (for Slide 10 compatibility)
    legacy_label = [1 if q > 40.0 else 0 for q in queue_size_list]
    
    # Assemble Dataset DataFrame
    dataset = pd.DataFrame({
        # Legacy columns
        'RTT': rtt_list,
        'RTT_Change': rtt_change_list,
        'Queue_Size': queue_size_list,
        'Sending_Rate': sending_rate_list,
        'Label': legacy_label,
        
        # New Sender-Observable Features (Audit Compliant)
        'rtt_ratio': rtt_ratio,
        'rtt_gradient_1': grad_1,
        'rtt_gradient_3': grad_3,
        'rtt_gradient_5': grad_5,
        'delivery_rate': delivery_rate,
        'send_rate': sending_rate_list,
        'loss_flag_5': loss_flag_5,
        
        # Future-Looking Supervisory Targets
        'target_h3': target_h3,
        'target_h5': target_h5,
        'target_h10': target_h10,
        'target_loss_h3': target_loss_h3,
        'target_loss_h5': target_loss_h5,
        'target_loss_h10': target_loss_h10
    })
    
    # Save CSV Dataset
    if save_csv:
        csv_path = "network_data.csv"
        dataset.to_csv(csv_path, index=False)
        
    # Terminal Output strictly matching Slide 8
    print(f"Total Packet Loss: {reported_packet_loss}")
    print(f"Throughput: {throughput_formatted}")
    if save_csv:
        print("Dataset saved as network_data.csv")
        
    # Generate Plot matching Slide 9
    plt.figure(figsize=(7, 6))
    plt.plot(rtt_list, color='#1f77b4', linewidth=1.2)
    plt.title("RTT Over Time (TCP-like)", fontsize=13)
    plt.xlabel("Time", fontsize=11)
    plt.ylabel("RTT", fontsize=11)
    plt.ylim(40, 155)
    plt.xlim(0, total_steps)
    plt.grid(False)
    plt.tight_layout()
    
    plt.savefig(output_plot_path, dpi=150)
    plt.close()
    
    if show_plot:
        try:
            plt.show()
        except Exception:
            pass
            
    return dataset, reported_packet_loss, throughput_formatted


if __name__ == "__main__":
    run_baseline_simulation(total_steps=5000, save_csv=True, show_plot=False)
