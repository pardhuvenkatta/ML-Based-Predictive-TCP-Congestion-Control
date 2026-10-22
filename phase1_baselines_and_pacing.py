"""
ML-Based Predictive TCP Congestion Control
Script: phase1_baselines_and_pacing.py

Phase 1: Baselines and Pacing Evaluation
1. Multi-flow BBR fairness sanity check (audited BBR vs. Vegas on S7 & S8).
2. Comparison of 3 No-Model Pacing Variants across all scenarios (20 seeds, 95% CI):
   - Variant A: EWMA delivery rate
   - Variant B: Standard Max-Filter (20 steps)
   - Variant C: Max-Filter with Fast Expiry (shorter 10-step window + reset on delay inflation)
3. Selection of the best no-model baseline using TRAIN scenarios only.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np


class ProtocolAgent:
    def __init__(self, name):
        self.name = name
        self.rate = 1.05
        self.min_rtt = 50.0
        self.rtt_history = [50.0] * 6
        self.recent_losses = [0] * 6
        self.est_delivery_rate = 1.0
        self.deliv_history = [1.0] * 20
        self.throttle_count = 0

    def get_min_rate(self):
        return max(0.02, 0.20 * self.est_delivery_rate)

    def observe_ack(self, rtt, loss_event, acked_proc):
        self.min_rtt = min(self.min_rtt, rtt)
        self.rtt_history.append(rtt)
        if len(self.rtt_history) > 20: self.rtt_history.pop(0)
        self.recent_losses.append(1 if loss_event else 0)
        if len(self.recent_losses) > 5: self.recent_losses.pop(0)
        self.est_delivery_rate = 0.85 * self.est_delivery_rate + 0.15 * acked_proc
        self.deliv_history.append(acked_proc)
        if len(self.deliv_history) > 20: self.deliv_history.pop(0)


# =====================================================================
# NO-MODEL PACING VARIANTS
# =====================================================================
class NoModelEWMA(ProtocolAgent):
    """Variant A: Classic EWMA delivery rate pacing."""
    def update_rate(self, loss_event):
        min_rate = self.get_min_rate()
        if loss_event:
            self.rate = max(min_rate, self.rate * 0.558)
            return
        curr_rtt = self.rtt_history[-1]
        rtt_ratio = curr_rtt / max(self.min_rtt, 1e-3)
        grad_1 = curr_rtt - self.rtt_history[-2]
        if grad_1 <= 0 and rtt_ratio < 1.12:
            self.rate += 0.05
        else:
            self.rate += (self.est_delivery_rate - self.rate) * 0.14


class NoModelMaxFilter(ProtocolAgent):
    """Variant B: Standard Max-Filter (20-step rolling window)."""
    def update_rate(self, loss_event):
        min_rate = self.get_min_rate()
        if loss_event:
            self.rate = max(min_rate, self.rate * 0.558)
            return
        curr_rtt = self.rtt_history[-1]
        rtt_ratio = curr_rtt / max(self.min_rtt, 1e-3)
        grad_1 = curr_rtt - self.rtt_history[-2]
        target = max(self.deliv_history) if self.deliv_history else self.est_delivery_rate
        if grad_1 <= 0 and rtt_ratio < 1.12:
            self.rate += 0.05
        else:
            self.rate += (target - self.rate) * 0.14


class NoModelMaxFilterFastExpiry(ProtocolAgent):
    """
    Variant C: Max-Filter with Fast Expiry.
    - Shorter rolling window (10 steps).
    - If rtt_ratio > 1.15 and grad_1 > 0 (queue forming) or on loss:
      flushes old high samples to prevent holding stale peak capacity!
    """
    def __init__(self, name):
        super().__init__(name)
        self.fast_history = [1.0] * 10

    def observe_ack(self, rtt, loss_event, acked_proc):
        super().observe_ack(rtt, loss_event, acked_proc)
        curr_rtt = self.rtt_history[-1]
        rtt_ratio = curr_rtt / max(self.min_rtt, 1e-3)
        grad_1 = curr_rtt - self.rtt_history[-2]
        
        # Fast expiry / reset when link contracts or congestion starts
        if (rtt_ratio > 1.15 and grad_1 > 0) or loss_event:
            self.fast_history = [acked_proc]
        else:
            self.fast_history.append(acked_proc)
            if len(self.fast_history) > 10:
                self.fast_history.pop(0)

    def update_rate(self, loss_event):
        min_rate = self.get_min_rate()
        if loss_event:
            self.rate = max(min_rate, self.rate * 0.558)
            return
        curr_rtt = self.rtt_history[-1]
        rtt_ratio = curr_rtt / max(self.min_rtt, 1e-3)
        grad_1 = curr_rtt - self.rtt_history[-2]
        target = max(self.fast_history) if self.fast_history else self.est_delivery_rate
        if grad_1 <= 0 and rtt_ratio < 1.12:
            self.rate += 0.05
        else:
            self.rate += (target - self.rate) * 0.14


# =====================================================================
# SIMULATION HARNESS
# =====================================================================
def run_sim(agents, total_steps=3000, seed=42, buffer_capacity=50.0, base_rtt=50.0,
            random_loss_rate=0.0, rtt_noise_std=1.0, bandwidth_pattern="constant",
            fixed_capacity=3.0, stagger_flows=False):
    rng = np.random.RandomState(seed)
    n_flows = len(agents)
    queue = 0.0
    total_loss = 0.0
    total_delivered = [0.0] * n_flows
    q_log, rtt_log = [], []
    
    start_times = [0] * n_flows
    if stagger_flows and n_flows > 1:
        for i in range(1, n_flows):
            start_times[i] = int(rng.uniform(50, 400))
            
    capacity_trace = np.zeros(total_steps)
    if bandwidth_pattern == "constant":
        capacity_trace[:] = fixed_capacity
    elif bandwidth_pattern == "random_walk":
        c = fixed_capacity
        for t in range(total_steps):
            c = np.clip(c + rng.normal(0, 0.10), 1.0, 9.0)
            capacity_trace[t] = c

    for t in range(total_steps):
        link_cap = capacity_trace[t]
        active_indices = [i for i in range(n_flows) if t >= start_times[i]]
        rates = [max(agents[i].get_min_rate(), agents[i].rate) if i in active_indices else 0.0 for i in range(n_flows)]
        total_send = sum(rates)
        queue += total_send
        
        loss_dropped = 0.0
        flow_losses = [0.0] * n_flows
        if queue > buffer_capacity:
            loss_dropped = queue - buffer_capacity
            total_loss += loss_dropped
            queue = buffer_capacity
            if total_send > 0:
                for i in active_indices:
                    flow_losses[i] = loss_dropped * (rates[i] / total_send)
                    
        total_proc = min(queue, link_cap)
        queue -= total_proc
        flow_proc = [0.0] * n_flows
        if total_send > 0 and total_proc > 0:
            for i in active_indices:
                flow_proc[i] = total_proc * (rates[i] / total_send)
                
        for i in active_indices:
            if random_loss_rate > 0.0 and rng.uniform(0, 1) < random_loss_rate:
                channel_dropped = min(flow_proc[i], 1.0)
                flow_proc[i] -= channel_dropped
                flow_losses[i] += channel_dropped
                total_loss += channel_dropped
            total_delivered[i] += flow_proc[i]
            
        true_rtt = base_rtt + (buffer_capacity * 1.9) if loss_dropped > 0 else base_rtt + (queue * 1.9)
        noise = rng.normal(0, rtt_noise_std)
        if rng.uniform(0, 1) < 0.01: noise += rng.uniform(10.0, 35.0)
        observed_rtt = max(base_rtt, true_rtt + noise)
        
        rtt_log.append(true_rtt)
        q_log.append(queue)
        
        for i in active_indices:
            agents[i].observe_ack(observed_rtt, flow_losses[i] > 0, flow_proc[i])
            agents[i].update_rate(flow_losses[i] > 0)

    agg_tp = sum(total_delivered) / total_steps
    agg_gp = max(0.0, (sum(total_delivered) - total_loss * 7.9) / total_steps) if total_loss > 0 else agg_tp
    flow_gps = [max(0.0, total_delivered[i] / max(1, (total_steps - start_times[i]))) for i in range(n_flows)]
    
    jfi = (sum(flow_gps)**2) / (n_flows * sum(g**2 for g in flow_gps)) if (n_flows > 1 and sum(flow_gps) > 0) else 1.0
    return {
        'loss': total_loss, 'goodput': agg_gp, 'avg_queue': np.mean(q_log),
        'avg_rtt': np.mean(rtt_log), 'jfi': jfi, 'mean_cap': np.mean(capacity_trace)
    }


def main():
    print("=" * 80)
    print("  PHASE 1: BASELINES AND PACING EVALUATION")
    print("=" * 80)
    
    seeds = list(range(42, 62))
    N = len(seeds)
    
    # -------------------------------------------------------------
    # 1. Multi-Flow BBR Fairness Sanity Check
    # -------------------------------------------------------------
    print("\n--- 1. BBR MULTI-FLOW FAIRNESS SANITY CHECK ---")
    from advanced_audit_and_regression import AuditedBBRAgent, ReferenceVegasAgent
    
    for flows, cap, name in [(2, 4.0, "2 Flows (S7)"), (8, 6.0, "8 Flows (S8)")]:
        bbr_jfis, vg_jfis = [], []
        bbr_losses, vg_losses = [], []
        for s in seeds:
            res_bbr = run_sim([AuditedBBRAgent(f"BBR_{i}") for i in range(flows)], total_steps=3000, seed=s,
                              fixed_capacity=cap, buffer_capacity=50.0, stagger_flows=True)
            res_vg = run_sim([ReferenceVegasAgent(f"VG_{i}") for i in range(flows)], total_steps=3000, seed=s,
                             fixed_capacity=cap, buffer_capacity=50.0, stagger_flows=True)
            bbr_jfis.append(res_bbr['jfi'])
            vg_jfis.append(res_vg['jfi'])
            bbr_losses.append(res_bbr['loss'])
            vg_losses.append(res_vg['loss'])
            
        print(f"  [{name}]")
        print(f"    Audited BBR: JFI = {np.mean(bbr_jfis):.3f} +/- {1.96*np.std(bbr_jfis)/np.sqrt(N):.3f} | Loss = {np.mean(bbr_losses):.1f}")
        print(f"    Vegas:       JFI = {np.mean(vg_jfis):.3f} +/- {1.96*np.std(vg_jfis)/np.sqrt(N):.3f} | Loss = {np.mean(vg_losses):.1f}")

    # -------------------------------------------------------------
    # 2. Compare 3 No-Model Pacing Variants across All Scenarios
    # -------------------------------------------------------------
    print("\n--- 2. COMPARISON OF 3 NO-MODEL PACING VARIANTS ---")
    scenarios = [
        # TRAIN scenarios
        {"id": "TR1", "family": "TRAIN", "name": "Clean Baseline (C=3.0, B=50)", "cap": 3.0, "buf": 50.0, "loss": 0.0, "pat": "constant", "flows": 1},
        {"id": "TR2", "family": "TRAIN", "name": "Moderate Link (C=5.0, B=50)", "cap": 5.0, "buf": 50.0, "loss": 0.0, "pat": "constant", "flows": 1},
        # HELD-OUT scenarios
        {"id": "HO1", "family": "HELD-OUT", "name": "Variable BW Walk (1-9)", "cap": 3.0, "buf": 50.0, "loss": 0.0, "pat": "random_walk", "flows": 1},
        {"id": "HO2", "family": "HELD-OUT", "name": "Random Loss 0.1%", "cap": 3.0, "buf": 50.0, "loss": 0.001, "pat": "constant", "flows": 1},
        {"id": "HO3", "family": "HELD-OUT", "name": "Random Loss 2.0%", "cap": 3.0, "buf": 50.0, "loss": 0.020, "pat": "constant", "flows": 1},
        {"id": "HO4", "family": "HELD-OUT", "name": "Shallow Buffer B=5", "cap": 3.0, "buf": 5.0, "loss": 0.0, "pat": "constant", "flows": 1},
        {"id": "HO5", "family": "HELD-OUT", "name": "Ultra-Shallow B=3", "cap": 3.0, "buf": 3.0, "loss": 0.0, "pat": "constant", "flows": 1},
        {"id": "HO6", "family": "HELD-OUT", "name": "Multi-Flow 2 Flows", "cap": 4.0, "buf": 50.0, "loss": 0.0, "pat": "constant", "flows": 2},
        {"id": "HO7", "family": "HELD-OUT", "name": "Multi-Flow 8 Flows", "cap": 6.0, "buf": 50.0, "loss": 0.0, "pat": "constant", "flows": 8}
    ]
    
    variant_makers = {
        'EWMA': lambda: NoModelEWMA("EWMA"),
        'MaxFilter_20': lambda: NoModelMaxFilter("MaxFilter"),
        'MaxFilter_FastExpiry': lambda: NoModelMaxFilterFastExpiry("FastExpiry")
    }
    
    rows = []
    
    for sc in scenarios:
        for v_name, v_fn in variant_makers.items():
            runs = []
            for s in seeds:
                agents = [v_fn() for _ in range(sc['flows'])]
                res = run_sim(agents, total_steps=3000, seed=s, buffer_capacity=sc['buf'],
                              random_loss_rate=sc['loss'], bandwidth_pattern=sc['pat'],
                              fixed_capacity=sc['cap'], stagger_flows=(sc['flows'] > 1))
                runs.append(res)
            gp = np.mean([r['goodput'] for r in runs])
            gp_ci = 1.96 * np.std([r['goodput'] for r in runs]) / np.sqrt(N)
            loss = np.mean([r['loss'] for r in runs])
            rtt = np.mean([r['avg_rtt'] for r in runs])
            rtt_ci = 1.96 * np.std([r['avg_rtt'] for r in runs]) / np.sqrt(N)
            q = np.mean([r['avg_queue'] for r in runs])
            jfi = np.mean([r['jfi'] for r in runs])
            
            rows.append({
                'Family': sc['family'],
                'Scenario': sc['name'],
                'Variant': v_name,
                'Goodput': f"{gp:.4f} +/- {gp_ci:.4f}",
                'gp_val': gp,
                'Loss': f"{loss:.1f}",
                'loss_val': loss,
                'Avg RTT (ms)': f"{rtt:.1f} +/- {rtt_ci:.1f}",
                'rtt_val': rtt,
                'Avg Queue': f"{q:.1f}",
                'JFI': f"{jfi:.3f}" if sc['flows'] > 1 else "-"
            })
            
    df_pacing = pd.DataFrame(rows)
    print("\n" + df_pacing[['Family', 'Scenario', 'Variant', 'Goodput', 'Loss', 'Avg RTT (ms)', 'Avg Queue', 'JFI']].to_string(index=False))
    
    # Selection of Best No-Model Baseline using TRAIN scenarios only
    df_train = df_pacing[df_pacing['Family'] == 'TRAIN']
    print("\n--- 3. TRAIN SELECTION OF BEST NO-MODEL BASELINE ---")
    for v in variant_makers.keys():
        sub = df_train[df_train['Variant'] == v]
        print(f"  {v}: Mean Goodput on TRAIN = {sub['gp_val'].mean():.4f} pkts/step | Mean Loss = {sub['loss_val'].mean():.1f} | Mean RTT = {sub['rtt_val'].mean():.1f} ms")
        
    df_pacing.to_csv("phase1_pacing_comparison.csv", index=False)
    print("\n[OK] Phase 1 results saved to 'phase1_pacing_comparison.csv'")


if __name__ == "__main__":
    main()
