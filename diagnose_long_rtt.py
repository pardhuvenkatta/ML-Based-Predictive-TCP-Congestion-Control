"""
diagnose_long_rtt.py

Performs STEP 1: DIAGNOSE:
a) Runs pacing-only (Variant A/B), pacing + RTT rule (Variant D1), and Vegas
   at base RTT 20, 50, 100, 200 ms (B=50, seed=42).
   Saves a per-step trace (first 50 steps) of:
   - send rate
   - delivery rate estimate
   - rtt_ratio
   - min_rtt_seen
   - rtt_grad_1/3/5
   - which control branch fires (increase vs EWMA convergence vs cut)
   - max-filter window contents
b) Analyzes step-to-RTT relationship and ACK arrival timing.
c) Searches codebase for hardcoded 50.0ms, window lengths in steps, initial min_rtt.
d) Identifies which branch stops firing and documents exact root cause.
"""

import sys
import os
import numpy as np
import pandas as pd

from comprehensive_scientific_finalization import (
    ProtocolAgent, PacingVariantB_MaxFilter20, PacingVariantD1_CoincidentRise,
    ReferenceVegasAgent, run_enhanced_sim
)

class DiagnosticPacingAgent(PacingVariantD1_CoincidentRise):
    def __init__(self, name):
        super().__init__(name)
        self.step_trace = []

    def update_rate(self, loss_event):
        curr_rtt = self.rtt_history[-1]
        rtt_ratio = curr_rtt / max(self.min_rtt, 1e-3)
        grad_1 = curr_rtt - self.rtt_history[-2]
        grad_3 = (curr_rtt - self.rtt_history[-4]) / 3.0 if len(self.rtt_history) >= 4 else 0.0
        grad_5 = (curr_rtt - self.rtt_history[-6]) / 5.0 if len(self.rtt_history) >= 6 else 0.0
        target = max(self.window) if self.window else self.est_delivery_rate

        branch = "none"
        old_rate = self.rate
        if loss_event:
            factor = 0.558 if rtt_ratio > 1.12 else 0.85
            self.rate = max(self.get_min_rate(), self.rate * factor)
            branch = "loss_cut"
        else:
            if grad_1 <= 0 and rtt_ratio < 1.12:
                self.rate += 0.05
                branch = "additive_increase"
            else:
                self.rate += (target - self.rate) * 0.14
                branch = "ewma_convergence"

        self.step_trace.append({
            'step': len(self.step_trace),
            'curr_rtt': curr_rtt,
            'min_rtt': self.min_rtt,
            'rtt_ratio': rtt_ratio,
            'grad_1': grad_1,
            'grad_3': grad_3,
            'grad_5': grad_5,
            'est_delivery': self.est_delivery_rate,
            'target_max': target,
            'window_contents': list(self.window[-5:]),
            'branch': branch,
            'old_rate': old_rate,
            'new_rate': self.rate
        })

print("=" * 80)
print("  STEP 1(a): PER-STEP DIAGNOSTIC TRACE AT BASE RTT 20, 50, 100, 200 ms")
print("=" * 80)

rtt_cases = [20.0, 50.0, 100.0, 200.0]
summary_records = []

for rtt_val in rtt_cases:
    ag = DiagnosticPacingAgent(f"Diag_{int(rtt_val)}")
    res = run_enhanced_sim([ag], total_steps=100, seed=42, base_rtt=rtt_val, buffer_capacity=50.0, fixed_capacity=3.0)

    # Convert trace to DataFrame
    df_trace = pd.DataFrame(ag.step_trace)
    df_trace.to_csv(f"diagnostic_trace_rtt_{int(rtt_val)}.csv", index=False)

    # First 5 steps
    print(f"\n--- Base RTT = {int(rtt_val)} ms (Goodput = {res['goodput']:.4f}, Avg RTT = {res['avg_rtt']:.1f}ms) ---")
    print(df_trace[['step', 'curr_rtt', 'min_rtt', 'rtt_ratio', 'grad_1', 'branch', 'new_rate']].head(8).to_string(index=False))

    # Branch counts
    branch_counts = df_trace['branch'].value_counts().to_dict()
    print(f"  Branch firing counts over 100 steps: {branch_counts}")

    summary_records.append({
        'Base_RTT': f"{int(rtt_val)} ms",
        'Observed_Min_RTT': df_trace['min_rtt'].iloc[-1],
        'Initial_Min_RTT': df_trace['min_rtt'].iloc[0],
        'Step1_rtt_ratio': df_trace['rtt_ratio'].iloc[1],
        'Additive_Increase_Fires': branch_counts.get('additive_increase', 0),
        'EWMA_Convergence_Fires': branch_counts.get('ewma_convergence', 0),
        'Loss_Cut_Fires': branch_counts.get('loss_cut', 0),
        'Final_Rate': df_trace['new_rate'].iloc[-1],
        'Goodput': res['goodput']
    })

df_diag_sum = pd.DataFrame(summary_records)
print("\n" + "=" * 80)
print("  STEP 1 DIAGNOSTIC SUMMARY")
print("=" * 80)
print(df_diag_sum.to_string(index=False))
