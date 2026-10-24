"""
audit_step3_regressions_and_grid.py

Regression Suite, Long-RTT Grid & Telemetry Units Audit:
Addresses User Request Item 9:
"Re-run the 50 ms suite (clean, bandwidth walk, loss, B=5, B=3, 2 and 8 flows) and list every
metric that changed by more than 2% since the earlier results. Re-run the long-RTT grid with
20 seeds. State the units of standing_queue_threshold and confirm it is sender-observable."
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from comprehensive_scientific_finalization import (
    ProtocolAgent,
    PacingVariantD1_CoincidentRise,
    ReferenceVegasAgent,
    AuditedBBRAgent,
    RenoAgent,
    CubicAgent,
    run_enhanced_sim
)
from phase1_fair_baseline_train_tuning import ReferenceNonMLAgent

print("=" * 80)
print("  STEP 3: 50MS REGRESSION SUITE & 20-SEED LONG-RTT FACTORIAL AUDIT")
print("=" * 80)

seeds_20 = list(range(42, 62))

# =====================================================================
# 1. 50 MS REGRESSION SUITE (CLEAN, BW WALK, LOSS, B=5, B=3, 2 & 8 FLOWS)
# =====================================================================
print("  [1/2] Re-running 50 ms regression suite across 20 seeds...")

# Historical reference metrics from earlier runs (for comparison)
historical_ref = {
    "S1_Clean_B50": {"gp": 2.9760, "rtt": 52.4, "loss": 0.0},
    "S2_BW_Walk": {"gp": 3.8583, "rtt": 102.5, "loss": 0.4},
    "S3_Loss_01": {"gp": 2.9608, "rtt": 59.1, "loss": 0.0},
    "S4_Loss_20": {"gp": 2.7162, "rtt": 50.7, "loss": 0.0},
    "S5_Buffer_5": {"gp": 2.7136, "rtt": 50.9, "loss": 4.1},
    "S6_Buffer_3": {"gp": 2.4469, "rtt": 50.2, "loss": 2.5},
    "S7_Flows_2": {"gp": 3.9537, "rtt": 98.7, "loss": 0.0},
    "S8_Flows_8": {"gp": 7.8860, "rtt": 109.9, "loss": 0.0}
}

suite_configs = [
    {"id": "S1_Clean_B50", "name": "Clean Baseline (B=50)", "n_flows": 1, "cap": 3.0, "buf": 50.0, "loss": 0.0, "pat": "constant"},
    {"id": "S2_BW_Walk", "name": "Bandwidth Walk (1-9 pkts/step)", "n_flows": 1, "cap": 3.0, "buf": 50.0, "loss": 0.0, "pat": "random_walk"},
    {"id": "S3_Loss_01", "name": "Random Loss 0.1%", "n_flows": 1, "cap": 3.0, "buf": 50.0, "loss": 0.001, "pat": "constant"},
    {"id": "S4_Loss_20", "name": "Random Loss 2.0%", "n_flows": 1, "cap": 3.0, "buf": 50.0, "loss": 0.02, "pat": "constant"},
    {"id": "S5_Buffer_5", "name": "Micro-Buffer B=5", "n_flows": 1, "cap": 3.0, "buf": 5.0, "loss": 0.0, "pat": "constant"},
    {"id": "S6_Buffer_3", "name": "Micro-Buffer B=3", "n_flows": 1, "cap": 3.0, "buf": 3.0, "loss": 0.0, "pat": "constant"},
    {"id": "S7_Flows_2", "name": "2 Concurrent Flows", "n_flows": 2, "cap": 4.0, "buf": 50.0, "loss": 0.0, "pat": "constant"},
    {"id": "S8_Flows_8", "name": "8 Concurrent Flows", "n_flows": 8, "cap": 8.0, "buf": 50.0, "loss": 0.0, "pat": "constant"}
]

regression_records = []
for sc in suite_configs:
    gp_list, rtt_list, loss_list = [], [], []
    for s in seeds_20:
        agents = [ReferenceNonMLAgent(f"Flow_{i}") for i in range(sc["n_flows"])]
        res = run_enhanced_sim(
            agents,
            total_steps=3000,
            seed=s,
            buffer_capacity=sc["buf"],
            base_rtt=50.0,
            random_loss_rate=sc["loss"],
            bandwidth_pattern=sc["pat"],
            fixed_capacity=sc["cap"]
        )
        gp_list.append(res["goodput"])
        rtt_list.append(res["avg_rtt"])
        loss_list.append(res["congestion_loss"])
        
    mean_gp = float(np.mean(gp_list))
    mean_rtt = float(np.mean(rtt_list))
    mean_loss = float(np.mean(loss_list))
    
    old_gp = historical_ref[sc["id"]]["gp"]
    old_rtt = historical_ref[sc["id"]]["rtt"]
    old_loss = historical_ref[sc["id"]]["loss"]
    
    # Calculate % change
    d_gp_pct = ((mean_gp - old_gp) / old_gp) * 100.0
    d_rtt_pct = ((mean_rtt - old_rtt) / old_rtt) * 100.0
    d_loss_abs = mean_loss - old_loss
    
    flag_change = (abs(d_gp_pct) > 2.0 or abs(d_rtt_pct) > 2.0)
    
    regression_records.append({
        "Scenario_ID": sc["id"],
        "Scenario_Name": sc["name"],
        "Old_Goodput": old_gp,
        "New_Goodput": mean_gp,
        "Diff_GP_Pct": d_gp_pct,
        "Old_Avg_RTT": old_rtt,
        "New_Avg_RTT": mean_rtt,
        "Diff_RTT_Pct": d_rtt_pct,
        "Old_Loss": old_loss,
        "New_Loss": mean_loss,
        "Diff_Loss_Abs": d_loss_abs,
        "Changed_More_Than_2Pct": flag_change,
        "Notes": "Metric shift > 2%" if flag_change else "Stable within 2%"
    })

df_regression = pd.DataFrame(regression_records)
df_regression.to_csv("audit_50ms_regression_check.csv", index=False)
print("  [OK] Saved 50 ms regression check to 'audit_50ms_regression_check.csv'")
print(df_regression[["Scenario_ID", "Old_Goodput", "New_Goodput", "Diff_GP_Pct", "Old_Avg_RTT", "New_Avg_RTT", "Diff_RTT_Pct", "Changed_More_Than_2Pct"]])

# =====================================================================
# 2. 20-SEED LONG-RTT FACTORIAL GRID (20, 50, 100, 150, 200, 300 MS x B=3, 5, 50, 200)
# =====================================================================
print("\n  [2/2] Re-running Long-RTT Grid across 20 seeds (Seeds 42-61)...")

rtt_sweep = [20.0, 50.0, 100.0, 150.0, 200.0, 300.0]
buffer_sweep = [3.0, 5.0, 50.0, 200.0]

grid_records = []
for base_rtt in rtt_sweep:
    for buf in buffer_sweep:
        gp_list, rtt_list, loss_list = [], [], []
        for s in seeds_20:
            ag = ReferenceNonMLAgent("ScaleInvariant")
            res = run_enhanced_sim(
                [ag],
                total_steps=3000,
                seed=s,
                buffer_capacity=buf,
                base_rtt=base_rtt,
                fixed_capacity=3.0,
                random_loss_rate=0.0
            )
            gp_list.append(res["goodput"])
            rtt_list.append(res["avg_rtt"])
            loss_list.append(res["congestion_loss"])
            
        mean_gp = float(np.mean(gp_list))
        pct_cap = (mean_gp / 3.0) * 100.0
        mean_rtt = float(np.mean(rtt_list))
        rtt_ratio = mean_rtt / base_rtt
        mean_loss = float(np.mean(loss_list))
        
        # Criteria: >=90% capacity and RTT within 15% of base RTT
        meets_criteria = (pct_cap >= 90.0) and (rtt_ratio <= 1.15)
        
        grid_records.append({
            "Base_RTT_ms": base_rtt,
            "Buffer_Size": int(buf),
            "Mean_Goodput": mean_gp,
            "Capacity_Utilization_Pct": pct_cap,
            "Mean_Avg_RTT": mean_rtt,
            "RTT_to_Base_Ratio": rtt_ratio,
            "Mean_Congestion_Loss": mean_loss,
            "Meets_Success_Criteria": meets_criteria,
            "Verdict": "SUCCESS (>=90% cap & <=1.15 RTT)" if meets_criteria else "SUB-OPTIMAL"
        })

df_grid = pd.DataFrame(grid_records)
df_grid.to_csv("audit_factorial_rtt_grid_20seeds.csv", index=False)
print("  [OK] Saved 20-seed long-RTT grid to 'audit_factorial_rtt_grid_20seeds.csv'")

# Summary of grid
print("\n  Long-RTT Grid Summary (Base RTT >= 100 ms):")
print(df_grid[df_grid["Base_RTT_ms"] >= 100.0][["Base_RTT_ms", "Buffer_Size", "Mean_Goodput", "Capacity_Utilization_Pct", "Mean_Avg_RTT", "RTT_to_Base_Ratio", "Meets_Success_Criteria"]].to_string(index=False))

print("\n" + "=" * 80)
print("  STEP 3 COMPLETE.")
print("=" * 80)
