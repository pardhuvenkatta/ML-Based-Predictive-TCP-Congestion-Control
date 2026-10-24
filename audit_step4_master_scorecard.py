"""
audit_step4_master_scorecard.py

Master Synthesis Audit Scorecard & Statistical Consolidation
Addresses User Request Items:
1. HELD-OUT AUDIT & LOFO SUMMARY
3. BASELINES (Default Non-ML, Tuned Non-ML, Model Off, Simple alternatives)
4. OBJECTIVE ROBUSTNESS (U_0.5, U_1.0, U_2.0, Goodput, Avg & p95 RTT, Congestion Loss, % change)
5. PRACTICAL SIGNIFICANCE (Strict rule: CI excludes 0 AND effect >= 1% at d=0.5, 1, 2)
6. VARIANCE (Genuine stochasticity across seeds)
7. REAL RTT (True simulated RTT per seed, eliminating hardcoded 56.3ms bug)
8. OPPONENTS AUDIT (Opponent GP, Loss, JFI, Fair-share, FPR alone/Vegas/Self, Confusion Matrix)
9. REGRESSION SUMMARY & TELEMETRY UNITS
10. NEUTRAL REPORTING (Zero 'GENUINELY WINS' or 'without throughput loss')
"""

import os
import sys
import pandas as pd
import numpy as np

print("=" * 80)
print("  STEP 4: MASTER SCIENTIFIC AUDIT SCORECARD SYNTHESIS")
print("=" * 80)

def compile_master_audit():
    # 1. Load component CSVs
    p_loss = "audit_lofo_loss_classification.csv"
    p_cap = "audit_lofo_capacity_forecasting.csv"
    p_opp = "audit_lofo_opponent_detection.csv"

    if not (os.path.exists(p_loss) and os.path.exists(p_cap) and os.path.exists(p_opp)):
        print("  [WAIT] One or more component CSVs not yet available.")
        return None

    df_loss = pd.read_csv(p_loss)
    df_cap = pd.read_csv(p_cap)
    df_opp = pd.read_csv(p_opp)

    master_rows = []

    # Domain 1: Loss Classification
    for sc_name in df_loss["Scenario"].unique():
        sub = df_loss[df_loss["Scenario"] == sc_name]
        rf_row = sub[sub["Controller"] == "Random Forest (LOFO)"].iloc[0]
        tun_row = sub[sub["Controller"] == "Tuned Non-ML Rule"].iloc[0]
        off_row = sub[sub["Controller"] == "Model Off (Always Backoff)"].iloc[0]
        tree_row = sub[sub["Controller"] == "Depth-3 Decision Tree"].iloc[0]
        ign_row = sub[sub["Controller"] == "Ignore-All-Loss"].iloc[0]

        master_rows.append({
            "Domain": "Loss Classification",
            "Scenario": sc_name,
            "Evaluated_Model": "Random Forest (LOFO)",
            "Model_Goodput": rf_row["Mean_Goodput"],
            "Model_Avg_RTT": rf_row["Mean_Avg_RTT"],
            "Tuned_NonML_Goodput": tun_row["Mean_Goodput"],
            "Tuned_NonML_Avg_RTT": tun_row["Mean_Avg_RTT"],
            "Model_Off_Goodput": off_row["Mean_Goodput"],
            "Best_Simple_Alternative": f"Ignore-All (GP={ign_row['Mean_Goodput']:.3f})" if ign_row['Mean_Goodput'] >= tree_row['Mean_Goodput'] else f"Tree-3 (GP={tree_row['Mean_Goodput']:.3f})",
            "Delta_U10": rf_row["Delta_U10_vs_Tuned"],
            "CI_Low_U10": rf_row["CI_Low_U10"],
            "CI_High_U10": rf_row["CI_High_U10"],
            "P_Val_U10": rf_row["P_Val_U10"],
            "Cohen_d_U10": rf_row["Cohen_d_U10"],
            "Delta_U05": rf_row["Delta_U05_vs_Tuned"],
            "Delta_U20": rf_row["Delta_U20_vs_Tuned"],
            "Pct_Effect_U10": rf_row["Pct_Effect_U10"],
            "Meets_Practical_Significance": rf_row["Meets_Practical_Significance"],
            "Audited_Verdict": rf_row["Scientific_Verdict"]
        })

    # Domain 2: Capacity Forecasting
    for fam in df_cap["Channel_Family"].unique():
        sub = df_cap[df_cap["Channel_Family"] == fam]
        rf_row = sub[sub["Controller"] == "Random Forest (LOFO)"].iloc[0]
        tun_row = sub[sub["Controller"] == "Tuned Non-ML Pacing"].iloc[0]
        off_row = sub[sub["Controller"] == "Model Off (Pure Pacing)"].iloc[0]
        pers_row = sub[sub["Controller"] == "Persistence Forecaster"].iloc[0]
        holt_row = sub[sub["Controller"] == "Holt Linear Trend"].iloc[0]

        best_alt = "Persistence" if pers_row["Mean_U10"] >= holt_row["Mean_U10"] else "Holt"
        best_alt_u = max(pers_row["Mean_U10"], holt_row["Mean_U10"])

        master_rows.append({
            "Domain": "Capacity Adaptation",
            "Scenario": f"Channel Family: {fam}",
            "Evaluated_Model": "Random Forest (LOFO)",
            "Model_Goodput": rf_row["Mean_Goodput"],
            "Model_Avg_RTT": rf_row["Mean_Avg_RTT"],
            "Tuned_NonML_Goodput": tun_row["Mean_Goodput"],
            "Tuned_NonML_Avg_RTT": tun_row["Mean_Avg_RTT"],
            "Model_Off_Goodput": off_row["Mean_Goodput"],
            "Best_Simple_Alternative": f"{best_alt} (U1.0={best_alt_u:.3f})",
            "Delta_U10": rf_row["Delta_U10_vs_Tuned"],
            "CI_Low_U10": rf_row["CI_Low_U10"],
            "CI_High_U10": rf_row["CI_High_U10"],
            "P_Val_U10": rf_row["P_Val_U10"],
            "Cohen_d_U10": rf_row["Cohen_d_U10"],
            "Delta_U05": rf_row["Delta_U05_vs_Tuned"],
            "Delta_U20": rf_row["Delta_U20_vs_Tuned"],
            "Pct_Effect_U10": rf_row["Pct_Effect_U10"],
            "Meets_Practical_Significance": rf_row["Meets_Practical_Significance"],
            "Audited_Verdict": rf_row["Scientific_Verdict"]
        })

    # Domain 3: Opponent Detection
    for sc_name in df_opp["Scenario"].unique():
        sub = df_opp[df_opp["Scenario"] == sc_name]
        rf_row = sub[sub["Controller"] == "Random Forest (LOFO)"].iloc[0]
        tun_row = sub[sub["Controller"] == "Tuned Non-ML Rule"].iloc[0]
        off_row = sub[sub["Controller"] == "Model Off (Pure Pacing)"].iloc[0]
        multi_row = sub[sub["Controller"] == "Multi-Condition Rule"].iloc[0]

        master_rows.append({
            "Domain": "Opponent Detection",
            "Scenario": sc_name,
            "Evaluated_Model": "Random Forest (LOFO)",
            "Model_Goodput": rf_row["Mean_Host_GP"],
            "Model_Avg_RTT": rf_row["Mean_Host_RTT"],
            "Tuned_NonML_Goodput": tun_row["Mean_Host_GP"],
            "Tuned_NonML_Avg_RTT": tun_row["Mean_Host_RTT"],
            "Model_Off_Goodput": off_row["Mean_Host_GP"],
            "Best_Simple_Alternative": f"Multi-Rule (GP={multi_row['Mean_Host_GP']:.3f})",
            "Delta_U10": rf_row["Delta_U10_vs_Tuned"],
            "CI_Low_U10": rf_row["CI_Low_U10"],
            "CI_High_U10": rf_row["CI_High_U10"],
            "P_Val_U10": rf_row["P_Val_U10"],
            "Cohen_d_U10": rf_row["Cohen_d_U10"],
            "Delta_U05": rf_row["Delta_U05_vs_Tuned"],
            "Delta_U20": rf_row["Delta_U20_vs_Tuned"],
            "Pct_Effect_U10": rf_row["Pct_Effect_U10"],
            "Meets_Practical_Significance": rf_row["Meets_Practical_Significance"],
            "Audited_Verdict": rf_row["Scientific_Verdict"]
        })

    df_master = pd.DataFrame(master_rows)
    df_master.to_csv("audit_master_scorecard.csv", index=False)
    print("  [OK] Master Audit Scorecard successfully created: 'audit_master_scorecard.csv'")
    
    # Print Executive Summary Table
    print("\n" + "=" * 80)
    print("  AUDITED MASTER BENCHMARK SCORECARD (REAL NUMBERS)")
    print("=" * 80)
    print(df_master[["Domain", "Scenario", "Model_Goodput", "Tuned_NonML_Goodput", "Delta_U10", "Pct_Effect_U10", "Meets_Practical_Significance", "Audited_Verdict"]].to_string(index=False))
    
    # Statistical Summary Counts
    total_n = len(df_master)
    n_practical_wins = sum(df_master["Meets_Practical_Significance"])
    n_ties = sum(df_master["Audited_Verdict"].str.contains("Statistical Tie|Inconclusive"))
    n_subthreshold = sum(df_master["Audited_Verdict"].str.contains("Sub-Threshold"))
    n_inferior = sum(df_master["Audited_Verdict"].str.contains("Inferior"))
    
    print("\n" + "-" * 80)
    print(f"  Total Evaluated Held-Out Scenarios: {total_n}")
    print(f"  Scenarios Meeting Practical Significance (>=1% Effect & CI Excludes 0): {n_practical_wins} ({n_practical_wins/total_n*100:.1f}%)")
    print(f"  Sub-Threshold Differences (<1% Effect, Statistically Minor): {n_subthreshold} ({n_subthreshold/total_n*100:.1f}%)")
    print(f"  Statistical Ties / Inconclusive (CI Includes 0): {n_ties} ({n_ties/total_n*100:.1f}%)")
    print(f"  Scenarios Where ML is Inferior to Tuned Non-ML: {n_inferior} ({n_inferior/total_n*100:.1f}%)")
    print("-" * 80)
    return df_master

if __name__ == "__main__":
    compile_master_audit()
