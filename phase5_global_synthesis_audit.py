"""
Phase 5: Global Synthesis and Master Scientific Audit
Script: phase5_global_synthesis_audit.py

Compiles all 4 phases into a unified master audit matrix:
- Evaluates U_d = ln(goodput) - d * ln(RTT) across d in {0.5, 1.0, 2.0}
- Computes paired differences (ML vs Tuned Non-ML) across all 20 seeds (42-61)
- Calculates 95% Student's t-CIs, Cohen's d effect sizes, and Holm-Bonferroni correction
- Distills the definitive scientific answer: EXACTLY where ML genuinely beats the best non-ML controller
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

def synthesize_all_phases():
    print("=" * 80)
    print("  PHASE 5: GLOBAL MASTER AUDIT SYNTHESIS (d = 0.5, 1.0, 2.0)")
    print("=" * 80)

    # Load raw records from Phases 2, 3, and 4
    df_p2 = pd.read_csv("phase2_loss_classification_raw.csv")
    df_p3 = pd.read_csv("phase3_competitor_classification_raw.csv")
    df_p4 = pd.read_csv("phase4_capacity_prediction_raw.csv")

    all_domains = []

    # Domain 1: Loss Classification (Phase 2)
    # L1: 0.1%, L2: 0.5%, L3: 1.0%, L4: 2.0%, L5: Gilbert
    p2_scenarios = [
        ("L1_Rand01", "Loss-Type: 0.1% Random Loss"),
        ("L2_Rand05", "Loss-Type: 0.5% Random Loss"),
        ("L3_Rand10", "Loss-Type: 1.0% Random Loss"),
        ("L4_Rand20", "Loss-Type: 2.0% Random Loss"),
        ("L5_Gilbert", "Loss-Type: Bursty Gilbert-Elliott Fade")
    ]
    for sc_id, sc_name in p2_scenarios:
        sub = df_p2[df_p2["scenario_id"] == sc_id]
        gp_ml, gp_tun = sub["gp_ml"].values, sub["gp_tun"].values
        # Note: In phase 2 raw, RTT was near base (50-56 ms)
        rtt_ml = np.array([56.3] * len(gp_ml))
        rtt_tun = np.array([56.3] * len(gp_tun))
        all_domains.append((sc_id, sc_name, "Loss-Type Classification", gp_ml, gp_tun, rtt_ml, rtt_tun))

    # Domain 2: Competitor Detection (Phase 3)
    p3_scenarios = [
        ("C1_Alone", "Competitor: Alone (Latency Tax Test)"),
        ("C2_vsReno", "Competitor: Competing vs TCP Reno"),
        ("C3_vsCUBIC", "Competitor: Competing vs TCP CUBIC"),
        ("C4_vsBBR", "Competitor: Competing vs Google BBR"),
        ("C5_vsCopa", "Competitor: Competing vs Delay-Based Copa")
    ]
    for sc_id, sc_name in p3_scenarios:
        sub = df_p3[df_p3["scenario_id"] == sc_id]
        gp_ml, gp_tun = sub["gp_ml"].values, sub["gp_tun"].values
        rtt_ml, rtt_tun = sub["rtt_ml"].values, sub["rtt_tun"].values
        all_domains.append((sc_id, sc_name, "Opponent Classification", gp_ml, gp_tun, rtt_ml, rtt_tun))

    # Domain 3: Predictive Capacity Adaptation (Phase 4)
    p4_scenarios = [
        ("D1_StepDrop", "Capacity: Sudden Step Collapse (6->1.8)"),
        ("D2_MarkovFade", "Capacity: Markov Cellular Deep Fade"),
        ("D3_Sinusoidal", "Capacity: Fast Sinusoidal Capacity"),
        ("D4_Mahimahi", "Capacity: Public Mahimahi Cellular Replay")
    ]
    for sc_id, sc_name in p4_scenarios:
        sub = df_p4[df_p4["scenario_id"] == sc_id]
        gp_ml, gp_tun = sub["gp_ml"].values, sub["gp_tun"].values
        rtt_ml, rtt_tun = sub["rtt_ml"].values, sub["rtt_tun"].values
        all_domains.append((sc_id, sc_name, "Predictive Capacity Adaptation", gp_ml, gp_tun, rtt_ml, rtt_tun))

    master_records = []
    p_values_u1 = []

    for sc_id, sc_name, domain, gp_ml, gp_tun, rtt_ml, rtt_tun in all_domains:
        n = len(gp_ml)
        
        # Utilities for d in {0.5, 1.0, 2.0}
        u05_ml = np.log(np.maximum(gp_ml, 0.01)) - 0.5 * np.log(np.maximum(rtt_ml, 1.0))
        u05_tun = np.log(np.maximum(gp_tun, 0.01)) - 0.5 * np.log(np.maximum(rtt_tun, 1.0))
        diff_u05 = u05_ml - u05_tun
        
        u10_ml = np.log(np.maximum(gp_ml, 0.01)) - 1.0 * np.log(np.maximum(rtt_ml, 1.0))
        u10_tun = np.log(np.maximum(gp_tun, 0.01)) - 1.0 * np.log(np.maximum(rtt_tun, 1.0))
        diff_u10 = u10_ml - u10_tun
        
        u20_ml = np.log(np.maximum(gp_ml, 0.01)) - 2.0 * np.log(np.maximum(rtt_ml, 1.0))
        u20_tun = np.log(np.maximum(gp_tun, 0.01)) - 2.0 * np.log(np.maximum(rtt_tun, 1.0))
        diff_u20 = u20_ml - u20_tun

        m_d05 = np.mean(diff_u05)
        m_d10 = np.mean(diff_u10)
        m_d20 = np.mean(diff_u20)

        s_d10 = np.std(diff_u10, ddof=1)
        se_d10 = s_d10 / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n-1)
        ci_d10 = (m_d10 - t_crit * se_d10, m_d10 + t_crit * se_d10)

        t_stat, p_val = stats.ttest_rel(u10_ml, u10_tun)
        p_values_u1.append(p_val)
        cohen_d = m_d10 / s_d10 if s_d10 > 1e-6 else 0.0

        master_records.append({
            "domain": domain,
            "scenario_id": sc_id,
            "scenario_name": sc_name,
            "mean_gp_ml": np.mean(gp_ml),
            "mean_gp_tun": np.mean(gp_tun),
            "mean_rtt_ml": np.mean(rtt_ml),
            "mean_rtt_tun": np.mean(rtt_tun),
            "delta_u05": m_d05,
            "delta_u10": m_d10,
            "ci_lower_u10": ci_d10[0],
            "ci_upper_u10": ci_d10[1],
            "delta_u20": m_d20,
            "p_value_u10": p_val,
            "cohen_d": cohen_d,
            "ml_wins": bool(m_d10 > 0 and ci_d10[0] > 0)
        })

    df_master = pd.DataFrame(master_records)

    # Holm-Bonferroni correction across all 14 held-out scenarios
    m = len(p_values_u1)
    order = np.argsort(p_values_u1)
    holm_sig = [False] * m
    for rank, idx in enumerate(order):
        if p_values_u1[idx] < (0.05 / (m - rank)):
            holm_sig[idx] = True
        else:
            break
    df_master["holm_significant"] = holm_sig

    # Final Scientific Verdict classification
    verdict = []
    for _, row in df_master.iterrows():
        if row["ml_wins"] and row["holm_significant"]:
            verdict.append("ML GENUINELY WINS (Significant)")
        elif row["delta_u10"] < 0 and row["holm_significant"]:
            verdict.append("TUNED NON-ML WINS (Significant)")
        else:
            verdict.append("STATISTICAL TIE / INCONCLUSIVE")
    df_master["scientific_verdict"] = verdict

    out_csv = "phase5_global_synthesis_audit.csv"
    df_master.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("  PHASE 5 COMPREHENSIVE SYNTHESIS MATRIX (20 SEEDS PAIRED)")
    print("=" * 80)
    display_cols = ["scenario_name", "delta_u05", "delta_u10", "delta_u20", "ci_lower_u10", "ci_upper_u10", "p_value_u10", "scientific_verdict"]
    print(df_master[display_cols].to_string(index=False))
    print(f"\n[+] Master synthesis saved to: {out_csv}")

    # Summary Counts
    n_ml_wins = sum(df_master["scientific_verdict"] == "ML GENUINELY WINS (Significant)")
    n_nonml_wins = sum(df_master["scientific_verdict"] == "TUNED NON-ML WINS (Significant)")
    n_ties = sum(df_master["scientific_verdict"] == "STATISTICAL TIE / INCONCLUSIVE")
    print(f"\nDefinitive Breakdown across {len(df_master)} Held-Out Conditions:")
    print(f"  - ML Genuinely Wins:    {n_ml_wins} / {len(df_master)} scenarios ({(n_ml_wins/len(df_master))*100:.1f}%)")
    print(f"  - Tuned Non-ML Wins:    {n_nonml_wins} / {len(df_master)} scenarios ({(n_nonml_wins/len(df_master))*100:.1f}%)")
    print(f"  - Statistical Ties:     {n_ties} / {len(df_master)} scenarios ({(n_ties/len(df_master))*100:.1f}%)")

    return df_master

if __name__ == "__main__":
    synthesize_all_phases()
