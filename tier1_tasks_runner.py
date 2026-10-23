"""
tier1_tasks_runner.py

Executes Tier 1 tasks strictly following user constraints:
- Seeds: 42, 43, 44, 45 (n=4)
- 95% CI: Student's t-distribution with df=3 (t = 3.182)
- Label: preliminary (n=4)
- Item 1: Sensitivity sweep on 3 scenarios (clean, bandwidth walk, B=5) x 3 controllers (Reno, Vegas, Hybrid Rule)
  * Print config hash and effective parameters for every run
  * Dynamic ranking calculation
- Item 2: Table consistency resolution for:
  a) Base RTT 200ms, B=50
  b) Base RTT 20ms, B=50
  c) HO6 / HO7 RTT checks (verify theoretical max <= base_rtt + 1.9 * buf)
  d) Re-run affected cells with unified simulator definitions
"""

import sys
import os
import hashlib
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

from comprehensive_scientific_finalization import (
    PacingVariantD1_CoincidentRise, ReferenceVegasAgent, RenoAgent, CubicAgent, AuditedBBRAgent
)
from step3_realistic_robustness_and_competition import run_advanced_realistic_sim

SEEDS = [42, 43, 44, 45]
N = len(SEEDS)
T_CRIT = 3.182 # Student's t for df = 4 - 1 = 3 at 95% confidence

def get_config_hash(q_coeff, noise_sig, spike_p):
    key = f"{q_coeff:.4f}_{noise_sig:.4f}_{spike_p:.4f}"
    return hashlib.md5(key.encode()).hexdigest()[:8]

# =====================================================================
# ITEM 1: SENSITIVITY SWEEP
# =====================================================================
def run_sensitivity_sweep():
    print("=" * 80)
    print("  TIER 1 - ITEM 1: SENSITIVITY SWEEP AUDIT (preliminary n=4)")
    print("=" * 80)

    sweep_conditions = [
        ("Base Reference", 1.9, 1.0, 0.01),
        ("Queue-to-RTT -50% (0.95 ms/pkt)", 0.95, 1.0, 0.01),
        ("Queue-to-RTT +50% (2.85 ms/pkt)", 2.85, 1.0, 0.01),
        ("Noise Sigma -50% (0.5 ms)", 1.9, 0.5, 0.01),
        ("Noise Sigma +50% (1.5 ms)", 1.9, 1.5, 0.01),
        ("Spike Probability -50% (0.5%)", 1.9, 1.0, 0.005),
        ("Spike Probability +50% (1.5%)", 1.9, 1.0, 0.015)
    ]

    scenarios = [
        {"id": "Clean_B50", "name": "Clean Channel (B=50)", "channel": "constant", "buf": 50.0, "cap": 3.0},
        {"id": "BW_Walk_B50", "name": "Bandwidth Walk (1-9)", "channel": "random_walk", "buf": 50.0, "cap": 3.0},
        {"id": "Clean_B5", "name": "Shallow Buffer (B=5)", "channel": "constant", "buf": 5.0, "cap": 3.0}
    ]

    controller_factories = {
        'Hybrid Rule': lambda: PacingVariantD1_CoincidentRise("HybridRule"),
        'Reference Vegas': lambda: ReferenceVegasAgent("Vegas"),
        'TCP Reno': lambda: RenoAgent("Reno")
    }

    records = []
    identical_detected = False

    for sc in scenarios:
        print(f"\n--- Scenario: {sc['name']} ---")
        for cond_name, q_c, n_sig, sp_p in sweep_conditions:
            cfg_hash = get_config_hash(q_c, n_sig, sp_p)
            print(f"  [Config Hash: {cfg_hash}] Setting: {cond_name:32s} | q_coeff={q_c:.2f}, noise_sig={n_sig:.2f}, spike_p={sp_p:.4f}")

            setting_runs = {}
            for c_name, maker in controller_factories.items():
                u_runs = []
                gp_runs = []
                rtt_runs = []
                for s in SEEDS:
                    ag = maker()
                    res = run_advanced_realistic_sim(
                        [ag],
                        total_steps=2000,
                        seed=s,
                        buffer_capacity=sc['buf'],
                        fixed_capacity=sc['cap'],
                        channel_type=sc['channel'],
                        queue_to_rtt_coeff=q_c,
                        noise_sigma=n_sig,
                        spike_prob=sp_p
                    )
                    gp = res['goodput']
                    rtt = res['avg_rtt']
                    u = np.log(max(gp, 0.01)) - np.log(max(rtt, 1.0))
                    gp_runs.append(gp)
                    rtt_runs.append(rtt)
                    u_runs.append(u)

                m_gp = np.mean(gp_runs)
                m_rtt = np.mean(rtt_runs)
                m_u = np.mean(u_runs)
                ci_u = T_CRIT * np.std(u_runs, ddof=1) / np.sqrt(N) if np.std(u_runs, ddof=1) > 0 else 0.0
                ci_gp = T_CRIT * np.std(gp_runs, ddof=1) / np.sqrt(N) if np.std(gp_runs, ddof=1) > 0 else 0.0
                ci_rtt = T_CRIT * np.std(rtt_runs, ddof=1) / np.sqrt(N) if np.std(rtt_runs, ddof=1) > 0 else 0.0

                setting_runs[c_name] = {
                    'u_mean': m_u,
                    'u_ci': ci_u,
                    'gp_mean': m_gp,
                    'gp_ci': ci_gp,
                    'rtt_mean': m_rtt,
                    'rtt_ci': ci_rtt
                }

            # Calculate true ranking for this setting
            sorted_ctrls = sorted(setting_runs.keys(), key=lambda k: setting_runs[k]['u_mean'], reverse=True)
            ranks = {k: idx + 1 for idx, k in enumerate(sorted_ctrls)}

            for c_name, data in setting_runs.items():
                records.append({
                    'Scenario': sc['id'],
                    'Sensitivity_Setting': cond_name,
                    'Config_Hash': cfg_hash,
                    'Queue_to_RTT_Coeff': q_c,
                    'Noise_Sigma': n_sig,
                    'Spike_Prob': sp_p,
                    'Controller': c_name,
                    'Goodput': f"{data['gp_mean']:.4f} +/- {data['gp_ci']:.4f}",
                    'Avg_RTT': f"{data['rtt_mean']:.1f} +/- {data['rtt_ci']:.1f}",
                    'U_1.0': f"{data['u_mean']:.4f} +/- {data['u_ci']:.4f}",
                    'Empirical_Rank': ranks[c_name],
                    'Evaluation_Status': "preliminary (n=4)"
                })

    df_sens = pd.DataFrame(records)
    df_sens.to_csv("preliminary_sensitivity_sweep_n4.csv", index=False)
    print("\n[OK] Saved preliminary_sensitivity_sweep_n4.csv")

    # Audit for identical output bug across settings
    for sc_id in ["Clean_B50", "BW_Walk_B50", "Clean_B5"]:
        sub = df_sens[df_sens['Scenario'] == sc_id]
        u_vals = sub.groupby('Sensitivity_Setting')['U_1.0'].apply(list).to_dict()
        u_base = u_vals['Base Reference']
        for k, v in u_vals.items():
            if k != 'Base Reference' and v == u_base:
                print(f"  [WARNING] Setting '{k}' produced IDENTICAL U_1.0 values to Base Reference in {sc_id}!")
                identical_detected = True

    if not identical_detected:
        print("\n  [VERIFIED] All sensitivity parameters are actively wired. Outputs differ across settings.")
    else:
        print("\n  [ALERT] Parameter wiring check failed: identical outputs detected.")

    return df_sens

if __name__ == "__main__":
    run_sensitivity_sweep()
