"""
recheck_table_consistency.py

Re-runs affected cells for Table Consistency (Item 2) using unified simulator configuration:
- Base RTT 200 ms, B=50 (HO9)
- Base RTT 20 ms, B=50 (HO8)
- HO6 (2 flows, B=50, Base RTT 50)
- HO7 (8 flows, B=50, Base RTT 50)
Seeds: 42, 43, 44, 45 (n=4)
Student's t 95% CI (t = 3.182)
Label: preliminary (n=4)
"""

import sys
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC

from comprehensive_scientific_finalization import (
    PacingVariantD1_CoincidentRise, ReferenceVegasAgent, RenoAgent, CubicAgent, AuditedBBRAgent,
    run_enhanced_sim
)
from step2_rl_controllers_ppo_sac import TCPCongestionEnv

SEEDS = [42, 43, 44, 45]
N = len(SEEDS)
T_CRIT = 3.182

# Load models
ppo_model = PPO.load("final_ppo_policy.zip")
sac_model = SAC.load("final_sac_policy.zip")

class RLAgent(PacingVariantD1_CoincidentRise):
    def __init__(self, name, sb3_model):
        super().__init__(name)
        self.sb3_model = sb3_model
        self.obs_buf = [[1.0, 0.0, 1.0, 1.0, 0.0]] * 8

    def observe_ack(self, rtt, loss_event, acked_proc):
        super().observe_ack(rtt, loss_event, acked_proc)
        curr_rtt = self.rtt_history[-1]
        r_ratio = curr_rtt / max(self.min_rtt, 1e-3)
        g1 = curr_rtt - self.rtt_history[-2]
        deliv = max(self.window) if self.window else self.est_delivery_rate
        self.obs_buf.append([r_ratio, g1, deliv, self.rate, 1.0 if loss_event else 0.0])
        if len(self.obs_buf) > 8: self.obs_buf.pop(0)

    def update_rate(self, loss_event):
        min_rate = self.get_min_rate()
        obs_flat = np.array(self.obs_buf, dtype=np.float32).flatten()
        action, _ = self.sb3_model.predict(obs_flat, deterministic=True)
        act_val = float(np.clip(action[0], -1.0, 1.0))
        multiplier = 1.0 + 0.5 * act_val
        est_deliv = max(self.window) if self.window else self.est_delivery_rate
        max_rate = max(1.0, 2.0 * est_deliv)
        self.rate = float(np.clip(self.rate * multiplier, min_rate, max_rate))

controllers = {
    'Hybrid Rule': lambda: PacingVariantD1_CoincidentRise("HybridRule"),
    'PPO Controller': lambda: RLAgent("PPO", ppo_model),
    'SAC Controller': lambda: RLAgent("SAC", sac_model),
    'Reference Vegas': lambda: ReferenceVegasAgent("Vegas"),
    'Audited BBR': lambda: AuditedBBRAgent("BBR"),
    'TCP Reno': lambda: RenoAgent("Reno"),
    'TCP CUBIC': lambda: CubicAgent("CUBIC")
}

test_scenarios = [
    {"id": "HO8_RTT20", "name": "Unseen Fast RTT (20ms)", "cap": 3.0, "buf": 50.0, "base_rtt": 20.0, "loss": 0.0, "pat": "constant", "flows": 1},
    {"id": "HO9_RTT200", "name": "Unseen Satellite RTT (200ms)", "cap": 3.0, "buf": 50.0, "base_rtt": 200.0, "loss": 0.0, "pat": "constant", "flows": 1},
    {"id": "HO6_2Flows", "name": "Multi-Flow 2 Flows (B=50)", "cap": 4.0, "buf": 50.0, "base_rtt": 50.0, "loss": 0.0, "pat": "constant", "flows": 2},
    {"id": "HO7_8Flows", "name": "Multi-Flow 8 Flows (B=50)", "cap": 6.0, "buf": 50.0, "base_rtt": 50.0, "loss": 0.0, "pat": "constant", "flows": 8}
]

records = []
print("=" * 80)
print("  TIER 1 - ITEM 2: TABLE CONSISTENCY RE-RUN (preliminary n=4)")
print("=" * 80)

for sc in test_scenarios:
    print(f"\nEvaluating {sc['id']}: {sc['name']} (Base RTT={sc['base_rtt']}ms, Buffer={sc['buf']}pkts, Flows={sc['flows']})...")
    theoretical_max_rtt = sc['base_rtt'] + 1.9 * sc['buf']
    print(f"  Physical Maximum RTT Ceiling: {sc['base_rtt']} + 1.9 * {sc['buf']} = {theoretical_max_rtt:.1f} ms")

    for c_name, maker in controllers.items():
        gp_list, rtt_list, loss_list, u1_list, jfi_list = [], [], [], [], []
        for s in SEEDS:
            agents = [maker() for _ in range(sc['flows'])]
            res = run_enhanced_sim(agents, total_steps=3000, seed=s, buffer_capacity=sc['buf'],
                                  base_rtt=sc['base_rtt'], random_loss_rate=sc['loss'],
                                  bandwidth_pattern=sc['pat'], fixed_capacity=sc['cap'],
                                  stagger_flows=(sc['flows'] > 1))
            gp_list.append(res['goodput'])
            rtt_list.append(res['avg_rtt'])
            loss_list.append(res['total_loss'])
            jfi_list.append(res['jfi'])
            u1 = np.log(max(res['goodput'], 0.01)) - np.log(max(res['avg_rtt'], 1.0))
            u1_list.append(u1)

        m_gp = np.mean(gp_list)
        ci_gp = T_CRIT * np.std(gp_list, ddof=1) / np.sqrt(N) if np.std(gp_list, ddof=1) > 0 else 0.0
        m_rtt = np.mean(rtt_list)
        ci_rtt = T_CRIT * np.std(rtt_list, ddof=1) / np.sqrt(N) if np.std(rtt_list, ddof=1) > 0 else 0.0
        m_loss = np.mean(loss_list)
        m_u1 = np.mean(u1_list)
        ci_u1 = T_CRIT * np.std(u1_list, ddof=1) / np.sqrt(N) if np.std(u1_list, ddof=1) > 0 else 0.0
        m_jfi = np.mean(jfi_list)

        # Check physical ceiling
        if m_rtt > theoretical_max_rtt + 2.0:
            print(f"  [ERROR] Controller {c_name} exceeded physical max RTT ({m_rtt:.1f} > {theoretical_max_rtt:.1f})!")

        records.append({
            'Scenario': sc['id'],
            'Base_RTT': sc['base_rtt'],
            'Buffer': sc['buf'],
            'Max_Physical_RTT': theoretical_max_rtt,
            'Controller': c_name,
            'Goodput': f"{m_gp:.4f} +/- {ci_gp:.4f}",
            'Loss': f"{m_loss:.1f}",
            'Avg_RTT': f"{m_rtt:.1f} +/- {ci_rtt:.1f}",
            'JFI': f"{m_jfi:.3f}" if sc['flows'] > 1 else "1.000",
            'U_1.0': f"{m_u1:.4f} +/- {ci_u1:.4f}",
            'Evaluation_Status': "preliminary (n=4)"
        })

df_cons = pd.DataFrame(records)
df_cons.to_csv("preliminary_table_consistency_rerun_n4.csv", index=False)
print("\n[OK] Saved preliminary_table_consistency_rerun_n4.csv")
print(df_cons[['Scenario', 'Controller', 'Goodput', 'Loss', 'Avg_RTT', 'U_1.0']].to_string(index=False))
