"""
Phase 3: Multi-Opponent Competitor Classification (CUBIC vs. BBR vs. Reno vs. Copa vs. Alone)
Script: phase3_competitor_classification_benchmark.py

Fast, optimized, publication-quality implementation:
- 5 Host-observable features: [rtt_ratio, grad_1, grad_5, rate_to_delivery, standing_queue_steps]
- Lightweight periodic evaluation (every 5 steps) matching kernel eBPF sampling
- Fair TRAIN tuning of the Non-ML threshold
- 20 seeds (42-61) on HELD-OUT competition scenarios:
  1. Alone (No Competitor, measuring latency tax)
  2. vs TCP Reno (Cap 6.0, Buffer 50)
  3. vs TCP CUBIC (Cap 6.0, Buffer 50)
  4. vs Google BBR (Cap 6.0, Buffer 50)
  5. vs Delay-Based Copa (Cap 6.0, Buffer 50)
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from comprehensive_scientific_finalization import (
    ProtocolAgent,
    PacingVariantD1_CoincidentRise,
    AuditedBBRAgent,
    RenoAgent,
    CubicAgent,
    run_enhanced_sim
)

class CopaAgent(ProtocolAgent):
    def __init__(self, name, delta=0.5):
        super().__init__(name)
        self.delta = delta
        self.standing_queue_steps = 0
        self.competitive = False

    def observe_ack(self, rtt, loss_event, acked_proc):
        super().observe_ack(rtt, loss_event, acked_proc)
        curr_rtt = self.rtt_history[-1]
        rtt_ratio = curr_rtt / max(self.min_rtt, 1e-3)
        if rtt_ratio > 1.15:
            self.standing_queue_steps += 1
        else:
            self.standing_queue_steps = max(0, self.standing_queue_steps - 1)
        if self.standing_queue_steps > 50:
            self.competitive = True
        elif self.standing_queue_steps == 0:
            self.competitive = False

    def update_rate(self, loss_event):
        min_rate = self.get_min_rate()
        if loss_event:
            self.rate = max(min_rate, self.rate * 0.70)
            return
        curr_rtt = self.rtt_history[-1]
        queue_delay_sec = max(1e-4, (curr_rtt - self.min_rtt) / 1000.0)
        eff_delta = (self.delta / 2.0) if self.competitive else self.delta
        target_rate = max(0.5, 1.0 / (eff_delta * queue_delay_sec * 50.0))
        if self.rate < target_rate:
            self.rate += 0.05
        else:
            self.rate = max(min_rate, self.rate - 0.05)


# 1. "Model Off": Pure delay-aware pacing
class PurePacingAgent(PacingVariantD1_CoincidentRise):
    pass


# 2. Rule-Based Competitor Detection Agent (Default & Tuned)
class RuleCompetitiveAgent(PacingVariantD1_CoincidentRise):
    def __init__(self, name, threshold=60, aggressive_step=0.10, competitive_backoff=0.70):
        super().__init__(name)
        self.standing_queue_steps = 0
        self.in_competitive_mode = False
        self.threshold = threshold
        self.aggressive_step = aggressive_step
        self.competitive_backoff = competitive_backoff
        self.mode_switches = 0

    def observe_ack(self, rtt, loss_event, acked_proc):
        super().observe_ack(rtt, loss_event, acked_proc)
        curr_rtt = self.rtt_history[-1]
        rtt_ratio = curr_rtt / max(self.min_rtt, 1e-3)

        if rtt_ratio > 1.20:
            self.standing_queue_steps += 1
        else:
            self.standing_queue_steps = max(0, self.standing_queue_steps - 2)

        if self.standing_queue_steps > self.threshold and not self.in_competitive_mode:
            self.in_competitive_mode = True
            self.mode_switches += 1
        elif self.standing_queue_steps == 0 and self.in_competitive_mode:
            self.in_competitive_mode = False

    def update_rate(self, loss_event):
        min_rate = self.get_min_rate()
        curr_rtt = self.rtt_history[-1]
        rtt_ratio = curr_rtt / max(self.min_rtt, 1e-3)
        grad_1 = curr_rtt - self.rtt_history[-2]

        if self.in_competitive_mode:
            if loss_event:
                self.rate = max(min_rate, self.rate * self.competitive_backoff)
            else:
                self.rate += self.aggressive_step
            return

        if loss_event:
            factor = 0.558 if rtt_ratio > 1.12 else 0.85
            self.rate = max(min_rate, self.rate * factor)
            return

        target = max(self.window) if self.window else self.est_delivery_rate
        if grad_1 <= 0 and rtt_ratio < 1.12:
            self.rate += 0.05
        else:
            self.rate += (target - self.rate) * 0.14


# 3. High-Performance ML Competitor Classifier Agent
class MLCompetitorAgent(PacingVariantD1_CoincidentRise):
    def __init__(self, name, model=None):
        super().__init__(name)
        self.model = model
        self.standing_queue_steps = 0
        self.detected_class = 0
        self.mode_switches = 0
        self.step_counter = 0

    def observe_ack(self, rtt, loss_event, acked_proc):
        super().observe_ack(rtt, loss_event, acked_proc)
        self.step_counter += 1
        curr_rtt = self.rtt_history[-1]
        rtt_ratio = curr_rtt / max(self.min_rtt, 1e-3)
        grad_1 = curr_rtt - self.rtt_history[-2]
        grad_5 = (curr_rtt - self.rtt_history[-6]) / 5.0 if len(self.rtt_history) >= 6 else grad_1

        if rtt_ratio > 1.20:
            self.standing_queue_steps += 1
        else:
            self.standing_queue_steps = max(0, self.standing_queue_steps - 2)

        # Periodic ML prediction every 5 steps (eBPF sampling cadence)
        if self.step_counter % 5 == 0 and self.model is not None:
            rate_ratio = self.rate / max(self.est_delivery_rate, 1e-3)
            feats = np.array([[
                rtt_ratio,
                grad_1,
                grad_5,
                rate_ratio,
                float(self.standing_queue_steps)
            ]])
            pred_class = self.model.predict(feats)[0]
            if pred_class != self.detected_class:
                self.mode_switches += 1
                self.detected_class = pred_class

    def update_rate(self, loss_event):
        min_rate = self.get_min_rate()
        curr_rtt = self.rtt_history[-1]
        rtt_ratio = curr_rtt / max(self.min_rtt, 1e-3)
        grad_1 = curr_rtt - self.rtt_history[-2]

        if self.detected_class == 1:
            # Loss-Based Competitor (Reno / CUBIC)
            if loss_event:
                self.rate = max(min_rate, self.rate * 0.70)
            else:
                self.rate += 0.12 # Strong probing
            return
        elif self.detected_class == 2:
            # Rate-Based Competitor (BBR)
            if loss_event:
                self.rate = max(min_rate, self.rate * 0.80)
            else:
                target = max(self.window) if self.window else self.est_delivery_rate
                self.rate += (target - self.rate) * 0.20
            return
        elif self.detected_class == 3:
            # Delay-Based Competitor (Copa / Vegas)
            if loss_event:
                self.rate = max(min_rate, self.rate * 0.85)
            else:
                if grad_1 <= 0 and rtt_ratio < 1.08:
                    self.rate += 0.04
                else:
                    self.rate = max(min_rate, self.rate - 0.02)
            return

        # Class 0: Alone (Standard Delay-Aware Pacing)
        if loss_event:
            factor = 0.558 if rtt_ratio > 1.12 else 0.85
            self.rate = max(min_rate, self.rate * factor)
            return

        target = max(self.window) if self.window else self.est_delivery_rate
        if grad_1 <= 0 and rtt_ratio < 1.12:
            self.rate += 0.05
        else:
            self.rate += (target - self.rate) * 0.14


# =====================================================================
# 2. TRAIN DATASET GENERATION & MODEL FITTING (TRAIN SEEDS)
# =====================================================================
def train_competitor_detector_model(train_seeds=[101, 102, 103]):
    print("\n[+] Generating Diverse TRAIN Telemetry for Competitor Detection...")
    X_samples = []
    y_labels = []
    
    configs = [
        {"class": 0, "opp": None,    "cap": 3.0},
        {"class": 1, "opp": "reno",  "cap": 6.0},
        {"class": 1, "opp": "cubic", "cap": 6.0},
        {"class": 2, "opp": "bbr",   "cap": 6.0},
        {"class": 3, "opp": "copa",  "cap": 6.0}
    ]
    
    for s in train_seeds:
        for cfg in configs:
            target_class = cfg["class"]
            opp_type = cfg["opp"]
            cap = cfg["cap"]
            
            ag = PacingVariantD1_CoincidentRise("AgentTrain")
            opp = None
            if opp_type == "reno": opp = RenoAgent("Reno")
            elif opp_type == "cubic": opp = CubicAgent("CUBIC")
            elif opp_type == "bbr": opp = AuditedBBRAgent("BBR")
            elif opp_type == "copa": opp = CopaAgent("Copa")
            
            flows = [ag, opp] if opp else [ag]
            rng = np.random.RandomState(s)
            queue = 0.0
            base_rtt = 50.0
            standing_steps = 0
            
            for f in flows:
                f.rate = 1.05
                f.min_rtt = base_rtt
                f.rtt_history = [base_rtt] * 6
                f.recent_losses = [0] * 6
                
            for t in range(1000):
                total_send = sum(max(0.02, f.rate) for f in flows)
                queue += total_send
                loss_event = False
                if queue > 50.0:
                    queue = 50.0
                    loss_event = True
                    
                proc = min(queue, cap)
                queue -= proc
                curr_rtt = base_rtt + (queue * 1.9) + rng.normal(0, 0.5)
                
                for f in flows:
                    f.observe_ack(curr_rtt, loss_event, proc / len(flows))
                    f.update_rate(loss_event)
                    
                rtt_ratio = curr_rtt / max(ag.min_rtt, 1e-3)
                grad_1 = curr_rtt - ag.rtt_history[-2]
                grad_5 = (curr_rtt - ag.rtt_history[-6]) / 5.0 if len(ag.rtt_history) >= 6 else grad_1
                if rtt_ratio > 1.20: standing_steps += 1
                else: standing_steps = max(0, standing_steps - 2)
                
                if t > 50 and t % 5 == 0:
                    rate_ratio = ag.rate / max(ag.est_delivery_rate, 1e-3)
                    feat = [
                        rtt_ratio,
                        grad_1,
                        grad_5,
                        rate_ratio,
                        float(standing_steps)
                    ]
                    X_samples.append(feat)
                    y_labels.append(target_class)
                    
    X = np.array(X_samples)
    y = np.array(y_labels)
    print(f"  [OK] Dataset Generated: {len(X)} samples across 4 opponent classes.")
    
    rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
    rf.fit(X, y)
    print(f"  [OK] RF Competitor Model Fitted: Training Accuracy = {accuracy_score(y, rf.predict(X)):.4f}")
    
    feat_names = ["rtt_ratio", "grad_1", "grad_5", "rate_ratio", "standing_steps"]
    for fn, imp in zip(feat_names, rf.feature_importances_):
        print(f"        - {fn:14s}: {imp:.4f}")
        
    return rf, X, y


# =====================================================================
# 3. FAIR TUNING OF NON-ML COMPETITOR RULE ON TRAIN
# =====================================================================
def tune_non_ml_competitive_rule(budget=25):
    print(f"\n[+] Tuning Non-ML Competitor Rule on TRAIN (Budget: {budget} trials)...")
    thresholds = [int(x) for x in np.linspace(20, 100, budget)]
    best_thresh = 60
    best_score = -float('inf')
    
    for th in thresholds:
        scores = []
        for s in [101, 102]:
            ag_alone = RuleCompetitiveAgent("Alone", threshold=th)
            res_alone = run_enhanced_sim([ag_alone], total_steps=1000, seed=s, fixed_capacity=3.0, buffer_capacity=50.0)
            u_alone = np.log(max(res_alone["flow_gps"][0], 0.01)) - 1.0 * np.log(max(res_alone["avg_rtt"], 1.0))
            
            ag_comp = RuleCompetitiveAgent("Comp", threshold=th)
            reno = RenoAgent("Reno")
            res_comp = run_enhanced_sim([ag_comp, reno], total_steps=1000, seed=s, fixed_capacity=6.0, buffer_capacity=50.0)
            u_comp = np.log(max(res_comp["flow_gps"][0], 0.01)) - 1.0 * np.log(max(res_comp["avg_rtt"], 1.0))
            
            scores.append(u_alone + u_comp)
        mean_s = np.mean(scores)
        if mean_s > best_score:
            best_score = mean_s
            best_thresh = th
            
    print(f"  [OK] Rule Tuning Complete. Optimal Standing Queue Threshold = {best_thresh} steps.")
    return best_thresh


# =====================================================================
# 4. HELD-OUT COMPETITION BENCHMARK ACROSS 20 SEEDS (42-61)
# =====================================================================
def run_phase3_benchmark():
    start_time = time.time()
    print("=" * 80)
    print("  PHASE 3: COMPETITOR CLASSIFICATION vs TUNED RULE BENCHMARK")
    print("=" * 80)
    
    rf_model, X_tr, y_tr = train_competitor_detector_model()
    tuned_threshold = tune_non_ml_competitive_rule(budget=25)
    default_threshold = 60
    
    scenarios = [
        {"id": "C1_Alone", "name": "Operating Alone (Latency Tax Test)", "opp": None, "cap": 3.0, "buf": 50.0},
        {"id": "C2_vsReno", "name": "Competing vs TCP Reno", "opp": "reno", "cap": 6.0, "buf": 50.0},
        {"id": "C3_vsCUBIC", "name": "Competing vs TCP CUBIC", "opp": "cubic", "cap": 6.0, "buf": 50.0},
        {"id": "C4_vsBBR", "name": "Competing vs Google BBR", "opp": "bbr", "cap": 6.0, "buf": 50.0},
        {"id": "C5_vsCopa", "name": "Competing vs Delay-Based Copa", "opp": "copa", "cap": 6.0, "buf": 50.0}
    ]
    
    seeds = list(range(42, 62)) # 20 fixed seeds
    records = []
    summary = []
    
    print(f"\n[+] Benchmarking across 20 Seeds (42-61) on {len(scenarios)} Held-Out Scenarios...")
    
    for sc in scenarios:
        sc_id = sc["id"]
        sc_name = sc["name"]
        print(f"  -> Scenario: {sc_name}")
        
        diffs_ml_vs_tun = []
        diffs_gp_vs_tun = []
        diffs_rtt_vs_tun = []
        
        for s in seeds:
            ag_off = PurePacingAgent("ModelOff")
            ag_def = RuleCompetitiveAgent("RuleDef", threshold=default_threshold)
            ag_tun = RuleCompetitiveAgent("RuleTun", threshold=tuned_threshold)
            ag_ml  = MLCompetitorAgent("MLAgent", model=rf_model)
            
            agents = [ag_off, ag_def, ag_tun, ag_ml]
            res_dict = {}
            
            for idx, candidate in enumerate(agents):
                opp = None
                if sc["opp"] == "reno": opp = RenoAgent(f"Reno_{idx}")
                elif sc["opp"] == "cubic": opp = CubicAgent(f"Cubic_{idx}")
                elif sc["opp"] == "bbr": opp = AuditedBBRAgent(f"BBR_{idx}")
                elif sc["opp"] == "copa": opp = CopaAgent(f"Copa_{idx}")
                
                flows = [candidate, opp] if opp else [candidate]
                res = run_enhanced_sim(
                    flows,
                    total_steps=2000,
                    seed=s,
                    fixed_capacity=sc["cap"],
                    buffer_capacity=sc["buf"],
                    base_rtt=50.0
                )
                gp = res["flow_gps"][0]
                rtt = res["avg_rtt"]
                u1 = np.log(max(gp, 0.01)) - 1.0 * np.log(max(rtt, 1.0))
                res_dict[candidate.name] = {"gp": gp, "rtt": rtt, "u1": u1}
                
            u_off = res_dict["ModelOff"]["u1"]
            u_def = res_dict["RuleDef"]["u1"]
            u_tun = res_dict["RuleTun"]["u1"]
            u_ml  = res_dict["MLAgent"]["u1"]
            
            d_tun = u_ml - u_tun
            diffs_ml_vs_tun.append(d_tun)
            diffs_gp_vs_tun.append(res_dict["MLAgent"]["gp"] - res_dict["RuleTun"]["gp"])
            diffs_rtt_vs_tun.append(res_dict["MLAgent"]["rtt"] - res_dict["RuleTun"]["rtt"])
            
            records.append({
                "scenario_id": sc_id,
                "seed": s,
                "u_off": u_off, "gp_off": res_dict["ModelOff"]["gp"], "rtt_off": res_dict["ModelOff"]["rtt"],
                "u_def": u_def, "gp_def": res_dict["RuleDef"]["gp"], "rtt_def": res_dict["RuleDef"]["rtt"],
                "u_tun": u_tun, "gp_tun": res_dict["RuleTun"]["gp"], "rtt_tun": res_dict["RuleTun"]["rtt"],
                "u_ml":  u_ml,  "gp_ml":  res_dict["MLAgent"]["gp"],  "rtt_ml":  res_dict["MLAgent"]["rtt"],
                "delta_vs_tun": d_tun
            })
            
        n = len(diffs_ml_vs_tun)
        m_diff = np.mean(diffs_ml_vs_tun)
        s_diff = np.std(diffs_ml_vs_tun, ddof=1)
        se = s_diff / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n-1)
        ci = (m_diff - t_crit * se, m_diff + t_crit * se)
        
        t_stat, p_val = stats.ttest_rel(
            [r["u_ml"] for r in records if r["scenario_id"] == sc_id],
            [r["u_tun"] for r in records if r["scenario_id"] == sc_id]
        )
        cohen_d = m_diff / s_diff if s_diff > 1e-6 else 0.0
        ml_wins = (m_diff > 0) and (ci[0] > 0)
        
        summary.append({
            "scenario_id": sc_id,
            "scenario_name": sc_name,
            "mean_u_off": np.mean([r["u_off"] for r in records if r["scenario_id"] == sc_id]),
            "mean_u_def": np.mean([r["u_def"] for r in records if r["scenario_id"] == sc_id]),
            "mean_u_tun": np.mean([r["u_tun"] for r in records if r["scenario_id"] == sc_id]),
            "mean_u_ml":  np.mean([r["u_ml"]  for r in records if r["scenario_id"] == sc_id]),
            "delta_vs_tun": m_diff,
            "ci_lower": ci[0],
            "ci_upper": ci[1],
            "p_value": p_val,
            "cohen_d": cohen_d,
            "mean_gp_ml": np.mean([r["gp_ml"] for r in records if r["scenario_id"] == sc_id]),
            "mean_gp_tun": np.mean([r["gp_tun"] for r in records if r["scenario_id"] == sc_id]),
            "mean_rtt_ml": np.mean([r["rtt_ml"] for r in records if r["scenario_id"] == sc_id]),
            "mean_rtt_tun": np.mean([r["rtt_tun"] for r in records if r["scenario_id"] == sc_id]),
            "mean_diff_gp": np.mean(diffs_gp_vs_tun),
            "mean_diff_rtt": np.mean(diffs_rtt_vs_tun),
            "ml_genuinely_wins": ml_wins
        })
        
    df_sum = pd.DataFrame(summary)
    
    p_vals = df_sum["p_value"].values
    order = np.argsort(p_vals)
    m = len(p_vals)
    holm_sig = [False] * m
    for rank, idx in enumerate(order):
        if p_vals[idx] < (0.05 / (m - rank)):
            holm_sig[idx] = True
        else:
            break
    df_sum["holm_significant"] = holm_sig
    
    csv_raw = "phase3_competitor_classification_raw.csv"
    csv_summary = "phase3_competitor_classification_benchmark.csv"
    pd.DataFrame(records).to_csv(csv_raw, index=False)
    df_sum.to_csv(csv_summary, index=False)
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("  PHASE 3 COMPLETE: COMPETITOR CLASSIFICATION SUMMARY OF REAL NUMBERS")
    print(f"  Execution time: {elapsed:.1f}s")
    print("=" * 80)
    print(df_sum[["scenario_name", "mean_u_tun", "mean_u_ml", "delta_vs_tun", "ci_lower", "ci_upper", "p_value", "holm_significant", "ml_genuinely_wins"]].to_string(index=False))
    print(f"\n[+] Raw per-seed runs saved to: {csv_raw}")
    print(f"[+] Paired summary saved to:   {csv_summary}")
    
    return df_sum

if __name__ == "__main__":
    run_phase3_benchmark()
