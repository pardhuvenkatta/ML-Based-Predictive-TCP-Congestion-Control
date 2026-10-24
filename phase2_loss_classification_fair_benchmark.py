"""
Phase 2: Loss-Type Classifier vs. Tuned Non-ML Rules (Wireless vs Congestion Loss)
Script: phase2_loss_classification_fair_benchmark.py

Evaluates whether Machine Learning genuinely beats the best Non-ML heuristics
at discriminating random wireless packet corruption from buffer overflow drops.

Fair Baseline Comparisons Enforced:
(a) Default Non-ML Rule (rtt_ratio > 1.12)
(b) Tuned Non-ML Rule (optimal threshold tuned on TRAIN with 50-trial budget)
(c) Standard Reactive Backoff (always cut rate by 44.2% on drop, "model off")
(d) ML Loss Classifier (Random Forest trained on diverse TRAIN dataset)

Evaluation across 20 Seeds (42-61) on HELD-OUT Loss Environments:
1. Low Random Loss (0.1%)
2. Moderate Random Loss (0.5%)
3. Standard Random Loss (1.0%)
4. High Random Loss (2.0%)
5. Bursty Gilbert-Elliott Loss Channel
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from comprehensive_scientific_finalization import (
    ProtocolAgent,
    PacingVariantD1_CoincidentRise,
    RenoAgent,
    CubicAgent
)

# =====================================================================
# 1. SIMULATOR FOR CONTROLLED LOSS BENCHMARKS
# =====================================================================
def run_loss_benchmark_sim(
    agent,
    total_steps=3000,
    seed=42,
    buffer_capacity=50.0,
    base_rtt=50.0,
    fixed_capacity=3.0,
    loss_mode="random",
    random_loss_rate=0.0,
    ge_p_good_to_bad=0.02,
    ge_p_bad_to_good=0.30,
    ge_bad_loss_prob=0.25
):
    rng = np.random.RandomState(seed)
    queue = 0.0
    agent.rate = 1.05
    agent.min_rtt = None
    agent.rtt_history = []
    agent.deliv_history = []
    agent.recent_losses = [0] * 6
    if hasattr(agent, 'window'): agent.window = []
    if hasattr(agent, 'standing_queue_steps'): agent.standing_queue_steps = 0
    if hasattr(agent, 'in_competitive_mode'): agent.in_competitive_mode = False
    
    total_loss = 0
    congestion_loss = 0
    wireless_loss = 0
    total_proc = 0
    rtt_log = []
    ge_state = 0
    
    for t in range(total_steps):
        rate = max(0.02, agent.rate)
        
        # 1. Wireless channel drop check
        is_wireless_drop = False
        if loss_mode == "random":
            if random_loss_rate > 0 and rng.uniform() < random_loss_rate:
                is_wireless_drop = True
        elif loss_mode == "gilbert_elliott":
            if ge_state == 0 and rng.uniform() < ge_p_good_to_bad:
                ge_state = 1
            elif ge_state == 1 and rng.uniform() < ge_p_bad_to_good:
                ge_state = 0
            if ge_state == 1 and rng.uniform() < ge_bad_loss_prob:
                is_wireless_drop = True
                
        # 2. Buffer ingress & Congestion drop check
        is_congestion_drop = False
        queue += rate
        if queue > buffer_capacity:
            c_loss = queue - buffer_capacity
            congestion_loss += c_loss
            total_loss += c_loss
            queue = buffer_capacity
            is_congestion_drop = True
            
        if is_wireless_drop:
            wireless_loss += 1.0
            total_loss += 1.0
            
        loss_event = is_congestion_drop or is_wireless_drop
        
        # 3. Router packet forwarding
        proc = min(queue, fixed_capacity)
        queue -= proc
        total_proc += proc
        
        # 4. RTT telemetry (50ms base + 1.9ms per queued packet)
        curr_rtt = base_rtt + (queue * 1.9) + rng.normal(0, 0.5)
        curr_rtt = max(10.0, curr_rtt)
        rtt_log.append(curr_rtt)
        
        # 5. Agent ACK Observation & Rate Update
        agent.observe_ack(curr_rtt, loss_event, proc)
        agent.update_rate(loss_event)
        
    avg_rtt = np.mean(rtt_log)
    # Effective goodput penalizes dropped packets and retransmissions
    goodput = (total_proc - 20.0 - total_loss * 5.0) / total_steps
    goodput = max(goodput, 0.001)
    
    return {
        "goodput": goodput,
        "avg_rtt": avg_rtt,
        "total_loss": total_loss,
        "congestion_loss": congestion_loss,
        "wireless_loss": wireless_loss
    }


# =====================================================================
# 2. ML LOSS CLASSIFIER CONTROLLER
# =====================================================================
class MLLossClassifierAgent(PacingVariantD1_CoincidentRise):
    def __init__(self, name, model=None, cut_factor=0.90):
        super().__init__(name)
        self.model = model
        self.cut_factor = cut_factor
        self.rf_decisions = []

    def update_rate(self, loss_event):
        min_rate = self.get_min_rate()
        curr_rtt = self.rtt_history[-1]
        rtt_ratio = curr_rtt / max(self.min_rtt, 1e-3)
        grad_1 = curr_rtt - self.rtt_history[-2]
        grad_3 = (curr_rtt - self.rtt_history[-4]) / 3.0 if len(self.rtt_history) >= 4 else grad_1
        grad_5 = (curr_rtt - self.rtt_history[-6]) / 5.0 if len(self.rtt_history) >= 6 else grad_1

        if loss_event:
            is_congestion = True
            if self.model is not None:
                deliv = max(self.window) if self.window else self.est_delivery_rate
                feats = np.array([[
                    rtt_ratio,
                    grad_1,
                    grad_3,
                    grad_5,
                    deliv,
                    self.rate,
                    sum(self.recent_losses)
                ]])
                pred = self.model.predict(feats)[0]
                is_congestion = bool(pred == 1)
                self.rf_decisions.append(1 if is_congestion else 0)
            else:
                is_congestion = (rtt_ratio > 1.12)

            factor = 0.558 if is_congestion else self.cut_factor
            self.rate = max(min_rate, self.rate * factor)
            return

        target = max(self.window) if self.window else self.est_delivery_rate
        if grad_1 <= 0 and rtt_ratio < 1.12:
            self.rate += 0.05
        else:
            self.rate += (target - self.rate) * 0.14


class RuleLossClassifierAgent(PacingVariantD1_CoincidentRise):
    def __init__(self, name, threshold=1.12, cut_factor=0.90):
        super().__init__(name)
        self.threshold = threshold
        self.cut_factor = cut_factor

    def update_rate(self, loss_event):
        min_rate = self.get_min_rate()
        curr_rtt = self.rtt_history[-1]
        rtt_ratio = curr_rtt / max(self.min_rtt, 1e-3)
        grad_1 = curr_rtt - self.rtt_history[-2]

        if loss_event:
            factor = 0.558 if rtt_ratio > self.threshold else self.cut_factor
            self.rate = max(min_rate, self.rate * factor)
            return

        target = max(self.window) if self.window else self.est_delivery_rate
        if grad_1 <= 0 and rtt_ratio < self.threshold:
            self.rate += 0.05
        else:
            self.rate += (target - self.rate) * 0.14


class StandardBackoffAgent(PacingVariantD1_CoincidentRise):
    def update_rate(self, loss_event):
        min_rate = self.get_min_rate()
        curr_rtt = self.rtt_history[-1]
        rtt_ratio = curr_rtt / max(self.min_rtt, 1e-3)
        grad_1 = curr_rtt - self.rtt_history[-2]

        if loss_event:
            self.rate = max(min_rate, self.rate * 0.558)
            return

        target = max(self.window) if self.window else self.est_delivery_rate
        if grad_1 <= 0 and rtt_ratio < 1.12:
            self.rate += 0.05
        else:
            self.rate += (target - self.rate) * 0.14


# =====================================================================
# 3. DIVERSE TRAIN DATASET GENERATION (BALANCED CONGESTION & WIRELESS DROPS)
# =====================================================================
def train_loss_classifier_model(train_seeds=[101, 102, 103, 104, 105]):
    print("\n[+] Generating Diverse TRAIN Dataset for Loss Classification...")
    X_samples = []
    y_labels = []
    
    # Use diverse policies (Reno, CUBIC, Pacing with random walk capacity)
    for s in train_seeds:
        rng = np.random.RandomState(s)
        for policy in ["reno", "cubic", "pacing_dyn_cap"]:
            for buf in [15.0, 35.0, 50.0]:
                for loss_r in [0.005, 0.020]:
                    queue = 0.0
                    base_rtt = 50.0
                    cap = 3.0
                    
                    if policy == "reno":
                        ag = RenoAgent("RenoTrain")
                    elif policy == "cubic":
                        ag = CubicAgent("CubicTrain")
                    else:
                        ag = PacingVariantD1_CoincidentRise("PacingTrain")
                        
                    ag.rate = 1.05
                    ag.min_rtt = base_rtt
                    ag.rtt_history = [base_rtt] * 6
                    ag.recent_losses = [0] * 6
                    
                    for t in range(1200):
                        if policy == "pacing_dyn_cap":
                            # Dynamic capacity shifts
                            cap = max(1.0, 3.0 + np.sin(t / 40.0) * 1.5 + rng.normal(0, 0.2))
                            
                        rate = max(0.02, ag.rate)
                        queue += rate
                        
                        is_cong = False
                        is_wire = False
                        if queue > buf:
                            queue = buf
                            is_cong = True
                        if rng.uniform() < loss_r:
                            is_wire = True
                            
                        loss_event = is_cong or is_wire
                        proc = min(queue, cap)
                        queue -= proc
                        curr_rtt = base_rtt + (queue * 1.9) + rng.normal(0, 0.5)
                        
                        ag.observe_ack(curr_rtt, loss_event, proc)
                        ag.update_rate(loss_event)
                        
                        if loss_event:
                            rtt_ratio = curr_rtt / max(ag.min_rtt, 1e-3)
                            grad_1 = curr_rtt - ag.rtt_history[-2]
                            grad_3 = (curr_rtt - ag.rtt_history[-4]) / 3.0 if len(ag.rtt_history) >= 4 else grad_1
                            grad_5 = (curr_rtt - ag.rtt_history[-6]) / 5.0 if len(ag.rtt_history) >= 6 else grad_1
                            deliv = max(ag.deliv_history) if ag.deliv_history else ag.est_delivery_rate
                            
                            feat = [
                                rtt_ratio,
                                grad_1,
                                grad_3,
                                grad_5,
                                deliv,
                                ag.rate,
                                sum(ag.recent_losses)
                            ]
                            label = 1 if is_cong else 0
                            X_samples.append(feat)
                            y_labels.append(label)
                            
    X = np.array(X_samples)
    y = np.array(y_labels)
    n_cong = np.sum(y == 1)
    n_wire = np.sum(y == 0)
    print(f"  [OK] Dataset Generated: {len(X)} loss events (Congestion Drops: {n_cong}, Wireless Drops: {n_wire})")
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X, y)
    y_pred = rf.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    print(f"  [OK] RF Classifier Fitted: Accuracy = {acc:.4f}, Precision = {prec:.4f}, Recall = {rec:.4f}, F1 = {f1:.4f}")
    
    feat_names = ["rtt_ratio", "grad_1", "grad_3", "grad_5", "delivery_rate", "send_rate", "loss_count"]
    print("       Feature Importances:")
    for fn, imp in zip(feat_names, rf.feature_importances_):
        print(f"        - {fn:14s}: {imp:.4f}")
        
    return rf, X, y


# =====================================================================
# 4. TUNING THE NON-ML RULE ON TRAIN WITH EQUIVALENT BUDGET
# =====================================================================
def tune_non_ml_rule_on_train(X_train, y_train, budget=50):
    print(f"\n[+] Tuning Non-ML Loss Classification Rule on TRAIN (Budget: {budget} trials)...")
    thresholds = np.linspace(1.02, 1.30, budget)
    best_thresh = 1.12
    best_f1 = 0.0
    
    rtt_ratios = X_train[:, 0]
    for th in thresholds:
        preds = (rtt_ratios > th).astype(int)
        f1 = f1_score(y_train, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th
            
    print(f"  [OK] Non-ML Rule Tuning Complete. Optimal Threshold = {best_thresh:.3f} (Train F1 = {best_f1:.4f})")
    return best_thresh


# =====================================================================
# 5. 20-SEED HELD-OUT LOSS EVALUATION (SEEDS 42-61)
# =====================================================================
def run_phase2_benchmark():
    start_time = time.time()
    print("=" * 80)
    print("  PHASE 2: LOSS CLASSIFIER vs TUNED NON-ML RULES (HELD-OUT EVALUATION)")
    print("=" * 80)
    
    # 1. Train Model & Tune Non-ML Rule
    rf_model, X_train, y_train = train_loss_classifier_model()
    tuned_threshold = tune_non_ml_rule_on_train(X_train, y_train, budget=50)
    default_threshold = 1.12
    
    # 2. Define Held-Out Loss Environments
    loss_scenarios = [
        {"id": "L1_Rand01", "name": "Random Loss 0.1%", "mode": "random", "loss_rate": 0.001},
        {"id": "L2_Rand05", "name": "Random Loss 0.5%", "mode": "random", "loss_rate": 0.005},
        {"id": "L3_Rand10", "name": "Random Loss 1.0%", "mode": "random", "loss_rate": 0.010},
        {"id": "L4_Rand20", "name": "Random Loss 2.0%", "mode": "random", "loss_rate": 0.020},
        {"id": "L5_Gilbert", "name": "Gilbert-Elliott Bursty Loss", "mode": "gilbert_elliott", "loss_rate": 0.0}
    ]
    
    seeds = list(range(42, 62)) # 20 fixed evaluation seeds
    records = []
    paired_summary = []
    
    print(f"\n[+] Evaluating across 20 Seeds (42-61) on {len(loss_scenarios)} Held-Out Loss Scenarios...")
    
    for sc in loss_scenarios:
        sc_id = sc["id"]
        sc_name = sc["name"]
        print(f"  -> Scenario: {sc_name}")
        
        diffs_ml_vs_def = []
        diffs_ml_vs_tun = []
        diffs_ml_vs_off = []
        
        diffs_gp_vs_tun = []
        diffs_rtt_vs_tun = []
        
        for s in seeds:
            ag_off = StandardBackoffAgent("StdBackoff")
            ag_def = RuleLossClassifierAgent("Rule_Def", threshold=default_threshold)
            ag_tun = RuleLossClassifierAgent("Rule_Tun", threshold=tuned_threshold)
            ag_ml  = MLLossClassifierAgent("ML_Classifier", model=rf_model)
            
            kwargs = {
                "total_steps": 2500,
                "seed": s,
                "buffer_capacity": 50.0,
                "base_rtt": 50.0,
                "fixed_capacity": 3.0,
                "loss_mode": sc["mode"],
                "random_loss_rate": sc["loss_rate"]
            }
            
            res_off = run_loss_benchmark_sim(ag_off, **kwargs)
            res_def = run_loss_benchmark_sim(ag_def, **kwargs)
            res_tun = run_loss_benchmark_sim(ag_tun, **kwargs)
            res_ml  = run_loss_benchmark_sim(ag_ml,  **kwargs)
            
            u_off = np.log(res_off["goodput"]) - 1.0 * np.log(res_off["avg_rtt"])
            u_def = np.log(res_def["goodput"]) - 1.0 * np.log(res_def["avg_rtt"])
            u_tun = np.log(res_tun["goodput"]) - 1.0 * np.log(res_tun["avg_rtt"])
            u_ml  = np.log(res_ml["goodput"])  - 1.0 * np.log(res_ml["avg_rtt"])
            
            d_def = u_ml - u_def
            d_tun = u_ml - u_tun
            d_off = u_ml - u_off
            
            diffs_ml_vs_def.append(d_def)
            diffs_ml_vs_tun.append(d_tun)
            diffs_ml_vs_off.append(d_off)
            
            diffs_gp_vs_tun.append(res_ml["goodput"] - res_tun["goodput"])
            diffs_rtt_vs_tun.append(res_ml["avg_rtt"] - res_tun["avg_rtt"])
            
            records.append({
                "scenario_id": sc_id,
                "seed": s,
                "gp_off": res_off["goodput"], "u_off": u_off,
                "gp_def": res_def["goodput"], "u_def": u_def,
                "gp_tun": res_tun["goodput"], "u_tun": u_tun,
                "gp_ml":  res_ml["goodput"],  "u_ml":  u_ml,
                "delta_ml_vs_tun": d_tun,
                "delta_ml_vs_def": d_def,
                "delta_ml_vs_off": d_off
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
        
        paired_summary.append({
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
            "mean_diff_gp": np.mean(diffs_gp_vs_tun),
            "ml_genuinely_wins": ml_wins
        })
        
    df_sum = pd.DataFrame(paired_summary)
    
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
    
    csv_raw = "phase2_loss_classification_raw.csv"
    csv_summary = "phase2_loss_classification_fair_benchmark.csv"
    pd.DataFrame(records).to_csv(csv_raw, index=False)
    df_sum.to_csv(csv_summary, index=False)
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("  PHASE 2 COMPLETE: LOSS CLASSIFIER SUMMARY OF REAL NUMBERS")
    print(f"  Execution time: {elapsed:.1f}s")
    print("=" * 80)
    print(df_sum[["scenario_name", "mean_u_tun", "mean_u_ml", "delta_vs_tun", "ci_lower", "ci_upper", "p_value", "holm_significant", "ml_genuinely_wins"]].to_string(index=False))
    print(f"\n[+] Raw per-seed runs saved to: {csv_raw}")
    print(f"[+] Paired summary saved to:   {csv_summary}")
    
    return df_sum

if __name__ == "__main__":
    run_phase2_benchmark()
