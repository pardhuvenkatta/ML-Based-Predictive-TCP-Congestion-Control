"""
Phase 1: Establishing the Fair Reference Baselines & TRAIN-Tuning Infrastructure
Script: phase1_fair_baseline_train_tuning.py

Ground Rules Enforced:
1. Strictly sender-observable telemetry (no router queue, buffer size, or link capacity).
2. Dynamic first-ACK bootstrapping for scale-invariance (no hardcoded base RTT).
3. Fair baseline comparison: Default Non-ML vs Tuned Non-ML (tuned on TRAIN scenarios with 50-trial budget).
4. Full evaluation across 20 seeds (42-61) on HELD-OUT scenarios (unseen RTT 300ms, buffer 3, buffer 200, loss, CUBIC, BBR, Copa).
5. Paired t-test, 95% CIs, effect sizes (Cohen's d), and Holm-Bonferroni correction.
"""

import os
import sys
import time
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats

# Import baseline components from existing codebase
from comprehensive_scientific_finalization import (
    ProtocolAgent,
    PacingVariantD1_CoincidentRise,
    ReferenceVegasAgent,
    AuditedBBRAgent,
    RenoAgent,
    CubicAgent,
    run_enhanced_sim
)

# =====================================================================
# 1. COPA AGENT IMPLEMENTATION (UNSEEN COMPETITOR FOR HELD-OUT)
# =====================================================================
class CopaAgent(ProtocolAgent):
    """
    Delay-based Copa agent (Arun & Balakrishnan, NSDI 2018).
    - Measures RTTmin.
    - Target rate: lambda = 1 / (delta * queuing_delay).
    - If rate < lambda: rate increases; if rate > lambda: rate decreases.
    - Scales smoothly and provides an unseen delay-based competitor.
    """
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
            
        # Copa competitive mode: if queue does not empty, double target rate
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
        
        # Smooth convergence to target
        if self.rate < target_rate:
            self.rate += 0.05
        else:
            self.rate = max(min_rate, self.rate - 0.05)


# =====================================================================
# 2. REFERENCE NON-ML CONTROLLER ("Pacing + RTT Rule + Competitive Mode")
# =====================================================================
class ReferenceNonMLAgent(PacingVariantD1_CoincidentRise):
    """
    The project's gold-standard non-ML reference controller:
    - Pacing to maximum delivery rate with Coincident-Rise flush ((rtt_ratio > 1.15 and grad_1 > 0)).
    - Dual-tier loss backoff (0.558 if rtt_ratio > threshold else cut_factor).
    - Copa/Nimbus-style competitive mode switching on persistent standing queue.
    - Scale-invariant first-ACK initialization (no hardcoded 50 ms).
    """
    def __init__(
        self,
        name,
        threshold=1.12,
        cut_factor=0.85,
        standing_queue_threshold=60,
        competitive_backoff=0.70,
        aggressive_step=0.10
    ):
        super().__init__(name)
        self.threshold = threshold
        self.cut_factor = cut_factor
        self.standing_queue_threshold = standing_queue_threshold
        self.competitive_backoff = competitive_backoff
        self.aggressive_step = aggressive_step
        self.standing_queue_steps = 0
        self.in_competitive_mode = False
        self.mode_switches = 0

    def observe_ack(self, rtt, loss_event, acked_proc):
        super().observe_ack(rtt, loss_event, acked_proc)
        curr_rtt = self.rtt_history[-1]
        rtt_ratio = curr_rtt / max(self.min_rtt, 1e-3)
        
        # Track persistent queue for competitive mode
        if rtt_ratio > 1.20:
            self.standing_queue_steps += 1
        else:
            self.standing_queue_steps = max(0, self.standing_queue_steps - 2)

        if self.standing_queue_steps > self.standing_queue_threshold and not self.in_competitive_mode:
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
            # Aggressive loss-resisting probing (CUBIC/Reno competition)
            if loss_event:
                self.rate = max(min_rate, self.rate * self.competitive_backoff)
            else:
                self.rate += self.aggressive_step
            return

        # Standard delay-aware pacing
        if loss_event:
            factor = 0.558 if rtt_ratio > self.threshold else self.cut_factor
            self.rate = max(min_rate, self.rate * factor)
            return

        target = max(self.window) if self.window else self.est_delivery_rate
        if grad_1 <= 0 and rtt_ratio < self.threshold:
            self.rate += 0.05
        else:
            self.rate += (target - self.rate) * 0.14


# =====================================================================
# 3. FAIR TRAIN TUNING (50 EVALUATION BUDGET)
# =====================================================================
def run_train_tuning_budget(budget=50, random_seed=42):
    """
    Tunes the Non-ML controller parameters on TRAIN scenarios only with a fixed compute budget.
    Parameters swept:
      - threshold in [1.03, 1.25]
      - cut_factor in [0.70, 0.95]
      - standing_queue_threshold in [20, 100]
      - competitive_backoff in [0.60, 0.85]
    """
    print(f"\n[+] Executing Fair TRAIN Tuning (Budget: {budget} iterations, Random Seed: {random_seed})...")
    rng = np.random.RandomState(random_seed)
    
    train_scenarios = [
        {"name": "TR1_Clean_Cap3", "cap": 3.0, "buf": 50.0, "rtt": 50.0, "loss": 0.0, "pat": "constant"},
        {"name": "TR2_Moderate_Cap5", "cap": 5.0, "buf": 50.0, "rtt": 100.0, "loss": 0.0, "pat": "constant"},
        {"name": "TR3_DynWalk_Cap4", "cap": 4.0, "buf": 50.0, "rtt": 50.0, "loss": 0.0, "pat": "random_walk"}
    ]
    tune_seeds = [101, 102, 103] # Strictly TRAIN tuning seeds
    
    best_score = -float('inf')
    best_params = None
    
    # 1. Include default configuration in evaluation
    candidates = [{
        "threshold": 1.12,
        "cut_factor": 0.85,
        "standing_queue_threshold": 60,
        "competitive_backoff": 0.70,
        "aggressive_step": 0.10
    }]
    
    # 2. Sample remaining budget randomly over parameter space
    for _ in range(budget - 1):
        candidates.append({
            "threshold": float(np.round(rng.uniform(1.03, 1.25), 3)),
            "cut_factor": float(np.round(rng.uniform(0.70, 0.95), 3)),
            "standing_queue_threshold": int(rng.randint(20, 100)),
            "competitive_backoff": float(np.round(rng.uniform(0.60, 0.85), 3)),
            "aggressive_step": float(np.round(rng.uniform(0.05, 0.15), 3))
        })
        
    for idx, cand in enumerate(candidates):
        scores = []
        for sc in train_scenarios:
            for s in tune_seeds:
                ag = ReferenceNonMLAgent("TuneAgent", **cand)
                res = run_enhanced_sim(
                    [ag],
                    total_steps=1500,
                    seed=s,
                    fixed_capacity=sc["cap"],
                    buffer_capacity=sc["buf"],
                    base_rtt=sc["rtt"],
                    random_loss_rate=sc["loss"],
                    bandwidth_pattern=sc["pat"]
                )
                gp = max(res["goodput"], 0.01)
                rtt = max(res["avg_rtt"], 1.0)
                u1 = np.log(gp) - 1.0 * np.log(rtt)
                scores.append(u1)
        mean_u1 = float(np.mean(scores))
        if mean_u1 > best_score:
            best_score = mean_u1
            best_params = cand
            
    print(f"  [OK] TRAIN Tuning Complete. Best Score U_1.0 = {best_score:.4f}")
    print(f"       Optimal Parameters: {best_params}")
    return best_params


# =====================================================================
# 4. HELD-OUT SCENARIO BENCHMARKING ACROSS 20 SEEDS (42-61)
# =====================================================================
def define_held_out_scenarios():
    """
    Defines strictly held-out scenario families:
    - Unseen Base RTT: 300 ms (extreme propagation delay)
    - Unseen Buffers: 3 pkts (micro-buffer), 200 pkts (deep buffer)
    - Unseen Loss: 1.0% random loss, bursty loss
    - Unseen Opponents: CUBIC, BBR, Copa
    """
    return [
        {
            "id": "HO1_UnseenRTT300",
            "name": "Unseen Long RTT (300ms)",
            "cap": 3.0, "buf": 50.0, "rtt": 300.0, "loss": 0.0, "pat": "constant",
            "opponent": None
        },
        {
            "id": "HO2_MicroBuffer3",
            "name": "Unseen Micro-Buffer (B=3)",
            "cap": 3.0, "buf": 3.0, "rtt": 50.0, "loss": 0.0, "pat": "constant",
            "opponent": None
        },
        {
            "id": "HO3_DeepBuffer200",
            "name": "Unseen Deep Buffer (B=200)",
            "cap": 3.0, "buf": 200.0, "rtt": 50.0, "loss": 0.0, "pat": "constant",
            "opponent": None
        },
        {
            "id": "HO4_RandomLoss1Pct",
            "name": "Unseen Wireless Loss (1.0%)",
            "cap": 3.0, "buf": 50.0, "rtt": 50.0, "loss": 0.010, "pat": "constant",
            "opponent": None
        },
        {
            "id": "HO5_UnseenComp_CUBIC",
            "name": "Unseen Opponent (TCP CUBIC)",
            "cap": 6.0, "buf": 50.0, "rtt": 50.0, "loss": 0.0, "pat": "constant",
            "opponent": "CUBIC"
        },
        {
            "id": "HO6_UnseenComp_BBR",
            "name": "Unseen Opponent (Google BBR)",
            "cap": 6.0, "buf": 50.0, "rtt": 50.0, "loss": 0.0, "pat": "constant",
            "opponent": "BBR"
        },
        {
            "id": "HO7_UnseenComp_Copa",
            "name": "Unseen Opponent (Delay-Based Copa)",
            "cap": 6.0, "buf": 50.0, "rtt": 50.0, "loss": 0.0, "pat": "constant",
            "opponent": "Copa"
        }
    ]


def run_phase1_benchmark():
    start_time = time.time()
    print("=" * 80)
    print("  PHASE 1: FAIR REFERENCE BASELINE & TRAIN TUNING BENCHMARK")
    print("=" * 80)
    
    # 1. Run Fair Tuning on TRAIN
    tuned_params = run_train_tuning_budget(budget=50, random_seed=42)
    default_params = {
        "threshold": 1.12,
        "cut_factor": 0.85,
        "standing_queue_threshold": 60,
        "competitive_backoff": 0.70,
        "aggressive_step": 0.10
    }
    
    scenarios = define_held_out_scenarios()
    seeds = list(range(42, 62)) # Exact 20 fixed evaluation seeds
    
    records = []
    paired_diffs = []
    
    print(f"\n[+] Running 20-Seed Held-Out Benchmark ({len(scenarios)} scenarios x 20 seeds = {len(scenarios)*20} runs per controller)...")
    
    for sc in scenarios:
        sc_id = sc["id"]
        sc_name = sc["name"]
        print(f"  -> Scenario: {sc_name}")
        
        diffs_u1 = []
        diffs_u05 = []
        diffs_u2 = []
        diffs_gp = []
        diffs_rtt = []
        diffs_loss = []
        
        for s in seeds:
            # Agent A: Default Reference Non-ML
            ag_def = ReferenceNonMLAgent("Default_NonML", **default_params)
            # Agent B: Tuned Reference Non-ML
            ag_tun = ReferenceNonMLAgent("Tuned_NonML", **tuned_params)
            
            # Setup opponent if multi-flow
            opp_def = None
            opp_tun = None
            if sc["opponent"] == "CUBIC":
                opp_def = CubicAgent("CUBIC_1")
                opp_tun = CubicAgent("CUBIC_2")
            elif sc["opponent"] == "BBR":
                opp_def = AuditedBBRAgent("BBR_1")
                opp_tun = AuditedBBRAgent("BBR_2")
            elif sc["opponent"] == "Copa":
                opp_def = CopaAgent("Copa_1")
                opp_tun = CopaAgent("Copa_2")
                
            # Run Default
            flow_def = [ag_def, opp_def] if opp_def else [ag_def]
            res_def = run_enhanced_sim(
                flow_def,
                total_steps=2500,
                seed=s,
                fixed_capacity=sc["cap"],
                buffer_capacity=sc["buf"],
                base_rtt=sc["rtt"],
                random_loss_rate=sc["loss"],
                bandwidth_pattern=sc["pat"]
            )
            r_def = res_def
            
            # Run Tuned
            flow_tun = [ag_tun, opp_tun] if opp_tun else [ag_tun]
            res_tun = run_enhanced_sim(
                flow_tun,
                total_steps=2500,
                seed=s,
                fixed_capacity=sc["cap"],
                buffer_capacity=sc["buf"],
                base_rtt=sc["rtt"],
                random_loss_rate=sc["loss"],
                bandwidth_pattern=sc["pat"]
            )
            r_tun = res_tun
            
            # Agent's individual goodput (flow 0)
            gp_def = r_def["flow_gps"][0]
            gp_tun = r_tun["flow_gps"][0]
            
            # Compute Utilities
            u05_def = np.log(max(gp_def, 0.01)) - 0.5 * np.log(max(r_def["avg_rtt"], 1.0))
            u10_def = np.log(max(gp_def, 0.01)) - 1.0 * np.log(max(r_def["avg_rtt"], 1.0))
            u20_def = np.log(max(gp_def, 0.01)) - 2.0 * np.log(max(r_def["avg_rtt"], 1.0))
            
            u05_tun = np.log(max(gp_tun, 0.01)) - 0.5 * np.log(max(r_tun["avg_rtt"], 1.0))
            u10_tun = np.log(max(gp_tun, 0.01)) - 1.0 * np.log(max(r_tun["avg_rtt"], 1.0))
            u20_tun = np.log(max(gp_tun, 0.01)) - 2.0 * np.log(max(r_tun["avg_rtt"], 1.0))
            
            diff_u10 = u10_tun - u10_def
            diff_u05 = u05_tun - u05_def
            diff_u20 = u20_tun - u20_def
            diff_gp = gp_tun - gp_def
            diff_rtt = r_tun["avg_rtt"] - r_def["avg_rtt"]
            diff_loss = r_tun["total_loss"] - r_def["total_loss"]
            
            diffs_u1.append(diff_u10)
            diffs_u05.append(diff_u05)
            diffs_u2.append(diff_u20)
            diffs_gp.append(diff_gp)
            diffs_rtt.append(diff_rtt)
            diffs_loss.append(diff_loss)
            
            records.append({
                "scenario_id": sc_id,
                "seed": s,
                "def_goodput": gp_def,
                "def_rtt": r_def["avg_rtt"],
                "def_loss": r_def["total_loss"],
                "def_u10": u10_def,
                "tun_goodput": gp_tun,
                "tun_rtt": r_tun["avg_rtt"],
                "tun_loss": r_tun["total_loss"],
                "tun_u10": u10_tun,
                "diff_u10": diff_u10
            })
            
        # Statistical summary per scenario
        n = len(diffs_u1)
        mean_diff_u1 = np.mean(diffs_u1)
        std_diff_u1 = np.std(diffs_u1, ddof=1)
        se_diff_u1 = std_diff_u1 / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n-1)
        ci_u1 = (mean_diff_u1 - t_crit * se_diff_u1, mean_diff_u1 + t_crit * se_diff_u1)
        
        # Paired t-test
        t_stat, p_val = stats.ttest_rel(
            [r["tun_u10"] for r in records if r["scenario_id"] == sc_id],
            [r["def_u10"] for r in records if r["scenario_id"] == sc_id]
        )
        cohen_d = mean_diff_u1 / std_diff_u1 if std_diff_u1 > 1e-6 else 0.0
        
        paired_diffs.append({
            "scenario_id": sc_id,
            "scenario_name": sc_name,
            "mean_def_u1": np.mean([r["def_u10"] for r in records if r["scenario_id"] == sc_id]),
            "mean_tun_u1": np.mean([r["tun_u10"] for r in records if r["scenario_id"] == sc_id]),
            "delta_u10": mean_diff_u1,
            "ci_lower": ci_u1[0],
            "ci_upper": ci_u1[1],
            "p_value": p_val,
            "cohen_d": cohen_d,
            "mean_diff_gp": np.mean(diffs_gp),
            "mean_diff_rtt": np.mean(diffs_rtt),
            "mean_diff_loss": np.mean(diffs_loss)
        })
        
    df_diffs = pd.DataFrame(paired_diffs)
    
    # Apply Holm-Bonferroni correction
    p_values = df_diffs["p_value"].values
    order = np.argsort(p_values)
    m = len(p_values)
    holm_significant = [False] * m
    for rank, idx in enumerate(order):
        alpha_corrected = 0.05 / (m - rank)
        if p_values[idx] < alpha_corrected:
            holm_significant[idx] = True
        else:
            break
    df_diffs["holm_significant"] = holm_significant
    
    # Save CSVs
    csv_raw = "phase1_fair_baseline_train_tuning_raw.csv"
    csv_summary = "phase1_fair_baseline_train_tuning.csv"
    pd.DataFrame(records).to_csv(csv_raw, index=False)
    df_diffs.to_csv(csv_summary, index=False)
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("  PHASE 1 COMPLETE: SUMMARY OF REAL NUMBERS")
    print(f"  Execution time: {elapsed:.1f}s")
    print("=" * 80)
    print(df_diffs[["scenario_name", "mean_def_u1", "mean_tun_u1", "delta_u10", "ci_lower", "ci_upper", "p_value", "holm_significant"]].to_string(index=False))
    print(f"\n[+] Raw per-seed runs saved to: {csv_raw}")
    print(f"[+] Paired summary saved to:   {csv_summary}")
    
    return df_diffs, tuned_params

if __name__ == "__main__":
    run_phase1_benchmark()
