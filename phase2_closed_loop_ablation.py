"""
ML-Based Predictive TCP Congestion Control
Script: phase2_closed_loop_ablation.py

Phase 2: Closed-Loop Predictor Ablation & Gate 1 Verification
1. Evaluates 4 Predictors under identical controller:
   (a) Persistence: ratio_hat = current_ratio
   (b) Linear Extrapolation: ratio_hat = current_ratio + 5 * grad_1 / min_rtt
   (c) Random Forest Regressor (trained on TRAIN families)
   (d) No Predictor (Phase 1 Fast-Expiry Pacing baseline)
2. Sweeps threshold in [1.05..1.30] and cut in [0.6..0.9] on TRAIN families.
3. Evaluates selected best configs on HELD-OUT families across 20 fixed seeds (95% CI).
4. Computes paired per-seed differences vs. No Predictor and vs. Linear Extrapolation.
5. Scores every controller with utility U_d = log(GP) - d*log(RTT) for d in [0.5, 1, 2] and GP/RTT.
6. Offline predictor evaluation (MAE, R2) on time-ordered held-out test traces.
7. Checks Gate 1: If RF does not beat linear extrapolation and persistence in closed loop,
   triggers Gate 1 exit to Phase 6.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from phase1_baselines_and_pacing import run_sim, NoModelMaxFilterFastExpiry
from advanced_audit_and_regression import extract_native_trees_reg, fast_eval_reg, AuditedBBRAgent, ReferenceVegasAgent, RenoAgent, CubicAgent, RuleBasedAgent


# =====================================================================
# SECTION 1: PREDICTOR AGENTS (IDENTICAL CONTROLLER ARCHITECTURE)
# =====================================================================
class BasePredictorAgent(NoModelMaxFilterFastExpiry):
    def __init__(self, name, mode="none", threshold=1.12, cut_factor=0.75, native_reg_trees=None):
        super().__init__(name)
        self.mode = mode
        self.threshold = threshold
        self.cut_factor = cut_factor
        self.native_reg_trees = native_reg_trees

    def predict_future_ratio(self):
        curr_rtt = self.rtt_history[-1]
        rtt_ratio = curr_rtt / max(self.min_rtt, 1e-3)
        grad_1 = curr_rtt - self.rtt_history[-2]
        
        if self.mode == "none":
            return 1.0
        elif self.mode == "persistence":
            return rtt_ratio
        elif self.mode == "linear":
            return rtt_ratio + (5.0 * grad_1 / max(self.min_rtt, 1e-3))
        elif self.mode == "rf":
            grad_3 = (curr_rtt - self.rtt_history[-4]) / 3.0
            grad_5 = (curr_rtt - self.rtt_history[-6]) / 5.0
            loss_flag = 1.0 if sum(self.recent_losses) > 0 else 0.0
            deliv_target = max(self.fast_history) if self.fast_history else self.est_delivery_rate
            feat = [rtt_ratio, grad_1, grad_3, grad_5, deliv_target, self.rate, loss_flag]
            return fast_eval_reg(self.native_reg_trees, feat)
        return 1.0

    def update_rate(self, loss_event):
        min_rate = self.get_min_rate()
        if loss_event:
            self.rate = max(min_rate, self.rate * 0.558)
            return
            
        curr_rtt = self.rtt_history[-1]
        rtt_ratio = curr_rtt / max(self.min_rtt, 1e-3)
        grad_1 = curr_rtt - self.rtt_history[-2]
        target_pacing = max(self.fast_history) if self.fast_history else self.est_delivery_rate
        
        # Predictor-guided rate reduction
        if self.mode != "none":
            pred_ratio = self.predict_future_ratio()
            if pred_ratio >= self.threshold and grad_1 > 0:
                self.throttle_count += 1
                self.rate = max(min_rate, self.rate * self.cut_factor)
                return
                
        # Phase 1 Pacing Baseline rule
        if grad_1 <= 0 and rtt_ratio < 1.12:
            self.rate += 0.05
        else:
            self.rate += (target_pacing - self.rate) * 0.14


# =====================================================================
# SECTION 2: SYNTHESIZE TRAIN DATA & FIT RF REGRESSOR
# =====================================================================
def generate_train_data_and_fit_rf():
    print("  [Step 1] Synthesizing time-ordered TRAIN traces (capacity 3-5, buffer 50, lossless)...", flush=True)
    features = ['rtt_ratio', 'rtt_gradient_1', 'rtt_gradient_3', 'rtt_gradient_5', 'delivery_rate', 'send_rate', 'loss_flag_5']
    train_runs = []
    rng = np.random.RandomState(42)
    
    # Generate 35 diverse trajectories (50,000 steps) strictly within TRAIN family specs
    for i in range(35):
        c_base = rng.uniform(3.0, 5.0)
        buf = 50.0
        base_rtt = rng.uniform(30.0, 80.0)
        agent = NoModelMaxFilterFastExpiry("Collector")
        rtt_h = [base_rtt] * 6
        recent_l = [0] * 6
        queue = 0.0
        min_r = base_rtt
        
        feats, rtts = [], []
        for t in range(1500):
            cap = c_base + np.sin(t / 60.0) * 0.8
            rtt = rtt_h[-1]
            min_r = min(min_r, rtt)
            r_ratio = rtt / max(min_r, 1e-3)
            g1 = rtt - rtt_h[-2]
            g3 = (rtt - rtt_h[-4]) / 3.0
            g5 = (rtt - rtt_h[-6]) / 5.0
            l_flag = 1.0 if sum(recent_l) > 0 else 0.0
            deliv_tgt = max(agent.fast_history)
            
            feat = [r_ratio, g1, g3, g5, deliv_tgt, agent.rate, l_flag]
            feats.append(feat)
            rtts.append(rtt)
            agent.update_rate(recent_l[-1] > 0)
            
            queue += max(agent.get_min_rate(), agent.rate)
            drop = 0.0
            if queue > buf:
                drop = queue - buf
                queue = buf
                agent.rate *= 0.558
            proc = min(queue, cap)
            queue -= proc
            new_rtt = base_rtt + (queue * 1.9)
            rtt_h.append(new_rtt)
            if len(rtt_h) > 10: rtt_h.pop(0)
            recent_l.append(1 if drop > 0 else 0)
            if len(recent_l) > 5: recent_l.pop(0)
            agent.observe_ack(new_rtt, drop > 0, proc)
            
        T = len(rtts)
        t_target = np.zeros(T)
        for t_idx in range(T):
            fut_idx = min(t_idx + 5, T - 1)
            t_target[t_idx] = rtts[fut_idx] / max(min_r, 1e-3)
            
        df_run = pd.DataFrame(feats, columns=features)
        df_run['target_ratio_h5'] = t_target
        train_runs.append(df_run)
        
    df_train = pd.concat(train_runs, ignore_index=True)
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    rf_reg.fit(df_train[features], df_train['target_ratio_h5'])
    native_trees = extract_native_trees_reg(rf_reg)
    print(f"  [OK] RF Regressor fitted on {len(df_train)} samples across TRAIN family.", flush=True)
    return rf_reg, native_trees, df_train


# =====================================================================
# SECTION 3: OFFLINE EVALUATION ON TIME-ORDERED HELD-OUT TEST TRACE
# =====================================================================
def evaluate_offline_held_out(rf_reg):
    print("\n  [Step 2] Offline Predictor Evaluation on Time-Ordered HELD-OUT Traces...", flush=True)
    features = ['rtt_ratio', 'rtt_gradient_1', 'rtt_gradient_3', 'rtt_gradient_5', 'delivery_rate', 'send_rate', 'loss_flag_5']
    # Synthesize held-out test data (random walk 1-9 pkts/step, buffer 5 pkts, 2% loss)
    test_runs = []
    rng = np.random.RandomState(999)
    for i in range(10):
        c_base = 3.0
        buf = 5.0  # Held-out shallow buffer
        base_rtt = 50.0
        agent = NoModelMaxFilterFastExpiry("TestAgent")
        rtt_h = [base_rtt] * 6
        recent_l = [0] * 6
        queue = 0.0
        min_r = base_rtt
        feats, rtts = [], []
        c = c_base
        for t in range(1000):
            c = np.clip(c + rng.normal(0, 0.10), 1.0, 9.0)
            rtt = rtt_h[-1]
            min_r = min(min_r, rtt)
            r_ratio = rtt / max(min_r, 1e-3)
            g1 = rtt - rtt_h[-2]
            g3 = (rtt - rtt_h[-4]) / 3.0
            g5 = (rtt - rtt_h[-6]) / 5.0
            l_flag = 1.0 if sum(recent_l) > 0 else 0.0
            deliv_tgt = max(agent.fast_history)
            
            feat = [r_ratio, g1, g3, g5, deliv_tgt, agent.rate, l_flag]
            feats.append(feat)
            rtts.append(rtt)
            agent.update_rate(recent_l[-1] > 0)
            
            queue += max(agent.get_min_rate(), agent.rate)
            drop = 0.0
            if queue > buf:
                drop = queue - buf
                queue = buf
                agent.rate *= 0.558
            proc = min(queue, c)
            queue -= proc
            new_rtt = base_rtt + (buf * 1.9) if drop > 0 else base_rtt + (queue * 1.9)
            rtt_h.append(new_rtt)
            if len(rtt_h) > 10: rtt_h.pop(0)
            recent_l.append(1 if drop > 0 else 0)
            if len(recent_l) > 5: recent_l.pop(0)
            agent.observe_ack(new_rtt, drop > 0, proc)
            
        T = len(rtts)
        t_target = np.zeros(T)
        for t_idx in range(T):
            fut_idx = min(t_idx + 5, T - 1)
            t_target[t_idx] = rtts[fut_idx] / max(min_r, 1e-3)
            
        df_t = pd.DataFrame(feats, columns=features)
        df_t['target_ratio_h5'] = t_target
        test_runs.append(df_t)
        
    df_test = pd.concat(test_runs, ignore_index=True)
    y_true = df_test['target_ratio_h5'].values
    y_persist = df_test['rtt_ratio'].values
    y_linear = df_test['rtt_ratio'].values + (5.0 * df_test['rtt_gradient_1'].values / 50.0)
    y_rf = rf_reg.predict(df_test[features])
    
    mae_p, r2_p = mean_absolute_error(y_true, y_persist), r2_score(y_true, y_persist)
    mae_l, r2_l = mean_absolute_error(y_true, y_linear), r2_score(y_true, y_linear)
    mae_rf, r2_rf = mean_absolute_error(y_true, y_rf), r2_score(y_true, y_rf)
    
    print(f"    Persistence:          MAE = {mae_p:.4f} | R2 = {r2_p:.4f}", flush=True)
    print(f"    Linear Extrapolation: MAE = {mae_l:.4f} | R2 = {r2_l:.4f}", flush=True)
    print(f"    Random Forest:        MAE = {mae_rf:.4f} | R2 = {r2_rf:.4f}", flush=True)
    return {'p': (mae_p, r2_p), 'l': (mae_l, r2_l), 'rf': (mae_rf, r2_rf)}


# =====================================================================
# SECTION 4: PARAMETER SWEEP ON TRAIN FAMILIES & SELECTION
# =====================================================================
def sweep_and_tune_on_train(native_rf_trees):
    print("\n  [Step 3] Parameter Sweep on TRAIN Families (th: 1.05..1.30, cut: 0.6..0.9)...", flush=True)
    train_seeds = [101, 102, 103, 104, 105]
    thresholds = [1.05, 1.10, 1.15, 1.20, 1.25, 1.30]
    cuts = [0.60, 0.70, 0.80, 0.90]
    
    best_configs = {}
    for mode in ["persistence", "linear", "rf"]:
        best_score = -999.0
        best_th, best_cut = 1.15, 0.75
        for th in thresholds:
            for cut in cuts:
                scores = []
                for s in train_seeds:
                    agent = BasePredictorAgent("TunAgent", mode=mode, threshold=th, cut_factor=cut, native_reg_trees=native_rf_trees)
                    res = run_sim([agent], total_steps=2000, seed=s, fixed_capacity=3.0, buffer_capacity=50.0)
                    # Objective score: U_1.0 = log(GP) - 1.0 * log(RTT)
                    u1 = np.log(max(res['goodput'], 0.01)) - 1.0 * np.log(max(res['avg_rtt'], 1.0))
                    scores.append(u1)
                mean_u = np.mean(scores)
                if mean_u > best_score:
                    best_score = mean_u
                    best_th, best_cut = th, cut
        best_configs[mode] = (best_th, best_cut, best_score)
        print(f"    Best config for [{mode}]: threshold = {best_th}, cut = {best_cut} (TRAIN U_1.0 = {best_score:.4f})", flush=True)
    return best_configs


# =====================================================================
# SECTION 5: HELD-OUT CLOSED-LOOP EVALUATION ACROSS 20 SEEDS
# =====================================================================
def evaluate_closed_loop_held_out(best_configs, native_rf_trees, seeds):
    print("\n  [Step 4] Closed-Loop Evaluation on HELD-OUT Families Across 20 Seeds...", flush=True)
    scenarios = [
        {"id": "HO1", "name": "Variable BW Walk (1-9)", "cap": 3.0, "buf": 50.0, "loss": 0.0, "pat": "random_walk", "flows": 1},
        {"id": "HO2", "name": "Random Loss 0.1%", "cap": 3.0, "buf": 50.0, "loss": 0.001, "pat": "constant", "flows": 1},
        {"id": "HO3", "name": "Random Loss 2.0%", "cap": 3.0, "buf": 50.0, "loss": 0.020, "pat": "constant", "flows": 1},
        {"id": "HO4", "name": "Shallow Buffer B=5", "cap": 3.0, "buf": 5.0, "loss": 0.0, "pat": "constant", "flows": 1},
        {"id": "HO5", "name": "Ultra-Shallow B=3", "cap": 3.0, "buf": 3.0, "loss": 0.0, "pat": "constant", "flows": 1},
        {"id": "HO6", "name": "Multi-Flow 2 Flows", "cap": 4.0, "buf": 50.0, "loss": 0.0, "pat": "constant", "flows": 2},
        {"id": "HO7", "name": "Multi-Flow 8 Flows", "cap": 6.0, "buf": 50.0, "loss": 0.0, "pat": "constant", "flows": 8}
    ]
    
    controllers = {
        'No Predictor': ('none', 1.15, 0.75),
        'Persistence': ('persistence', best_configs['persistence'][0], best_configs['persistence'][1]),
        'Linear Extrapolation': ('linear', best_configs['linear'][0], best_configs['linear'][1]),
        'Random Forest': ('rf', best_configs['rf'][0], best_configs['rf'][1])
    }
    
    N = len(seeds)
    all_eval_records = []
    paired_diff_records = []
    utility_records = []
    
    # Store per-seed utility arrays for Gate 1 statistical test
    rf_u1_all, lin_u1_all, per_u1_all = [], [], []
    
    for sc in scenarios:
        sc_data = {}
        for c_name, (mode, th, cut) in controllers.items():
            runs = []
            for s in seeds:
                agents = [BasePredictorAgent(f"{c_name}_{i}", mode=mode, threshold=th, cut_factor=cut, native_reg_trees=native_rf_trees) for i in range(sc['flows'])]
                res = run_sim(agents, total_steps=3000, seed=s, buffer_capacity=sc['buf'],
                              random_loss_rate=sc['loss'], bandwidth_pattern=sc['pat'],
                              fixed_capacity=sc['cap'], stagger_flows=(sc['flows'] > 1))
                runs.append(res)
            sc_data[c_name] = runs
            
            gp = np.mean([r['goodput'] for r in runs])
            gp_ci = 1.96 * np.std([r['goodput'] for r in runs]) / np.sqrt(N)
            loss = np.mean([r['loss'] for r in runs])
            rtt = np.mean([r['avg_rtt'] for r in runs])
            rtt_ci = 1.96 * np.std([r['avg_rtt'] for r in runs]) / np.sqrt(N)
            jfi = np.mean([r['jfi'] for r in runs])
            actions = np.mean([r.get('throttle_fires', 0.0) for r in runs])
            
            # Utility scores
            u_05 = np.mean([np.log(max(r['goodput'], 0.01)) - 0.5 * np.log(max(r['avg_rtt'], 1.0)) for r in runs])
            u_10 = np.mean([np.log(max(r['goodput'], 0.01)) - 1.0 * np.log(max(r['avg_rtt'], 1.0)) for r in runs])
            u_20 = np.mean([np.log(max(r['goodput'], 0.01)) - 2.0 * np.log(max(r['avg_rtt'], 1.0)) for r in runs])
            power = np.mean([r['goodput'] / max(r['avg_rtt'], 1.0) * 1000 for r in runs])
            
            all_eval_records.append({
                'Scenario': sc['id'],
                'Scenario_Name': sc['name'],
                'Controller': c_name,
                'Goodput': f"{gp:.4f} +/- {gp_ci:.4f}",
                'Loss': f"{loss:.1f}",
                'Avg RTT (ms)': f"{rtt:.1f} +/- {rtt_ci:.1f}",
                'JFI': f"{jfi:.3f}" if sc['flows'] > 1 else "1.000",
                'Actions': f"{actions:.1f}",
                'U_0.5': f"{u_05:.4f}",
                'U_1.0': f"{u_10:.4f}",
                'U_2.0': f"{u_20:.4f}",
                'Power (GP/RTT*1k)': f"{power:.2f}",
                'gp_val': gp, 'loss_val': loss, 'rtt_val': rtt, 'u1_val': u_10
            })
            
        # Paired per-seed differences vs. No Predictor and vs. Linear Extrapolation
        runs_no_pred = sc_data['No Predictor']
        runs_rf = sc_data['Random Forest']
        runs_lin = sc_data['Linear Extrapolation']
        runs_per = sc_data['Persistence']
        
        diff_gp_rf_vs_none = [runs_rf[i]['goodput'] - runs_no_pred[i]['goodput'] for i in range(N)]
        diff_rtt_rf_vs_none = [runs_rf[i]['avg_rtt'] - runs_no_pred[i]['avg_rtt'] for i in range(N)]
        
        diff_u1_rf_vs_lin = [
            (np.log(max(runs_rf[i]['goodput'], 0.01)) - np.log(max(runs_rf[i]['avg_rtt'], 1.0))) -
            (np.log(max(runs_lin[i]['goodput'], 0.01)) - np.log(max(runs_lin[i]['avg_rtt'], 1.0)))
            for i in range(N)
        ]
        diff_u1_rf_vs_per = [
            (np.log(max(runs_rf[i]['goodput'], 0.01)) - np.log(max(runs_rf[i]['avg_rtt'], 1.0))) -
            (np.log(max(runs_per[i]['goodput'], 0.01)) - np.log(max(runs_per[i]['avg_rtt'], 1.0)))
            for i in range(N)
        ]
        
        rf_u1_all.extend([np.log(max(r['goodput'], 0.01)) - np.log(max(r['avg_rtt'], 1.0)) for r in runs_rf])
        lin_u1_all.extend([np.log(max(r['goodput'], 0.01)) - np.log(max(r['avg_rtt'], 1.0)) for r in runs_lin])
        per_u1_all.extend([np.log(max(r['goodput'], 0.01)) - np.log(max(r['avg_rtt'], 1.0)) for r in runs_per])
        
        def fmt_p(arr):
            m = np.mean(arr)
            ci = 1.96 * np.std(arr) / np.sqrt(len(arr))
            sig = "*" if (m - ci > 0 or m + ci < 0) else "ns"
            return f"{m:+.4f} +/- {ci:.4f} ({sig})"
            
        paired_diff_records.append({
            'Scenario': sc['id'],
            'dGP (RF - NoPred)': fmt_p(diff_gp_rf_vs_none),
            'dRTT (RF - NoPred)': fmt_p(diff_rtt_rf_vs_none),
            'dU_1.0 (RF - Linear)': fmt_p(diff_u1_rf_vs_lin),
            'dU_1.0 (RF - Persist)': fmt_p(diff_u1_rf_vs_per)
        })
        
    df_eval = pd.DataFrame(all_eval_records)
    df_paired = pd.DataFrame(paired_diff_records)
    
    print("\n--- HELD-OUT PERFORMANCE & UTILITY SCORES ---")
    print(df_eval[['Scenario', 'Controller', 'Goodput', 'Loss', 'Avg RTT (ms)', 'U_1.0', 'Power (GP/RTT*1k)']].to_string(index=False))
    
    print("\n--- PAIRED DIFFERENCES & OBJECTIVE ADVANTAGE ---")
    print(df_paired.to_string(index=False))
    
    df_eval.to_csv("phase2_held_out_evaluation.csv", index=False)
    df_paired.to_csv("phase2_paired_differences.csv", index=False)
    
    # -------------------------------------------------------------
    # SECTION 6: GATE 1 DECISION LOGIC
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  GATE 1 EVALUATION: DOES RANDOM FOREST BEAT LINEAR & PERSISTENCE?")
    print("=" * 80)
    diff_rf_vs_lin_all = np.array(rf_u1_all) - np.array(lin_u1_all)
    diff_rf_vs_per_all = np.array(rf_u1_all) - np.array(per_u1_all)
    
    m_lin = np.mean(diff_rf_vs_lin_all)
    ci_lin = 1.96 * np.std(diff_rf_vs_lin_all) / np.sqrt(len(diff_rf_vs_lin_all))
    
    m_per = np.mean(diff_rf_vs_per_all)
    ci_per = 1.96 * np.std(diff_rf_vs_per_all) / np.sqrt(len(diff_rf_vs_per_all))
    
    print(f"  Overall dU_1.0 (RF vs. Linear Extrapolation): {m_lin:+.4f} +/- {ci_lin:.4f} (95% CI: [{m_lin - ci_lin:.4f}, {m_lin + ci_lin:.4f}])")
    print(f"  Overall dU_1.0 (RF vs. Persistence):          {m_per:+.4f} +/- {ci_per:.4f} (95% CI: [{m_per - ci_per:.4f}, {m_per + ci_per:.4f}])")
    
    # Gate 1 Condition: Paired CI on objective score must EXCLUDE ZERO in its favor (i.e. lower bound > 0)
    rf_beats_linear = (m_lin - ci_lin) > 0.0
    rf_beats_persist = (m_per - ci_per) > 0.0
    
    gate_1_passed = rf_beats_linear and rf_beats_persist
    
    if not gate_1_passed:
        print("\n  [GATE 1 VERDICT]:")
        print("  ==========================================================================")
        print("  ML predictor adds no measurable benefit")
        print("  ==========================================================================")
        print("  Statistical Reason: Random Forest closed-loop objective score does not")
        print("  significantly beat Linear Extrapolation / Persistence on held-out families.")
        print("  Triggering GATE 1: SKIPPING Phases 3, 4, 5 -> Proceeding directly to Phase 6.")
    else:
        print("\n  [GATE 1 VERDICT]: Random Forest passed Gate 1! Proceeding to Phase 3.")
        
    return gate_1_passed, df_eval, df_paired


def main():
    print("=" * 80)
    print("  PHASE 2: CLOSED-LOOP PREDICTOR ABLATION")
    print("=" * 80)
    
    seeds = list(range(42, 62))
    rf_reg, native_rf_trees, df_train = generate_train_data_and_fit_rf()
    offline_metrics = evaluate_offline_held_out(rf_reg)
    best_configs = sweep_and_tune_on_train(native_rf_trees)
    gate_1_passed, df_eval, df_paired = evaluate_closed_loop_held_out(best_configs, native_rf_trees, seeds)


if __name__ == "__main__":
    main()
