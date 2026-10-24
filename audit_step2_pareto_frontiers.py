"""
audit_step2_pareto_frontiers.py

Objective Robustness and Pareto Frontier Sweep:
Addresses User Request Item 4:
"Plot goodput-vs-RTT frontiers by sweeping parameters for both ML and non-ML variants
to show whether ML dominates or only sits at another operating point."

Self-contained script that sweeps:
1. Loss Domain (2.0% Random Loss)
2. Capacity Domain (Markov Cellular Deep Fades)
3. Opponent Domain (vs TCP Reno)
Generates: 'pareto_frontier_audit_3domains.png' and 'audit_pareto_frontiers_summary.csv'
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from comprehensive_scientific_finalization import (
    ProtocolAgent,
    PacingVariantD1_CoincidentRise,
    ReferenceVegasAgent,
    AuditedBBRAgent,
    RenoAgent,
    CubicAgent
)
from phase1_fair_baseline_train_tuning import CopaAgent, ReferenceNonMLAgent

print("=" * 80)
print("  STEP 2: PARETO FRONTIER SWEEP (SELF-CONTAINED)")
print("=" * 80)

def get_pareto_frontier(gps, rtts):
    points = sorted(zip(gps, rtts), key=lambda p: (p[1], -p[0]))
    frontier = []
    max_gp_so_far = -1.0
    for gp, rtt in points:
        if gp > max_gp_so_far:
            frontier.append((gp, rtt))
            max_gp_so_far = gp
    frontier = sorted(frontier, key=lambda p: p[1])
    return frontier

# =====================================================================
# 1. LOSS DOMAIN PARETO SIMULATION
# =====================================================================
def run_loss_sweep_sim(mode, threshold=1.12, cut_factor=0.85, rf_model=None, prob_thresh=0.5, seed=42):
    rng = np.random.RandomState(seed)
    queue = 0.0
    rate = 1.05
    min_rtt = None
    rtt_hist = []
    deliv_hist = []
    recent_losses = [0] * 6
    window = []
    
    total_loss = 0
    total_proc = 0
    rtt_log = []
    
    for t in range(2500):
        r = max(0.02, rate)
        is_wireless = (rng.uniform() < 0.02)
        
        queue += r
        is_congestion = False
        if queue > 50.0:
            c_loss = queue - 50.0
            total_loss += c_loss
            queue = 50.0
            is_congestion = True
            
        if is_wireless: total_loss += 1.0
        loss_event = is_congestion or is_wireless
        
        proc = min(queue, 3.0)
        queue -= proc
        total_proc += proc
        
        curr_rtt = 50.0 + queue * 1.9 + rng.normal(0, 0.8)
        curr_rtt = max(10.0, curr_rtt)
        rtt_log.append(curr_rtt)
        
        if min_rtt is None or curr_rtt < min_rtt: min_rtt = curr_rtt
        rtt_hist.append(curr_rtt)
        deliv_hist.append(proc)
        recent_losses.append(1 if loss_event else 0)
        if len(recent_losses) > 6: recent_losses.pop(0)
        
        if not window: window = [proc] * 20
        else:
            window.append(proc)
            if len(window) > 20: window.pop(0)
            
        rtt_ratio = curr_rtt / max(min_rtt, 1e-3)
        g1 = curr_rtt - rtt_hist[-2] if len(rtt_hist) >= 2 else 0.0
        g3 = (curr_rtt - rtt_hist[-4]) / 3.0 if len(rtt_hist) >= 4 else g1
        g5 = (curr_rtt - rtt_hist[-6]) / 5.0 if len(rtt_hist) >= 6 else g1
        deliv = max(window) if window else proc
        
        if loss_event:
            is_cong_pred = True
            if mode == "ignore_all":
                is_cong_pred = False
            elif mode == "model_off":
                is_cong_pred = True
            elif mode == "rule":
                is_cong_pred = (rtt_ratio > threshold)
            elif mode == "rf" and rf_model is not None:
                feat = np.array([[rtt_ratio, g1, g3, g5, deliv, rate, float(sum(recent_losses))]])
                prob = rf_model.predict_proba(feat)[0][1] if hasattr(rf_model, "predict_proba") else rf_model.predict(feat)[0]
                is_cong_pred = (prob >= prob_thresh)
                
            factor = 0.558 if is_cong_pred else cut_factor
            rate = max(0.02, rate * factor)
        else:
            if g1 <= 0 and rtt_ratio < 1.12:
                rate += 0.05
            else:
                rate += (deliv - rate) * 0.14
                
    gp = max(0.001, (total_proc - 20.0 - total_loss * 5.0) / 2500)
    avg_rtt = np.mean(rtt_log)
    return gp, avg_rtt

print("  Training quick RF loss model for Pareto sweep...")
X_loss, y_loss = [], []
for s in [10, 11]:
    rng = np.random.RandomState(s)
    for _ in range(500):
        rtt_ratio = rng.uniform(1.0, 1.4)
        g1 = rng.normal(0, 1.0)
        g3 = rng.normal(0, 0.8)
        g5 = rng.normal(0, 0.5)
        deliv = rng.uniform(1.5, 3.0)
        rate = rng.uniform(1.5, 3.5)
        loss_cnt = rng.choice([0, 1, 2])
        is_cong = 1 if (rtt_ratio > 1.18 or (rtt_ratio > 1.10 and g1 > 0.5)) else 0
        X_loss.append([rtt_ratio, g1, g3, g5, deliv, rate, float(loss_cnt)])
        y_loss.append(is_cong)

rf_loss_sweep = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42).fit(X_loss, y_loss)

print("  Sweeping Loss Domain Pareto Frontier...")
loss_pts_nonml = []
for th in [1.03, 1.06, 1.10, 1.15, 1.20, 1.25, 1.30]:
    for cf in [0.65, 0.75, 0.85, 0.90, 0.95]:
        gps, rtts = [], []
        for s in [42, 43, 44]:
            gp, rtt = run_loss_sweep_sim("rule", threshold=th, cut_factor=cf, seed=s)
            gps.append(gp); rtts.append(rtt)
        loss_pts_nonml.append({"Type": "Non-ML Rule", "Goodput": np.mean(gps), "Avg_RTT": np.mean(rtts)})

loss_pts_ml = []
for p_th in [0.2, 0.35, 0.5, 0.65, 0.8]:
    for cf in [0.65, 0.75, 0.85, 0.90, 0.95]:
        gps, rtts = [], []
        for s in [42, 43, 44]:
            gp, rtt = run_loss_sweep_sim("rf", cut_factor=cf, rf_model=rf_loss_sweep, prob_thresh=p_th, seed=s)
            gps.append(gp); rtts.append(rtt)
        loss_pts_ml.append({"Type": "ML Classifier", "Goodput": np.mean(gps), "Avg_RTT": np.mean(rtts)})

# Ignore all loss & Model off
gps, rtts = [], []
for s in [42, 43, 44]:
    gp, rtt = run_loss_sweep_sim("ignore_all", seed=s)
    gps.append(gp); rtts.append(rtt)
loss_ignore_pt = (np.mean(gps), np.mean(rtts))

gps, rtts = [], []
for s in [42, 43, 44]:
    gp, rtt = run_loss_sweep_sim("model_off", seed=s)
    gps.append(gp); rtts.append(rtt)
loss_off_pt = (np.mean(gps), np.mean(rtts))

# =====================================================================
# 2. CAPACITY DOMAIN PARETO SIMULATION
# =====================================================================
def run_cap_sweep_sim(mode, flush_ratio=1.12, drain_factor=0.80, rf_model=None, drain_weight=0.15, seed=42):
    rng = np.random.RandomState(seed)
    total_steps = 2500
    state_caps = [1.2, 2.5, 5.0, 7.5]
    P = np.array([[0.90, 0.08, 0.02, 0.00], [0.05, 0.88, 0.07, 0.00], [0.00, 0.06, 0.88, 0.06], [0.00, 0.02, 0.08, 0.90]])
    curr_state = 1
    cap_trace = np.zeros(total_steps)
    for t in range(total_steps):
        if rng.uniform() < 0.10: curr_state = int(rng.choice(4, p=P[curr_state]))
        cap_trace[t] = state_caps[curr_state] + rng.normal(0, 0.08)
    cap_trace = np.clip(cap_trace, 0.5, 12.0)
    
    queue = 0.0
    rate = float(cap_trace[0])
    min_rtt = None
    rtt_hist = []
    deliv_hist = []
    recent_losses = [0] * 6
    window = []
    total_deliv = 0.0
    c_loss_tot = 0.0
    rtt_log = []
    ewma_c = 3.0
    pred_cap_cached = 3.0
    
    for t in range(total_steps):
        r = max(0.02, rate)
        queue += r
        c_loss = 0.0
        if queue > 50.0:
            c_loss = queue - 50.0
            c_loss_tot += c_loss
            queue = 50.0
        loss_event = (c_loss > 0)
        proc = min(queue, cap_trace[t])
        queue -= proc
        total_deliv += proc
        
        curr_rtt = 50.0 + queue * 1.9 + rng.normal(0, 0.8)
        curr_rtt = max(10.0, curr_rtt)
        rtt_log.append(curr_rtt)
        
        if min_rtt is None or curr_rtt < min_rtt: min_rtt = curr_rtt
        rtt_hist.append(curr_rtt)
        deliv_hist.append(proc)
        recent_losses.append(1 if loss_event else 0)
        if len(recent_losses) > 6: recent_losses.pop(0)
        
        if not window: window = [proc] * 20
        else:
            window.append(proc)
            if len(window) > 20: window.pop(0)
            
        rtt_ratio = curr_rtt / max(min_rtt, 1e-3)
        g1 = curr_rtt - rtt_hist[-2] if len(rtt_hist) >= 2 else 0.0
        g5 = (curr_rtt - rtt_hist[-6]) / 5.0 if len(rtt_hist) >= 6 else g1
        deliv = max(window) if window else proc
        deliv_g = deliv - (deliv_hist[-2] if len(deliv_hist) >= 2 else deliv)
        
        if loss_event:
            factor = 0.558 if rtt_ratio > 1.12 else 0.85
            rate = max(0.02, rate * factor)
            continue
            
        if mode == "persistence":
            pred_cap = deliv
            if pred_cap < rate * 0.95:
                rate = max(0.02, rate * 0.85 + pred_cap * 0.15)
                continue
        elif mode == "ewma":
            ewma_c = 0.3 * deliv + 0.7 * ewma_c
            if ewma_c < rate * 0.95:
                rate = max(0.02, rate * 0.85 + ewma_c * 0.15)
                continue
        elif mode == "rf" and rf_model is not None:
            if t % 5 == 0:
                feat = np.array([[rtt_ratio, g1, g5, deliv, rate, deliv_g, float(sum(recent_losses))]])
                pred_cap_cached = float(rf_model.predict(feat)[0])
            if pred_cap_cached < rate * 0.95:
                rate = max(0.02, rate * (1.0 - drain_weight) + pred_cap_cached * drain_weight)
                continue
        elif mode == "nonml_pacing":
            if rtt_ratio > flush_ratio and g1 > 0:
                rate = max(0.02, rate * drain_factor)
                window = [proc]
                continue
                
        if g1 <= 0 and rtt_ratio < 1.12:
            rate += 0.05
        else:
            rate += (deliv - rate) * 0.14
            
    gp = max(0.001, (total_deliv - c_loss_tot * 5.0) / total_steps)
    avg_rtt = np.mean(rtt_log)
    return gp, avg_rtt

print("  Training quick RF capacity model...")
X_cap, y_cap = [], []
for s in [10, 11]:
    rng = np.random.RandomState(s)
    for _ in range(500):
        rtt_ratio = rng.uniform(1.0, 1.4)
        g1 = rng.normal(0, 1.0)
        g5 = rng.normal(0, 0.5)
        deliv = rng.uniform(1.5, 6.0)
        rate = rng.uniform(1.5, 6.0)
        deliv_g = rng.normal(0, 0.3)
        loss_cnt = rng.choice([0, 1])
        fut_cap = deliv + deliv_g * 5.0 + rng.normal(0, 0.2)
        X_cap.append([rtt_ratio, g1, g5, deliv, rate, deliv_g, float(loss_cnt)])
        y_cap.append(max(0.5, fut_cap))

rf_cap_sweep = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42).fit(X_cap, y_cap)

print("  Sweeping Capacity Domain Pareto Frontier...")
cap_pts_nonml = []
for fr in [1.04, 1.08, 1.12, 1.16, 1.22]:
    for df in [0.65, 0.75, 0.82, 0.88]:
        gps, rtts = [], []
        for s in [42, 43, 44]:
            gp, rtt = run_cap_sweep_sim("nonml_pacing", flush_ratio=fr, drain_factor=df, seed=s)
            gps.append(gp); rtts.append(rtt)
        cap_pts_nonml.append({"Type": "Non-ML Pacing", "Goodput": np.mean(gps), "Avg_RTT": np.mean(rtts)})

cap_pts_ml = []
for dw in [0.08, 0.12, 0.16, 0.20]:
    gps, rtts = [], []
    for s in [42, 43, 44]:
        gp, rtt = run_cap_sweep_sim("rf", rf_model=rf_cap_sweep, drain_weight=dw, seed=s)
        gps.append(gp); rtts.append(rtt)
    cap_pts_ml.append({"Type": "ML Forecaster", "Goodput": np.mean(gps), "Avg_RTT": np.mean(rtts)})

gps, rtts = [], []
for s in [42, 43, 44]:
    gp, rtt = run_cap_sweep_sim("persistence", seed=s)
    gps.append(gp); rtts.append(rtt)
cap_persist_pt = (np.mean(gps), np.mean(rtts))

# =====================================================================
# 3. OPPONENT DOMAIN PARETO SIMULATION
# =====================================================================
def run_opp_sweep_sim(mode, standing_th=60, agg_step=0.10, rf_model=None, seed=42):
    rng = np.random.RandomState(seed)
    total_steps = 2500
    queue = 0.0
    r_host = 1.05
    min_rtt_h = None
    rtt_hist_h = []
    deliv_hist_h = []
    window_h = []
    standing_cnt = 0
    in_comp = False
    
    # Reno opponent
    r_opp = 1.05
    cwnd_opp = 2.0
    ssthresh_opp = 30.0
    
    deliv_h, loss_h = 0.0, 0.0
    rtt_log_h = []
    
    for t in range(total_steps):
        r_h = max(0.02, r_host)
        r_o = max(0.02, r_opp)
        
        queue += (r_h + r_o)
        loss_flag_h, loss_flag_o = False, False
        if queue > 50.0:
            excess = queue - 50.0
            frac_h = r_h / (r_h + r_o)
            loss_h += excess * frac_h
            queue = 50.0
            loss_flag_h = True
            loss_flag_o = True
            
        proc_tot = min(queue, 6.0)
        frac_serv_h = r_h / (r_h + r_o)
        proc_h = proc_tot * frac_serv_h
        proc_o = proc_tot * (1.0 - frac_serv_h)
        queue -= proc_tot
        deliv_h += proc_h
        
        curr_rtt_h = 50.0 + queue * 1.9 + rng.normal(0, 0.8)
        rtt_log_h.append(curr_rtt_h)
        
        if min_rtt_h is None or curr_rtt_h < min_rtt_h: min_rtt_h = curr_rtt_h
        rtt_hist_h.append(curr_rtt_h)
        deliv_hist_h.append(proc_h)
        
        if not window_h: window_h = [proc_h] * 20
        else:
            window_h.append(proc_h)
            if len(window_h) > 20: window_h.pop(0)
            
        rtt_ratio_h = curr_rtt_h / max(min_rtt_h, 1e-3)
        g1_h = curr_rtt_h - rtt_hist_h[-2] if len(rtt_hist_h) >= 2 else 0.0
        g5_h = (curr_rtt_h - rtt_hist_h[-6]) / 5.0 if len(rtt_hist_h) >= 6 else g1_h
        deliv_rate_h = max(window_h) if window_h else proc_h
        rate_ratio_h = r_host / max(deliv_rate_h, 0.1)
        
        if rtt_ratio_h > 1.20: standing_cnt += 1
        else: standing_cnt = max(0, standing_cnt - 2)
        
        if mode == "rule":
            in_comp = (standing_cnt > standing_th)
        elif mode == "rf" and rf_model is not None:
            if t % 5 == 0:
                feat = np.array([[rtt_ratio_h, g1_h, g5_h, rate_ratio_h, float(standing_cnt)]])
                pred_c = int(rf_model.predict(feat)[0])
                in_comp = (pred_c in [1, 2])
        elif mode == "pure_pacing":
            in_comp = False
            
        if in_comp:
            if loss_flag_h: r_host = max(0.02, r_host * 0.70)
            else: r_host += agg_step
        else:
            if loss_flag_h:
                factor = 0.558 if rtt_ratio_h > 1.12 else 0.85
                r_host = max(0.02, r_host * factor)
            else:
                if g1_h <= 0 and rtt_ratio_h < 1.12: r_host += 0.05
                else: r_host += (deliv_rate_h - r_host) * 0.14
                
        # Reno update
        if loss_flag_o:
            ssthresh_opp = max(2.0, cwnd_opp / 2.0)
            cwnd_opp = ssthresh_opp
        else:
            if cwnd_opp < ssthresh_opp: cwnd_opp += 1.0
            else: cwnd_opp += 1.0 / cwnd_opp
        r_opp = max(0.1, cwnd_opp / (curr_rtt_h / 50.0))
        
    gp_h = max(0.001, (deliv_h - loss_h * 5.0) / total_steps)
    avg_rtt = np.mean(rtt_log_h)
    return gp_h, avg_rtt

print("  Training quick RF opponent detector...")
X_opp, y_opp = [], []
for s in [10, 11]:
    rng = np.random.RandomState(s)
    for _ in range(500):
        rtt_ratio = rng.uniform(1.0, 1.8)
        g1 = rng.normal(0, 1.0)
        g5 = rng.normal(0, 0.5)
        rate_ratio = rng.uniform(0.5, 2.5)
        st_steps = rng.uniform(0, 100)
        cls_target = 1 if (st_steps > 40 and rtt_ratio > 1.25) else 0
        X_opp.append([rtt_ratio, g1, g5, rate_ratio, st_steps])
        y_opp.append(cls_target)

rf_opp_sweep = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42).fit(X_opp, y_opp)

print("  Sweeping Opponent Domain Pareto Frontier...")
opp_pts_nonml = []
for sq in [30, 50, 70, 90, 120]:
    for step in [0.05, 0.08, 0.12, 0.16]:
        gps, rtts = [], []
        for s in [42, 43, 44]:
            gp, rtt = run_opp_sweep_sim("rule", standing_th=sq, agg_step=step, seed=s)
            gps.append(gp); rtts.append(rtt)
        opp_pts_nonml.append({"Type": "Non-ML Rule", "Goodput": np.mean(gps), "Avg_RTT": np.mean(rtts)})

opp_pts_ml = []
for step in [0.05, 0.08, 0.12, 0.16]:
    gps, rtts = [], []
    for s in [42, 43, 44]:
        gp, rtt = run_opp_sweep_sim("rf", agg_step=step, rf_model=rf_opp_sweep, seed=s)
        gps.append(gp); rtts.append(rtt)
    opp_pts_ml.append({"Type": "ML Detector", "Goodput": np.mean(gps), "Avg_RTT": np.mean(rtts)})

gps, rtts = [], []
for s in [42, 43, 44]:
    gp, rtt = run_opp_sweep_sim("pure_pacing", seed=s)
    gps.append(gp); rtts.append(rtt)
opp_purepace_pt = (np.mean(gps), np.mean(rtts))

# =====================================================================
# 4. PLOTTING AND EXPORTING PARETO FRONTIERS
# =====================================================================
print("  Rendering 3-Panel Pareto Visualization...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

# Panel 1: Loss
ax1 = axes[0]
df_loss_nm = pd.DataFrame(loss_pts_nonml)
df_loss_m = pd.DataFrame(loss_pts_ml)
ax1.scatter(df_loss_nm["Avg_RTT"], df_loss_nm["Goodput"], color="#1f77b4", alpha=0.45, label="Non-ML Sweep Grid", s=35)
ax1.scatter(df_loss_m["Avg_RTT"], df_loss_m["Goodput"], color="#d62728", alpha=0.85, label="ML Classifier Grid", s=45, marker="^")
ax1.scatter([loss_ignore_pt[1]], [loss_ignore_pt[0]], color="#2ca02c", s=90, marker="*", label="Ignore-All-Loss")
ax1.scatter([loss_off_pt[1]], [loss_off_pt[0]], color="#7f7f7f", s=70, marker="x", label="Model Off")

nf_loss = get_pareto_frontier(df_loss_nm["Goodput"].values, df_loss_nm["Avg_RTT"].values)
mf_loss = get_pareto_frontier(df_loss_m["Goodput"].values, df_loss_m["Avg_RTT"].values)
ax1.plot([p[1] for p in nf_loss], [p[0] for p in nf_loss], color="#1f77b4", linestyle="--", linewidth=1.8, label="Non-ML Pareto Frontier")
ax1.plot([p[1] for p in mf_loss], [p[0] for p in mf_loss], color="#d62728", linestyle="-", linewidth=2.0, label="ML Pareto Frontier")
ax1.set_title("Domain 1: Loss Classification (2.0% Loss)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Average RTT (ms)", fontsize=10)
ax1.set_ylabel("Goodput (pkts/step)", fontsize=10)
ax1.grid(True, linestyle=":", alpha=0.5)
ax1.legend(fontsize=8, loc="lower right")

# Panel 2: Capacity
ax2 = axes[1]
df_cap_nm = pd.DataFrame(cap_pts_nonml)
df_cap_m = pd.DataFrame(cap_pts_ml)
ax2.scatter(df_cap_nm["Avg_RTT"], df_cap_nm["Goodput"], color="#1f77b4", alpha=0.45, label="Non-ML Sweep Grid", s=35)
ax2.scatter(df_cap_m["Avg_RTT"], df_cap_m["Goodput"], color="#d62728", alpha=0.85, label="ML Forecaster Grid", s=45, marker="^")
ax2.scatter([cap_persist_pt[1]], [cap_persist_pt[0]], color="#ff7f0e", s=80, marker="D", label="Persistence Baseline")

nf_cap = get_pareto_frontier(df_cap_nm["Goodput"].values, df_cap_nm["Avg_RTT"].values)
mf_cap = get_pareto_frontier(df_cap_m["Goodput"].values, df_cap_m["Avg_RTT"].values)
ax2.plot([p[1] for p in nf_cap], [p[0] for p in nf_cap], color="#1f77b4", linestyle="--", linewidth=1.8, label="Non-ML Frontier")
ax2.plot([p[1] for p in mf_cap], [p[0] for p in mf_cap], color="#d62728", linestyle="-", linewidth=2.0, label="ML Frontier")
ax2.set_title("Domain 2: Capacity Adaptation (Markov Fade)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Average RTT (ms)", fontsize=10)
ax2.set_ylabel("Goodput (pkts/step)", fontsize=10)
ax2.grid(True, linestyle=":", alpha=0.5)
ax2.legend(fontsize=8, loc="lower right")

# Panel 3: Opponent
ax3 = axes[2]
df_opp_nm = pd.DataFrame(opp_pts_nonml)
df_opp_m = pd.DataFrame(opp_pts_ml)
ax3.scatter(df_opp_nm["Avg_RTT"], df_opp_nm["Goodput"], color="#1f77b4", alpha=0.45, label="Non-ML Sweep Grid", s=35)
ax3.scatter(df_opp_m["Avg_RTT"], df_opp_m["Goodput"], color="#d62728", alpha=0.85, label="ML Detector Grid", s=45, marker="^")
ax3.scatter([opp_purepace_pt[1]], [opp_purepace_pt[0]], color="#7f7f7f", s=80, marker="x", label="Model Off (Pure Pacing)")

nf_opp = get_pareto_frontier(df_opp_nm["Goodput"].values, df_opp_nm["Avg_RTT"].values)
mf_opp = get_pareto_frontier(df_opp_m["Goodput"].values, df_opp_m["Avg_RTT"].values)
ax3.plot([p[1] for p in nf_opp], [p[0] for p in nf_opp], color="#1f77b4", linestyle="--", linewidth=1.8, label="Non-ML Frontier")
ax3.plot([p[1] for p in mf_opp], [p[0] for p in mf_opp], color="#d62728", linestyle="-", linewidth=2.0, label="ML Frontier")
ax3.set_title("Domain 3: Competition vs. TCP Reno", fontsize=11, fontweight="bold")
ax3.set_xlabel("Average RTT (ms)", fontsize=10)
ax3.set_ylabel("Host Goodput (pkts/step)", fontsize=10)
ax3.grid(True, linestyle=":", alpha=0.5)
ax3.legend(fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig("pareto_frontier_audit_3domains.png", dpi=300)
plt.close()
print("  [OK] Saved Pareto figure to 'pareto_frontier_audit_3domains.png'")

df_summary = pd.concat([
    df_loss_nm.assign(Domain="Loss Classification"),
    df_loss_m.assign(Domain="Loss Classification"),
    df_cap_nm.assign(Domain="Capacity Adaptation"),
    df_cap_m.assign(Domain="Capacity Adaptation"),
    df_opp_nm.assign(Domain="Opponent Detection"),
    df_opp_m.assign(Domain="Opponent Detection")
], ignore_index=True)

df_summary.to_csv("audit_pareto_frontiers_summary.csv", index=False)
print("  [OK] Saved sweep data to 'audit_pareto_frontiers_summary.csv'")
print("=" * 80)
print("  STEP 2 COMPLETE.")
print("=" * 80)
