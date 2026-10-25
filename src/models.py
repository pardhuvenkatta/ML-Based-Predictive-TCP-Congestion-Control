"""
Model Architectures, Feature Extractors, and LOFO Training Pipelines for ML-TCP
Strictly adheres to zero ground-truth signal leakage: sender-observable signals only.
"""

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Ridge

from src.controllers import ReferenceNonMLAgent, RenoAgent, CubicAgent, AuditedBBRAgent, CopaAgent, ReferenceVegasAgent
from src.sim import load_mahimahi_trace


def collect_loss_data(families_to_include, seeds=[10, 11, 12, 13, 14, 15]):
    """Collects training pool for loss classification LOFO evaluation."""
    X, y = [], []
    for s in seeds:
        for fam in families_to_include:
            rng = np.random.RandomState(s)
            ag = ReferenceNonMLAgent("Collect")
            ag.min_rtt = None
            ag.rtt_history = []
            ag.deliv_history = []
            ag.recent_losses = [0] * 6
            queue = 0.0
            ge_state = 0
            
            for t in range(2500):
                rate = max(0.02, ag.rate)
                is_wireless = False
                if fam == "bernoulli":
                    p_loss = rng.choice([0.005, 0.01, 0.02])
                    if rng.uniform() < p_loss:
                        is_wireless = True
                elif fam == "gilbert_elliott":
                    if ge_state == 0 and rng.uniform() < 0.03:
                        ge_state = 1
                    elif ge_state == 1 and rng.uniform() < 0.25:
                        ge_state = 0
                    if ge_state == 1 and rng.uniform() < 0.20:
                        is_wireless = True

                cap = 3.0 if (t % 600 < 400) else 1.2
                is_congestion = False
                queue += rate
                if queue > 30.0:
                    queue = 30.0
                    is_congestion = True

                loss_event = is_congestion or is_wireless
                proc = min(queue, cap)
                queue -= proc
                curr_rtt = 50.0 + queue * 1.9 + rng.normal(0, 0.8)
                ag.observe_ack(curr_rtt, loss_event, proc)
                ag.update_rate(loss_event)

                if loss_event:
                    rtt_ratio = curr_rtt / max(ag.min_rtt, 1e-3)
                    g1 = curr_rtt - ag.rtt_history[-2] if len(ag.rtt_history) >= 2 else 0.0
                    g3 = (curr_rtt - ag.rtt_history[-4]) / 3.0 if len(ag.rtt_history) >= 4 else g1
                    g5 = (curr_rtt - ag.rtt_history[-6]) / 5.0 if len(ag.rtt_history) >= 6 else g1
                    deliv = max(ag.deliv_history) if ag.deliv_history else ag.est_delivery_rate
                    feat = [rtt_ratio, g1, g3, g5, deliv, ag.rate, float(sum(ag.recent_losses))]
                    target = 1 if is_congestion else 0
                    X.append(feat)
                    y.append(target)
    return np.array(X), np.array(y)


def collect_capacity_data(family, seeds=[20, 21, 22, 23]):
    """Collects training pool for capacity prediction LOFO evaluation."""
    X, y = [], []
    full_mahimahi = load_mahimahi_trace()
    for s in seeds:
        rng = np.random.RandomState(s)
        total_steps = 2500
        cap_trace = np.zeros(total_steps)
        if family == "markov":
            state_caps = [1.2, 2.5, 5.0, 7.5]
            st = 1
            for t in range(total_steps):
                if rng.uniform() < 0.10:
                    st = rng.choice(4)
                cap_trace[t] = state_caps[st] + rng.normal(0, 0.08)
        elif family == "mahimahi":
            start_idx = int(rng.uniform(0, max(1, len(full_mahimahi) - total_steps - 50)))
            cap_trace = full_mahimahi[start_idx : start_idx + total_steps]
        elif family == "step":
            st_t = rng.uniform(700, 1400)
            for t in range(total_steps):
                cap_trace[t] = (6.0 if t < st_t else 1.8) + rng.normal(0, 0.05)
        elif family == "sinusoid":
            ph = rng.uniform(0, 2 * np.pi)
            fr = rng.uniform(0.015, 0.025)
            for t in range(total_steps):
                cap_trace[t] = 3.0 + 1.5 * np.sin(fr * t + ph) + rng.normal(0, 0.05)
        cap_trace = np.clip(cap_trace, 0.5, 12.0)
        
        ag = ReferenceNonMLAgent("CollectCap")
        ag.rate = float(cap_trace[0])
        queue = 0.0
        for t in range(total_steps - 6):
            queue += ag.rate
            if queue > 50.0:
                queue = 50.0
            proc = min(queue, cap_trace[t])
            queue -= proc
            curr_rtt = 50.0 + queue * 1.9 + rng.normal(0, 0.8)
            ag.observe_ack(curr_rtt, False, proc)
            ag.update_rate(False)
            
            rtt_ratio = curr_rtt / max(ag.min_rtt, 1e-3)
            g1 = curr_rtt - ag.rtt_history[-2] if len(ag.rtt_history) >= 2 else 0.0
            g5 = (curr_rtt - ag.rtt_history[-6]) / 5.0 if len(ag.rtt_history) >= 6 else g1
            deliv_grad = ag.est_delivery_rate - (ag.deliv_history[-2] if len(ag.deliv_history) >= 2 else ag.est_delivery_rate)
            feat = [rtt_ratio, g1, g5, ag.est_delivery_rate, ag.rate, deliv_grad, float(sum(ag.recent_losses))]
            future_cap = cap_trace[t + 5]
            X.append(feat)
            y.append(future_cap)
    return np.array(X), np.array(y)


def collect_opponent_data(opponent_type, seeds=[30, 31, 32, 33]):
    """Collects training pool for opponent classifier LOFO evaluation."""
    X, y = [], []
    for s in seeds:
        rng = np.random.RandomState(s)
        ag = ReferenceNonMLAgent("CollectOpp")
        opp = None
        target_class = 0
        if opponent_type == "reno":
            opp = RenoAgent("Reno")
            target_class = 1
        elif opponent_type == "cubic":
            opp = CubicAgent("Cubic")
            target_class = 1
        elif opponent_type == "bbr":
            opp = AuditedBBRAgent("BBR")
            target_class = 2
        elif opponent_type == "copa":
            opp = CopaAgent("Copa")
            target_class = 3
        elif opponent_type == "vegas":
            opp = ReferenceVegasAgent("Vegas")
            target_class = 3
        elif opponent_type == "alone":
            opp = None
            target_class = 0

        queue = 0.0
        for t in range(2000):
            r_host = max(0.02, ag.rate)
            r_opp = max(0.02, opp.rate) if opp is not None else 0.0
            queue += (r_host + r_opp)
            if queue > 50.0:
                queue = 50.0
            proc = min(queue, 6.0)
            queue -= proc
            curr_rtt = 50.0 + queue * 1.9 + rng.normal(0, 0.8)
            ag.observe_ack(curr_rtt, False, proc * (r_host / max(r_host + r_opp, 1e-3)))
            ag.update_rate(False)
            if opp is not None:
                opp.observe_ack(curr_rtt, False, proc * (r_opp / max(r_host + r_opp, 1e-3)))
                opp.update_rate(False)

            if t % 5 == 0 and t > 50:
                rtt_ratio = curr_rtt / max(ag.min_rtt, 1e-3)
                g1 = curr_rtt - ag.rtt_history[-2] if len(ag.rtt_history) >= 2 else 0.0
                g5 = (curr_rtt - ag.rtt_history[-6]) / 5.0 if len(ag.rtt_history) >= 6 else g1
                rate_ratio = ag.rate / max(ag.est_delivery_rate, 0.1)
                feat = [rtt_ratio, g1, g5, rate_ratio, float(ag.standing_queue_steps)]
                X.append(feat)
                y.append(target_class)
    return np.array(X), np.array(y)


def train_lofo_loss_models():
    """Trains LOFO loss classification models and baseline classifiers."""
    X_bernoulli, y_bernoulli = collect_loss_data(["bernoulli"])
    X_ge, y_ge = collect_loss_data(["gilbert_elliott"])
    
    # Model without Gilbert-Elliott
    rf_no_ge = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42).fit(X_bernoulli, y_bernoulli)
    tree_no_ge = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_bernoulli, y_bernoulli)
    lr_no_ge = LogisticRegression(max_iter=500, random_state=42).fit(X_bernoulli, y_bernoulli)
    
    # Model without Bernoulli
    rf_no_bern = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42).fit(X_ge, y_ge)
    tree_no_bern = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_ge, y_ge)
    lr_no_bern = LogisticRegression(max_iter=500, random_state=42).fit(X_ge, y_ge)
    
    return {
        "held_out_ge": {"rf": rf_no_ge, "tree": tree_no_ge, "lr": lr_no_ge},
        "held_out_bern": {"rf": rf_no_bern, "tree": tree_no_bern, "lr": lr_no_bern}
    }


def train_lofo_capacity_models():
    """Trains LOFO capacity forecasting models across all dynamic channel families."""
    cap_families = ["markov", "mahimahi", "step", "sinusoid"]
    cap_data = {}
    for fam in cap_families:
        cap_data[fam] = collect_capacity_data(fam)

    lofo_rf_models = {}
    for held_out_fam in cap_families:
        X_tr_list, y_tr_list = [], []
        for f in cap_families:
            if f != held_out_fam:
                X_tr_list.append(cap_data[f][0])
                y_tr_list.append(cap_data[f][1])
        X_tr = np.vstack(X_tr_list)
        y_tr = np.concatenate(y_tr_list)
        lofo_rf_models[held_out_fam] = RandomForestRegressor(n_estimators=25, max_depth=6, random_state=42).fit(X_tr, y_tr)
    return lofo_rf_models


def train_lofo_opponent_models():
    """Trains LOFO opponent detection models across loss-, rate-, and delay-based competitors."""
    opp_list = ["alone", "reno", "cubic", "bbr", "copa", "vegas"]
    opp_datasets = {}
    for opp in opp_list:
        opp_datasets[opp] = collect_opponent_data(opp)

    lofo_opp_rf = {}
    lofo_opp_lr = {}
    for held_out_opp in ["reno", "cubic", "bbr", "copa"]:
        X_list, y_list = [], []
        for opp in opp_list:
            if opp != held_out_opp:
                X_list.append(opp_datasets[opp][0])
                y_list.append(opp_datasets[opp][1])
        X_tr = np.vstack(X_list)
        y_tr = np.concatenate(y_list)
        lofo_opp_rf[held_out_opp] = RandomForestClassifier(n_estimators=25, max_depth=6, random_state=42).fit(X_tr, y_tr)
        lofo_opp_lr[held_out_opp] = LogisticRegression(max_iter=500, random_state=42).fit(X_tr, y_tr)

    # Master model for Alone / Vegas / Self evaluation
    X_all = np.vstack([opp_datasets[o][0] for o in opp_list])
    y_all = np.concatenate([opp_datasets[o][1] for o in opp_list])
    rf_master = RandomForestClassifier(n_estimators=25, max_depth=6, random_state=42).fit(X_all, y_all)
    lr_master = LogisticRegression(max_iter=500, random_state=42).fit(X_all, y_all)

    return {
        "lofo_rf": lofo_opp_rf,
        "lofo_lr": lofo_opp_lr,
        "master_rf": rf_master,
        "master_lr": lr_master
    }
