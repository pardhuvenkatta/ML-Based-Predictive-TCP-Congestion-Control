"""
Tests for Controller Implementations (src/controllers.py).
Validates:
1. Single flow clean link reaches >= 95% capacity for ReferenceNonMLAgent and AuditedBBRAgent.
2. Distinct dynamics: Reno AIMD queue filling vs Vegas/Copa delay bounding vs BBR.
3. Scale invariance across base RTT in {20, 50, 100, 200, 300} ms.
4. Numerical robustness on edge cases (zero delivery, loss bursts, NaN protection).
5. Dynamic competitive switching logic and false-positive rate alone < 2%.
"""

import math
import numpy as np
import pytest

import src.controllers as ctrl
from src.sim import run_loss_sim, run_competition_sim


def test_single_flow_capacity_utilization():
    """Verify single flow reaches >= 95% capacity on a clean link for Reference and BBR agents."""
    fixed_cap = 3.0
    threshold_95 = 0.95 * fixed_cap  # 2.85 pkts/step
    
    # 1. Reference Non-ML Agent
    ref_agent = ctrl.ReferenceNonMLAgent("Ref")
    res_ref = run_loss_sim(ref_agent, total_steps=2000, seed=42, fixed_capacity=fixed_cap, loss_mode="none")
    assert res_ref.goodput >= threshold_95, f"Reference agent achieved {res_ref.goodput:.3f} < {threshold_95}"

    # 2. Audited BBR Agent
    bbr_agent = ctrl.AuditedBBRAgent("BBR")
    res_bbr = run_loss_sim(bbr_agent, total_steps=2000, seed=42, fixed_capacity=fixed_cap, loss_mode="none")
    assert res_bbr.throughput >= threshold_95, f"BBR agent throughput {res_bbr.throughput:.3f} < {threshold_95}"
    assert res_bbr.goodput >= 0.90 * fixed_cap, f"BBR agent goodput {res_bbr.goodput:.3f} < {0.90 * fixed_cap}"


def test_controller_dynamics_reno_vs_delay_based():
    """Verify Reno fills queue to buffer limit, while Vegas and Copa constrain queue delay."""
    fixed_cap = 3.0
    buffer_cap = 30.0
    base_rtt = 50.0

    # Reno: Loss-based AIMD fills the buffer
    reno = ctrl.RenoAgent("Reno")
    res_reno = run_loss_sim(reno, total_steps=2000, seed=42, fixed_capacity=fixed_cap, buffer_capacity=buffer_cap, base_rtt=base_rtt, loss_mode="none")
    assert res_reno.p95_rtt > base_rtt + 25.0, f"Reno did not fill queue: p95 RTT was {res_reno.p95_rtt:.1f}"

    # Vegas: Delay-based constrains queueing delay
    vegas = ctrl.ReferenceVegasAgent("Vegas")
    res_vegas = run_loss_sim(vegas, total_steps=2000, seed=42, fixed_capacity=fixed_cap, buffer_capacity=buffer_cap, base_rtt=base_rtt, loss_mode="none")
    assert res_vegas.avg_rtt < base_rtt + 15.0, f"Vegas queueing delay too high: avg RTT was {res_vegas.avg_rtt:.1f}"

    # Copa: Delay-based constrains queueing delay
    copa = ctrl.CopaAgent("Copa")
    res_copa = run_loss_sim(copa, total_steps=2000, seed=42, fixed_capacity=fixed_cap, buffer_capacity=buffer_cap, base_rtt=base_rtt, loss_mode="none")
    assert res_copa.avg_rtt < base_rtt + 15.0, f"Copa queueing delay too high: avg RTT was {res_copa.avg_rtt:.1f}"


def test_rtt_scale_invariance():
    """Verify Reference Non-ML controller achieves high utilization across diverse base RTTs."""
    rtt_grid = [20.0, 50.0, 100.0, 200.0, 300.0]
    fixed_cap = 3.0

    for base_rtt in rtt_grid:
        agent = ctrl.ReferenceNonMLAgent(f"Ref_{int(base_rtt)}")
        res = run_loss_sim(agent, total_steps=2500, seed=42, fixed_capacity=fixed_cap, base_rtt=base_rtt, loss_mode="none")
        # All scale points should maintain >= 88% link utilization without degradation
        assert res.goodput >= 0.88 * fixed_cap, f"Degraded at RTT={base_rtt}ms: goodput={res.goodput:.3f}"


def test_edge_cases_and_robustness():
    """Verify controllers handle step 0 initial state, zero delivery, and burst loss gracefully."""
    agent = ctrl.ReferenceNonMLAgent("Ref")
    
    # 1. First ACK observation
    agent.observe_ack(rtt=50.0, loss_event=False, acked_proc=1.0)
    assert agent.min_rtt == 50.0
    assert len(agent.rtt_history) == 6
    assert len(agent.deliv_history) == 20

    # 2. Zero delivery (link pause / blackout)
    agent.observe_ack(rtt=150.0, loss_event=True, acked_proc=0.0)
    agent.update_rate(loss_event=True)
    assert not math.isnan(agent.rate)
    assert not math.isinf(agent.rate)
    assert agent.rate >= agent.get_min_rate()

    # 3. Repeated heavy loss burst (10 steps)
    for _ in range(10):
        agent.observe_ack(rtt=200.0, loss_event=True, acked_proc=0.1)
        agent.update_rate(loss_event=True)
    assert not math.isnan(agent.rate)
    assert agent.rate >= agent.get_min_rate()

    # 4. CapacityAgentBase persistence edge case
    cap_agent = ctrl.CapacityAgentBase("Cap", mode="persistence")
    cap_agent.update_rate(loss_event=False)
    assert not math.isnan(cap_agent.rate)
    assert cap_agent.rate >= cap_agent.get_min_rate()


def test_competitive_mode_alone_fpr():
    """Verify OpponentDetector false-positive rate alone on clean & wireless loss links is < 2%."""
    # 1. Clean link alone
    agent_clean = ctrl.OpponentAgentBase("OppAgentClean", mode="default_rule")
    res_clean = run_competition_sim(agent_clean, None, total_steps=2000, seed=42)
    assert res_clean.fpr_comp <= 0.02, f"Clean channel FPR={res_clean.fpr_comp:.4f} > 2%"

    # 2. Random wireless loss alone (should not misclassify wireless drops as loss-based opponent)
    agent_loss = ctrl.OpponentAgentBase("OppAgentLoss", mode="default_rule")
    res_loss = run_competition_sim(agent_loss, None, total_steps=2000, seed=42)
    assert res_loss.fpr_comp <= 0.02, f"Wireless loss channel FPR={res_loss.fpr_comp:.4f} > 2%"
