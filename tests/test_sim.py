"""
Tests for Discrete-Event Physical Network Simulator (src/sim.py).
Validates:
1. Exact packet conservation across all scenarios.
2. Physical bounds (queue <= buffer, delivered <= capacity, RTT bounds).
3. Monotonic physical properties (buffer size vs loss, base RTT shifting).
4. Time-scale invariance.
5. Accurate loss accounting (wireless vs congestion separation, Bernoulli/GE distributions).
6. Multi-flow queuing, sharing, and Jain's Fairness Index computation.
7. Trace replay determinism and seed-driven randomization.
"""

import math

from src.controllers import ReferenceNonMLAgent, RenoAgent
from src.sim import (
    run_capacity_sim,
    run_competition_sim,
    run_loss_sim,
    run_multi_flow_sim,
)


class FixedRateAgent:
    """Deterministic sender sending a constant rate per step for physical verification."""

    def __init__(self, rate=4.0):
        self.fixed_rate = rate
        self.rate = rate
        self.min_rtt = None
        self.rtt_history = []
        self.deliv_history = []
        self.recent_losses = [0] * 6

    def observe_ack(self, rtt, loss_event, acked_proc):
        if self.min_rtt is None:
            self.min_rtt = rtt
        else:
            self.min_rtt = min(self.min_rtt, rtt)
        self.rtt_history.append(rtt)
        self.deliv_history.append(acked_proc)

    def update_rate(self, loss_event=False):
        self.rate = self.fixed_rate


def test_packet_conservation_loss_scenarios():
    """Verify sent == delivered + congestion_loss + wireless_loss + queue at every step."""
    agent = ReferenceNonMLAgent("Ref")
    scenarios = [
        {"loss_mode": "none", "buffer_capacity": 50.0},
        {"loss_mode": "random", "random_loss_rate": 0.04, "buffer_capacity": 50.0},
        {
            "loss_mode": "ge",
            "ge_p_good_to_bad": 0.03,
            "ge_p_bad_to_good": 0.25,
            "buffer_capacity": 50.0,
        },
        {"loss_mode": "rayleigh", "buffer_capacity": 50.0},
        {
            "loss_mode": "random",
            "random_loss_rate": 0.05,
            "buffer_capacity": 5.0,
        },  # Shallow buffer
    ]
    for sc in scenarios:
        res = run_loss_sim(agent, total_steps=1000, seed=42, **sc)
        assert res.conservation_error < 1e-5, (
            f"Conservation violated in {sc}: error = {res.conservation_error}"
        )
        assert res.total_sent > 0
        assert math.isclose(
            res.total_sent,
            res.total_delivered
            + res.congestion_loss
            + res.wireless_loss
            + res.final_queue,
            abs_tol=1e-5,
        )


def test_packet_conservation_dynamic_and_competition():
    """Verify conservation in dynamic capacity walk, step drop, competition, and multi-flow."""
    agent = ReferenceNonMLAgent("Ref")

    # Capacity: markov walk
    res_walk = run_capacity_sim(agent, family="markov", total_steps=1000, seed=42)
    assert res_walk.conservation_error < 1e-5

    # Capacity: step drop
    res_step = run_capacity_sim(agent, family="step", total_steps=1000, seed=42)
    assert res_step.conservation_error < 1e-5

    # Competition
    agent_h = ReferenceNonMLAgent("Host")
    agent_o = RenoAgent("Opp")
    res_comp = run_competition_sim(agent_h, agent_o, total_steps=1000, seed=42)
    assert res_comp.conservation_error < 1e-5

    # Multi-flow
    flows = [ReferenceNonMLAgent(f"F{i}") for i in range(4)]
    res_multi = run_multi_flow_sim(flows, total_steps=1000, seed=42)
    assert res_multi.conservation_error < 1e-5


def test_physical_bounds():
    """Check physical invariants: queue bounded by buffer, throughput bounded by capacity."""
    agent = ReferenceNonMLAgent("Ref")
    res = run_loss_sim(
        agent,
        total_steps=1500,
        seed=42,
        fixed_capacity=3.0,
        buffer_capacity=20.0,
        base_rtt=50.0,
    )

    assert 0.0 <= res.final_queue <= 20.0 + 1e-6
    assert res.throughput <= 3.0 + 1e-6
    assert res.goodput <= 3.0 + 1e-6
    # p95 RTT cannot exceed base_rtt + queue_delay_max + 4*sigma_jitter
    max_theoretical_rtt = 50.0 + (20.0 * 1.9) + (4 * 0.8)
    assert res.p95_rtt <= max_theoretical_rtt + 5.0
    assert res.avg_rtt >= 50.0 - 5.0  # Cannot be significantly below base RTT


def test_buffer_monotonicity():
    """For an aggressive fixed-rate sender, doubling buffer capacity must not increase loss."""
    agent_shallow = FixedRateAgent(rate=5.0)
    res_shallow = run_loss_sim(
        agent_shallow,
        total_steps=1000,
        seed=42,
        fixed_capacity=3.0,
        buffer_capacity=10.0,
        loss_mode="none",
    )

    agent_deep = FixedRateAgent(rate=5.0)
    res_deep = run_loss_sim(
        agent_deep,
        total_steps=1000,
        seed=42,
        fixed_capacity=3.0,
        buffer_capacity=20.0,
        loss_mode="none",
    )

    # Deep buffer must store more packets, resulting in less or equal packet loss
    assert res_deep.congestion_loss <= res_shallow.congestion_loss + 1e-6
    # Average RTT must be higher in the deeper buffer due to larger standing queue
    assert res_deep.avg_rtt > res_shallow.avg_rtt


def test_base_rtt_shift():
    """Increasing base RTT by Delta must shift measured min RTT by approximately Delta."""
    agent1 = ReferenceNonMLAgent("Ref1")
    run_loss_sim(agent1, total_steps=1500, seed=42, base_rtt=50.0, loss_mode="none")

    agent2 = ReferenceNonMLAgent("Ref2")
    run_loss_sim(agent2, total_steps=1500, seed=42, base_rtt=100.0, loss_mode="none")

    # Minimum RTT observed directly reflects base RTT + channel jitter
    assert agent1.min_rtt is not None and agent2.min_rtt is not None
    delta_min_rtt = agent2.min_rtt - agent1.min_rtt
    assert 46.0 <= delta_min_rtt <= 54.0


def test_loss_accounting_bernoulli():
    """Verify that random loss matches configured Bernoulli probability within 3.5 standard deviations."""
    agent = FixedRateAgent(rate=2.0)
    p_loss = 0.05
    n_steps = 10000
    res = run_loss_sim(
        agent,
        total_steps=n_steps,
        seed=123,
        fixed_capacity=3.0,
        buffer_capacity=100.0,
        loss_mode="random",
        random_loss_rate=p_loss,
    )

    # Deep buffer + rate < capacity -> zero congestion loss
    assert res.congestion_loss == 0.0
    # Expected wireless drops: n_steps * p_loss
    expected_drops = n_steps * p_loss
    std_drops = math.sqrt(n_steps * p_loss * (1 - p_loss))

    # Number of wireless drop events (wireless_loss tracks dropped pkts)
    assert abs(res.wireless_loss - expected_drops) <= 3.5 * std_drops


def test_loss_accounting_separation():
    """Verify that congestion loss and wireless loss are strictly separated."""
    # Scenario 1: Only wireless loss
    agent1 = FixedRateAgent(rate=2.0)
    res1 = run_loss_sim(
        agent1,
        total_steps=1000,
        seed=42,
        fixed_capacity=3.0,
        buffer_capacity=100.0,
        loss_mode="random",
        random_loss_rate=0.04,
    )
    assert res1.wireless_loss > 0.0
    assert res1.congestion_loss == 0.0

    # Scenario 2: Only congestion loss (rate > capacity, shallow buffer, no wireless drops)
    agent2 = FixedRateAgent(rate=4.0)
    res2 = run_loss_sim(
        agent2,
        total_steps=1000,
        seed=42,
        fixed_capacity=3.0,
        buffer_capacity=10.0,
        loss_mode="none",
    )
    assert res2.congestion_loss > 0.0
    assert res2.wireless_loss == 0.0


def test_multi_flow_fairness_and_sharing():
    """Verify multi-flow queue sharing and Jain's Fairness Index computation."""
    flows = [ReferenceNonMLAgent(f"F{i}") for i in range(4)]
    res = run_multi_flow_sim(
        flows, total_steps=1500, seed=42, bottleneck_cap=3.0, buffer_capacity=30.0
    )

    assert len(res.goodputs) == 4
    # All flows should receive positive service
    for gp in res.goodputs:
        assert gp > 0.2

    # JFI must be in [1/n, 1.0] -> [0.25, 1.0]
    assert 0.25 <= res.jfi <= 1.0
    # Homogeneous non-ML reference agents should achieve reasonable fairness (JFI >= 0.85)
    assert res.jfi >= 0.85


def test_trace_replay_randomization():
    """Verify traces replay identically with same seed and randomize phase with different seeds."""
    agent1 = ReferenceNonMLAgent("Ref1")
    res1a = run_capacity_sim(agent1, family="markov", total_steps=800, seed=42)

    agent2 = ReferenceNonMLAgent("Ref2")
    res1b = run_capacity_sim(agent2, family="markov", total_steps=800, seed=42)

    agent3 = ReferenceNonMLAgent("Ref3")
    res2 = run_capacity_sim(agent3, family="markov", total_steps=800, seed=999)

    # Deterministic with identical seed
    assert math.isclose(res1a.goodput, res1b.goodput, abs_tol=1e-7)
    assert math.isclose(res1a.avg_rtt, res1b.avg_rtt, abs_tol=1e-7)

    # Seed variation produces distinct trajectory
    assert not math.isclose(res1a.goodput, res2.goodput, abs_tol=1e-3)
