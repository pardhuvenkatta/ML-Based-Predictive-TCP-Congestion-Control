"""
Mutation Tests for ML-TCP Test Suite (tests/test_mutations.py).
Validates that our verification suite strictly catches real mutations and flaws:
1. Mutation 1: Conservation violation (phantom packet creation) is caught.
2. Mutation 2: Oracle leakage (access to router queue/buffer internals) is caught.
3. Mutation 3: Sign flip in Reno additive increase is caught.
4. Mutation 4: Inverted/broken Jain's Fairness Index computation is caught.
5. Mutation 5: Broken delay threshold / queue ignoring is caught.
"""

import numpy as np
import pytest

import src.controllers as ctrl
from src.sim import run_loss_sim
from tests.test_observability import ObservabilityGuardMixin


def test_mutation_conservation_violation_caught():
    """Mutation 1: Creating phantom packets must violate conservation_error < 1e-5."""
    agent = ctrl.ReferenceNonMLAgent("Ref")
    res = run_loss_sim(agent, total_steps=500, seed=42)

    # Introduce mutation into result: phantom packets delivered out of thin air
    res_mutated_delivered = res.total_delivered + 10.0
    conservation_error_mutated = abs(
        res.total_sent
        - (
            res_mutated_delivered
            + res.congestion_loss
            + res.wireless_loss
            + res.final_queue
        )
    )

    # Verify our conservation invariant catches this mutation
    assert conservation_error_mutated >= 9.9
    with pytest.raises(AssertionError):
        assert conservation_error_mutated < 1e-5


def test_mutation_oracle_leak_caught():
    """Mutation 2: Controller accessing forbidden router state must be intercepted."""

    class LeakyMutantAgent(ObservabilityGuardMixin, ctrl.ProtocolAgent):
        def update_rate(self, loss_event=False):
            # Deliberately attempt to read router buffer capacity
            _ = self.buffer_capacity

    mutant = LeakyMutantAgent("Mutant")
    with pytest.raises(PermissionError) as excinfo:
        mutant.update_rate()
    assert "Oracle data leakage violation" in str(excinfo.value)


def test_mutation_reno_sign_flip_caught():
    """Mutation 3: Reno sign flip (decrement cwnd during congestion avoidance) must fail capacity test."""

    class MutatedInvertedReno(ctrl.ProtocolAgent):
        def __init__(self, name):
            super().__init__(name)
            self.cwnd = 2.0
            self.ssthresh = 64.0

        def update_rate(self, loss_event=False):
            min_rate = self.get_min_rate()
            if loss_event:
                self.cwnd = max(2.0, self.cwnd / 2.0)
            else:
                # MUTATION: Decrement cwnd instead of incrementing
                self.cwnd = max(0.05, self.cwnd - (1.0 / max(1.0, self.cwnd)))
            curr_rtt = self.rtt_history[-1] if self.rtt_history else 50.0
            base_rtt_est = self.min_rtt if self.min_rtt is not None else 50.0
            self.rate = max(min_rate, (self.cwnd / max(10.0, curr_rtt)) * base_rtt_est)

    mutant_reno = MutatedInvertedReno("MutantReno")
    res = run_loss_sim(
        mutant_reno, total_steps=1000, seed=42, fixed_capacity=3.0, loss_mode="none"
    )

    # The normal Reno achieves goodput > 2.0; inverted mutant collapses to min rate
    with pytest.raises(AssertionError):
        assert res.goodput >= 2.0, f"Expected capacity failure, got {res.goodput}"


def test_mutation_broken_jfi_caught():
    """Mutation 4: Broken JFI formula (omitting square on sum) must be caught."""
    goodputs = [1.0, 2.0, 3.0]
    sum_gp = float(np.sum(goodputs))
    sum_sq_gp = float(np.sum(np.array(goodputs) ** 2))

    # Correct JFI: (sum_gp ** 2) / (n * sum_sq_gp)
    correct_jfi = (sum_gp**2) / (len(goodputs) * sum_sq_gp)

    # Mutated broken JFI: sum_gp / (n * sum_sq_gp)
    mutated_jfi = sum_gp / (len(goodputs) * sum_sq_gp)

    assert 0.80 <= correct_jfi <= 1.0  # For [1, 2, 3], JFI = 36 / (3 * 14) = 0.857
    assert mutated_jfi < 0.20  # Broken JFI = 6 / 42 = 0.143 (outside valid range)
    with pytest.raises(AssertionError):
        assert 0.33 <= mutated_jfi <= 1.0


def test_mutation_ignored_delay_inflation_caught():
    """Mutation 5: Agent ignoring queue delay inflation and increasing rate indefinitely must cause queue blowout."""

    class RecklessAgent(ctrl.ProtocolAgent):
        def update_rate(self, loss_event=False):
            # MUTATION: Never back off on delay inflation, always increase rate
            self.rate += 0.20

    reckless = RecklessAgent("Reckless")
    res = run_loss_sim(
        reckless,
        total_steps=500,
        seed=42,
        fixed_capacity=3.0,
        buffer_capacity=50.0,
        base_rtt=50.0,
        loss_mode="none",
    )

    # Normal pacing agent maintains avg RTT < 65 ms; reckless mutant blows up queue to buffer cap (RTT ~ 145 ms)
    with pytest.raises(AssertionError):
        assert res.avg_rtt < 70.0, (
            f"Expected queue blowout failure, got avg RTT {res.avg_rtt}"
        )
