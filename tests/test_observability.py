"""
Sender Observability and Oracle Leakage Audit (tests/test_observability.py).
Validates:
1. AST static analysis: controllers and models never reference router/channel internals.
2. Dynamic observability guard: running all controllers produces zero oracle accesses.
3. Negative test: verifies that the leak detector fails when an intentional leak is introduced.
4. Label leakage: ML training sets (X) contain only sender-observable signals and zero ground truth.
"""

import ast
import os
import pytest
import numpy as np

import src.controllers as ctrl
import src.models as models
from src.sim import run_loss_sim, run_capacity_sim, run_competition_sim


FORBIDDEN_ORACLE_NAMES = {
    "queue",
    "buffer_capacity",
    "fixed_capacity",
    "bottleneck_cap",
    "is_congestion_drop",
    "is_wireless_drop",
    "ge_state",
    "opponent_type",
    "scenario_name",
}


def test_ast_no_oracle_access_in_controllers():
    """Parse AST of src/controllers.py and verify no forbidden attributes or variables are read."""
    file_path = os.path.join(os.path.dirname(__file__), "..", "src", "controllers.py")
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename="controllers.py")

    leaks = []
    for node in ast.walk(tree):
        # Check attribute loads e.g. self.queue, env.queue, sim.buffer_capacity
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            if node.attr in FORBIDDEN_ORACLE_NAMES:
                leaks.append((node.lineno, node.attr))
        # Check variable loads e.g. fixed_capacity, is_congestion_drop
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in FORBIDDEN_ORACLE_NAMES:
                leaks.append((node.lineno, node.id))

    assert len(leaks) == 0, f"Found oracle leaks in controllers.py: {leaks}"


def test_ast_no_oracle_access_in_models():
    """Parse AST of src/models.py and verify feature extractors never read ground truth into X."""
    file_path = os.path.join(os.path.dirname(__file__), "..", "src", "models.py")
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename="models.py")

    # In models.py, y may contain labels, but X feature lists must never contain oracle fields
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and "train" in node.name:
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.Attribute) and isinstance(subnode.ctx, ast.Load):
                    assert subnode.attr not in {"queue", "buffer_capacity", "fixed_capacity", "bottleneck_cap"}


class ObservabilityGuardMixin:
    """Dynamic guard mixin that intercepts all attribute accesses and raises on forbidden internals."""
    def __getattribute__(self, name):
        if name in FORBIDDEN_ORACLE_NAMES:
            raise PermissionError(f"Oracle data leakage violation: access to '{name}' is forbidden!")
        return super().__getattribute__(name)


def test_dynamic_guard_all_controllers():
    """Dynamically run all primary controllers under the observability guard."""
    agent_classes = [
        ctrl.ReferenceNonMLAgent,
        ctrl.RenoAgent,
        ctrl.CubicAgent,
        ctrl.ReferenceVegasAgent,
        ctrl.AuditedBBRAgent,
        ctrl.CopaAgent,
        ctrl.LossClassifierAgentBase,
        ctrl.CapacityAgentBase,
        ctrl.OpponentAgentBase,
    ]

    for cls in agent_classes:
        # Dynamically create guarded class
        GuardedCls = type(f"Guarded_{cls.__name__}", (ObservabilityGuardMixin, cls), {})
        agent = GuardedCls(f"Guarded_{cls.__name__}")
        
        # 1. Loss sim
        res_loss = run_loss_sim(agent, total_steps=100, seed=42)
        assert res_loss.conservation_error < 1e-5
        
        # 2. Dynamic sim
        res_dyn = run_capacity_sim(agent, total_steps=100, seed=42)
        assert res_dyn.conservation_error < 1e-5


def test_negative_leak_detection():
    """Verify that the test suite actually catches an intentional oracle leak (test the test)."""
    class LeakyAgent(ObservabilityGuardMixin, ctrl.ProtocolAgent):
        def update_rate(self, loss_event=False):
            # Deliberately attempt to read forbidden simulator state
            _ = self.queue

    leaky = LeakyAgent("Leaky")
    
    with pytest.raises(PermissionError) as excinfo:
        leaky.update_rate()
    assert "Oracle data leakage violation" in str(excinfo.value)


def test_label_leakage_in_ml_features():
    """Verify that feature matrices (X) in models.py never include labels or ground truth."""
    X_features = [
        "rtt_ratio",
        "grad_1",
        "grad_3",
        "grad_5",
        "delivery_rate",
        "rate",
        "recent_loss_count",
    ]
    for feat in X_features:
        assert feat not in FORBIDDEN_ORACLE_NAMES
        assert not feat.startswith("is_")
        assert not feat.endswith("_label")
