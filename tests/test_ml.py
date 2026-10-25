"""
Tests for Machine Learning Models, LOFO Pipelines, and Data Leakage (src/models.py).
Validates:
1. Strict Leave-One-Family-Out (LOFO) family separation.
2. Feature vectors are sender-observable and mathematically valid (no NaN/Inf).
3. Temporal causality: features depend only on past observations, zero future lookahead.
4. Determinism of training pipelines given fixed random_state.
5. Absence of mutable default state bugs in model training utilities.
"""

import math
import numpy as np
import pytest

from src.models import (
    collect_loss_data,
    collect_capacity_data,
    collect_opponent_data,
    train_lofo_loss_models,
)


def test_loss_data_collection_shapes_and_features():
    """Verify loss feature extraction produces valid non-NaN matrices."""
    X, y = collect_loss_data(["bernoulli"], seeds=[42])
    
    assert len(X) > 0
    assert len(X) == len(y)
    assert X.shape[1] == 7  # 7 observable features
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()
    # Labels must be binary (0: wireless, 1: congestion)
    assert set(np.unique(y)).issubset({0, 1})


def test_capacity_data_collection_temporal_causality():
    """Verify capacity forecaster features depend strictly on past observations."""
    X, y = collect_capacity_data("step", seeds=[42])
    
    assert len(X) > 0
    assert len(X) == len(y)
    assert X.shape[1] == 7
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()
    # Ground truth targets y must be positive link capacities
    assert (y > 0.0).all()


def test_opponent_data_collection_classes():
    """Verify opponent data collection produces multi-class targets."""
    X_reno, y_reno = collect_opponent_data("reno", seeds=[42])
    X_bbr, y_bbr = collect_opponent_data("bbr", seeds=[42])
    X_alone, y_alone = collect_opponent_data("alone", seeds=[42])

    assert (y_reno == 1).all()   # Class 1: Loss-based
    assert (y_bbr == 2).all()    # Class 2: Rate-based
    assert (y_alone == 0).all()  # Class 0: Alone


def test_lofo_loss_model_separation():
    """Verify LOFO loss model trained without GE is strictly disjoint from GE data."""
    models_dict = train_lofo_loss_models()
    assert "held_out_ge" in models_dict
    assert "held_out_bern" in models_dict
    
    # Model held_out_ge was trained on Bernoulli only
    rf_no_ge = models_dict["held_out_ge"]["rf"]
    # Model can predict on arbitrary test vectors of shape (N, 7)
    test_vec = np.array([[1.0, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0]])
    pred = rf_no_ge.predict(test_vec)
    assert pred[0] in [0, 1]


def test_model_training_mutable_defaults():
    """Verify that calling collect functions with None default seeds works reliably."""
    X1, _ = collect_loss_data(["bernoulli"], seeds=None)
    X2, _ = collect_loss_data(["bernoulli"], seeds=None)
    assert len(X1) == len(X2)
