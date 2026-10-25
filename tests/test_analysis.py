"""
Tests for Statistical Analysis and Utility Metrics (src/analysis.py).
Validates:
1. Kleinrock utility Ud computation and monotonicity.
2. Paired t-test, standard errors, and p-values against scipy.stats.ttest_rel.
3. 95% Student's t confidence intervals.
4. Cohen's d effect sizes.
5. Step-down Holm-Bonferroni correction.
6. Practical significance logic (CI excludes 0 AND >= 1% effect across d=0.5, 1.0, 2.0).
7. 2D Pareto frontier non-dominance guarantees.
"""

import math
import numpy as np
import pytest
from scipy import stats

from src.analysis import (
    calc_u,
    paired_test,
    holm_bonferroni,
    check_practical_significance,
    compute_pareto_front,
)


def test_calc_u_properties():
    """Verify Kleinrock generalized utility Ud math and monotonicity."""
    gp = 2.5
    rtt = 50.0
    
    # Hand calculation: ln(2.5) - 1.0 * ln(50.0)
    expected_u10 = math.log(2.5) - 1.0 * math.log(50.0)
    assert math.isclose(calc_u(gp, rtt, d=1.0), expected_u10, rel_tol=1e-9)

    # Monotonicity checks
    # Higher goodput must increase utility
    assert calc_u(gp + 0.5, rtt, d=1.0) > calc_u(gp, rtt, d=1.0)
    # Higher RTT must decrease utility
    assert calc_u(gp, rtt + 10.0, d=1.0) < calc_u(gp, rtt, d=1.0)
    # Higher d penalizes delay more heavily
    assert calc_u(gp, rtt, d=2.0) < calc_u(gp, rtt, d=1.0) < calc_u(gp, rtt, d=0.5)


def test_paired_t_test_against_scipy():
    """Verify paired_test against standard scipy.stats.ttest_rel."""
    rng = np.random.RandomState(42)
    test_vals = rng.normal(loc=2.8, scale=0.2, size=30).tolist()
    ref_vals = rng.normal(loc=2.5, scale=0.2, size=30).tolist()

    res = paired_test(test_vals, ref_vals)
    scipy_res = stats.ttest_rel(test_vals, ref_vals)

    assert math.isclose(res["mean_diff"], float(np.mean(np.array(test_vals) - np.array(ref_vals))), rel_tol=1e-9)
    assert math.isclose(res["p_val"], float(scipy_res.pvalue), rel_tol=1e-7)

    # 95% Confidence Interval check
    diffs = np.array(test_vals) - np.array(ref_vals)
    n = len(diffs)
    expected_se = np.std(diffs, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    assert math.isclose(res["se_diff"], expected_se, rel_tol=1e-9)
    assert math.isclose(res["ci_low"], float(np.mean(diffs) - t_crit * expected_se), rel_tol=1e-9)
    assert math.isclose(res["ci_high"], float(np.mean(diffs) + t_crit * expected_se), rel_tol=1e-9)


def test_cohen_d_effect_size():
    """Verify Cohen's d computation."""
    diffs = [1.0, 1.2, 0.8, 1.1, 0.9]
    test_vals = [x + 5.0 for x in diffs]
    ref_vals = [5.0] * 5

    res = paired_test(test_vals, ref_vals)
    expected_d = np.mean(diffs) / np.std(diffs, ddof=1)
    assert math.isclose(res["cohen_d"], expected_d, rel_tol=1e-7)


def test_holm_bonferroni_multi_test_correction():
    """Verify Holm-Bonferroni step-down adjustment."""
    raw_p = [0.001, 0.012, 0.040, 0.150]
    adj_p = holm_bonferroni(raw_p)

    # Multipliers: 4, 3, 2, 1
    # 0.001 * 4 = 0.004
    # 0.012 * 3 = 0.036
    # 0.040 * 2 = 0.080
    # 0.150 * 1 = 0.150
    assert math.isclose(adj_p[0], 0.004, rel_tol=1e-7)
    assert math.isclose(adj_p[1], 0.036, rel_tol=1e-7)
    assert math.isclose(adj_p[2], 0.080, rel_tol=1e-7)
    assert math.isclose(adj_p[3], 0.150, rel_tol=1e-7)

    # Monotonicity check
    assert adj_p[0] <= adj_p[1] <= adj_p[2] <= adj_p[3]


def test_practical_significance_audit():
    """Verify practical significance classification across all decision branches."""
    ref_u = -1.5
    
    # 1. Genuine Win: CI strictly > 0 AND effect >= 1.0% across d=0.5, 1.0, 2.0
    win, desc, pct = check_practical_significance(
        diff_u05=0.05, diff_u10=0.05, diff_u20=0.05,
        ci_low_u10=0.02, ci_high_u10=0.08,
        ref_u05=-1.0, ref_u10=-1.5, ref_u20=-2.0
    )
    assert win is True
    assert "Statistically Outperforms" in desc

    # 2. Sub-threshold: CI strictly > 0, but effect < 1.0%
    sub, desc_sub, _ = check_practical_significance(
        diff_u05=0.001, diff_u10=0.001, diff_u20=0.001,
        ci_low_u10=0.0005, ci_high_u10=0.0015,
        ref_u05=-1.0, ref_u10=-1.5, ref_u20=-2.0
    )
    assert sub is False
    assert "Sub-Threshold Difference" in desc_sub

    # 3. Inconclusive: CI overlaps zero
    tie, desc_tie, _ = check_practical_significance(
        diff_u05=0.02, diff_u10=0.02, diff_u20=0.02,
        ci_low_u10=-0.01, ci_high_u10=0.05,
        ref_u05=-1.0, ref_u10=-1.5, ref_u20=-2.0
    )
    assert tie is False
    assert "Inconclusive" in desc_tie

    # 4. Inferior: CI strictly negative
    inf, desc_inf, _ = check_practical_significance(
        diff_u05=-0.05, diff_u10=-0.05, diff_u20=-0.05,
        ci_low_u10=-0.08, ci_high_u10=-0.02,
        ref_u05=-1.0, ref_u10=-1.5, ref_u20=-2.0
    )
    assert inf is False
    assert "Inferior" in desc_inf


def test_pareto_frontier_correctness():
    """Verify 2D Pareto frontier contains no dominated points."""
    # (goodput, rtt)
    # We want max goodput, min RTT
    points_gp = [1.0, 2.0, 2.5, 2.2, 3.0]
    points_rtt = [30.0, 40.0, 50.0, 60.0, 70.0]
    # Point (2.2, 60.0) is dominated by (2.5, 50.0) which has higher GP and lower RTT!

    frontier = compute_pareto_front(points_gp, points_rtt)
    frontier_set = set(frontier)

    assert (2.2, 60.0) not in frontier_set
    assert (1.0, 30.0) in frontier_set
    assert (2.0, 40.0) in frontier_set
    assert (2.5, 50.0) in frontier_set
    assert (3.0, 70.0) in frontier_set

    # Verify every point on frontier has strictly increasing goodput as RTT increases
    for i in range(len(frontier) - 1):
        assert frontier[i + 1][0] > frontier[i][0]
        assert frontier[i + 1][1] > frontier[i][1]
