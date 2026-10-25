"""
Statistical Analysis, Utility Scoring, and Pareto Frontier Utilities for ML-TCP
Implements Ud = ln(goodput) - d * ln(RTT), paired Student's t-tests, Holm-Bonferroni correction,
Cohen's d effect sizes, and practical significance validation.
"""

import numpy as np
from scipy import stats


def calc_u(goodput, avg_rtt, d=1.0):
    """
    Computes generalized Kleinrock-style utility:
    U_d = ln(goodput) - d * ln(avg_rtt)
    """
    gp = max(1e-4, goodput)
    rtt = max(1.0, avg_rtt)
    return float(np.log(gp) - d * np.log(rtt))


def paired_test(test_vals, ref_vals):
    """
    Computes paired Student's t-test, 95% confidence intervals, and Cohen's d effect size.
    """
    diffs = np.array(test_vals) - np.array(ref_vals)
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    se_d = std_d / np.sqrt(n) if n > 1 else 1e-6
    t_stat = mean_d / se_d if se_d > 1e-12 else 0.0
    p_val = float(2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=n - 1))) if n > 1 else 1.0
    t_crit = stats.t.ppf(0.975, df=n - 1) if n > 1 else 2.0
    ci_low = float(mean_d - t_crit * se_d)
    ci_high = float(mean_d + t_crit * se_d)
    cohen_d = float(mean_d / std_d) if std_d > 1e-9 else 0.0
    return {
        "mean_diff": mean_d,
        "std_diff": std_d,
        "se_diff": se_d,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_val": p_val,
        "cohen_d": cohen_d,
    }


def holm_bonferroni(p_values):
    """
    Performs step-down Holm-Bonferroni correction on an array of p-values.
    Returns: corrected adjusted p-values.
    """
    p_vals = np.array(p_values)
    m = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    adj_p = np.zeros(m)

    current_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        multiplier = m - rank
        raw_p = p_vals[idx]
        adj = min(1.0, raw_p * multiplier)
        current_max = max(current_max, adj)
        adj_p[idx] = current_max
    return adj_p.tolist()


def check_practical_significance(
    diff_u05, diff_u10, diff_u20, ci_low_u10, ci_high_u10, ref_u05, ref_u10, ref_u20
):
    """
    Validates practical significance:
    Requires 95% CI to strictly exclude zero AND relative effect >= 1.0% across all d in {0.5, 1.0, 2.0}.
    """
    ci_excludes_zero = ci_low_u10 > 0 and ci_high_u10 > 0
    pct_u05 = (diff_u05 / abs(ref_u05)) * 100.0 if abs(ref_u05) > 1e-6 else 0.0
    pct_u10 = (diff_u10 / abs(ref_u10)) * 100.0 if abs(ref_u10) > 1e-6 else 0.0
    pct_u20 = (diff_u20 / abs(ref_u20)) * 100.0 if abs(ref_u20) > 1e-6 else 0.0

    exceeds_1pct = (pct_u05 >= 1.0) and (pct_u10 >= 1.0) and (pct_u20 >= 1.0)

    if ci_excludes_zero and exceeds_1pct:
        return True, "Statistically Outperforms (>=1% Effect & CI Excludes 0)", pct_u10
    elif ci_excludes_zero and not exceeds_1pct:
        return False, "Sub-Threshold Difference (<1% Effect)", pct_u10
    elif ci_low_u10 <= 0 <= ci_high_u10:
        return False, "Statistical Tie / Inconclusive (CI Includes 0)", pct_u10
    else:
        return False, "Inferior to Baseline", pct_u10


def compute_pareto_front(goodputs, rtts):
    """
    Calculates Pareto non-dominated frontier points:
    Maximizes goodput, minimizes RTT.
    """
    points = sorted(zip(goodputs, rtts), key=lambda x: (x[1], -x[0]))
    frontier = []
    max_gp = -float("inf")
    for gp, rtt in points:
        if gp > max_gp:
            frontier.append((gp, rtt))
            max_gp = gp
    return frontier
