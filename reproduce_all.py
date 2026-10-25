#!/usr/bin/env python3
"""
ML-TCP: Unified Experiment Reproduction CLI
Allows reproducing any scientific audit experiment or the entire benchmark suite.

Usage:
    python reproduce_all.py --experiment lofo_audit
    python reproduce_all.py --experiment pareto_frontiers
    python reproduce_all.py --experiment long_rtt_grid
    python reproduce_all.py --experiment regression_50ms
    python reproduce_all.py --experiment master_scorecard
    python reproduce_all.py --experiment copa_robustness
    python reproduce_all.py --experiment multi_flow_scaling
    python reproduce_all.py --experiment bbr_competition
    python reproduce_all.py --experiment all
    python reproduce_all.py --all --quick
"""

import argparse
import sys
import time

from src.experiments import (
    run_all,
    run_bbr_competition,
    run_copa_robustness,
    run_lofo_audit,
    run_long_rtt_grid,
    run_master_scorecard,
    run_multi_flow_scaling,
    run_pareto_frontiers,
    run_regression_50ms,
)

EXPERIMENT_MAP = {
    "lofo_audit": run_lofo_audit,
    "pareto_frontiers": run_pareto_frontiers,
    "long_rtt_grid": run_long_rtt_grid,
    "regression_50ms": run_regression_50ms,
    "master_scorecard": run_master_scorecard,
    "copa_robustness": run_copa_robustness,
    "multi_flow_scaling": run_multi_flow_scaling,
    "bbr_competition": run_bbr_competition,
    "all": run_all,
}


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce ML-TCP scientific audit experiments and benchmark suites."
    )
    parser.add_argument(
        "--experiment",
        "-e",
        choices=list(EXPERIMENT_MAP.keys()),
        default=None,
        help="Specific experiment to execute.",
    )
    parser.add_argument(
        "--all", "-a", action="store_true", help="Execute all experiments sequentially."
    )
    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Run fast verification mode with fewer seeds/steps.",
    )
    parser.add_argument(
        "--seeds",
        "-s",
        type=str,
        default=None,
        help="Comma-separated random seeds (e.g., '42,43,44,45').",
    )

    args = parser.parse_args()

    custom_seeds = None
    if args.seeds:
        custom_seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    target_exp = "all" if args.all else args.experiment

    if target_exp is None:
        parser.print_help()
        print("\nAvailable experiments:")
        for name in EXPERIMENT_MAP:
            print(f"  - {name}")
        sys.exit(0)

    start_time = time.time()
    print(f"Starting reproduction of: '{target_exp}' (quick={args.quick})...")

    func = EXPERIMENT_MAP[target_exp]
    if target_exp == "all" or target_exp in ["lofo_audit", "pareto_frontiers"]:
        func(quick=args.quick)
    elif target_exp in [
        "long_rtt_grid",
        "copa_robustness",
        "multi_flow_scaling",
        "bbr_competition",
    ]:
        if custom_seeds:
            func(seeds=custom_seeds)
        else:
            func()
    else:
        func()

    elapsed = time.time() - start_time
    print(f"\n[EXECUTION COMPLETED] Runtime: {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
