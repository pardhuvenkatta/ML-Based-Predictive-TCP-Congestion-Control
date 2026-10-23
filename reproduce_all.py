"""
ML-Based Predictive TCP Congestion Control
Script: reproduce_all.py (End-to-End Master Reproduction Runner)

Single-command script that executes all three comprehensive benchmark suites:
1. step1_supervised_and_loss_classification.py:
   - Broad randomized dataset generation (67,500 train / 22,500 test samples).
   - Supervised models: RF, LightGBM, Small MLP, GRU, 1D-CNN regression & classification.
   - Offline MAE, R2, accuracy, precision, recall, inference latency, model size.
   - Closed-loop loss classifier benchmark (0.1%, 0.5%, 1.0%, 2.0% loss & Gilbert-Elliott).
   - Closed-loop supervised held-out evaluation across 20 fixed seeds.
2. step2_rl_controllers_ppo_sac.py:
   - Gymnasium continuous action TCP environment.
   - 12 reward configurations swept across 3 seeds on domain randomization with competition curriculum.
   - Validation selection of best reward setting.
   - PPO and SAC training and evaluation.
   - Policy distillation into Decision Tree and Small MLP.
   - Comprehensive held-out evaluation with paired CIs across 20 seeds.
   - Gate F check.
3. step3_realistic_robustness_and_competition.py:
   - Realistic channel conditions: Cellular Markov fades, ACK compression, short flows, CoDel AQM, RTT heterogeneity.
   - Multi-flow scaling up to 16 concurrent flows (equal vs. staggered starts).
   - Copa/Nimbus competitive mode switching across buffer sizes B=10, 50, 200.
   - Step-change stability metrics and +/-50% simulator parameter sensitivity sweep.
   - Generates publication plots (realistic_time_series_step_change.png, multi_flow_scaling_jfi_chart.png).

Usage:
  python reproduce_all.py
"""

import os
import sys
import time
import subprocess

def run_phase(phase_name, script_path):
    print("\n" + "=" * 80)
    print(f"  RUNNING: {phase_name}")
    print(f"  Script:  {script_path}")
    print("=" * 80)
    t0 = time.time()
    cmd = [sys.executable, "-u", script_path]
    proc = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"  [ERROR] {phase_name} failed with exit code {proc.returncode} after {elapsed:.1f}s")
        sys.exit(proc.returncode)
    else:
        print(f"  [OK] {phase_name} finished successfully in {elapsed:.1f}s")
    return elapsed

def main():
    total_start = time.time()
    print("*" * 80)
    print("  ML-BASED PREDICTIVE TCP CONGESTION CONTROL: MASTER REPRODUCTION SUITE")
    print("  Authors: BAHAAR (Reg. 24BCE1760), PARDHU (Reg. 24BCE1754)")
    print("*" * 80)

    phases = [
        ("Step 1: Supervised Models & Loss Classifier", "step1_supervised_and_loss_classification.py"),
        ("Step 2: RL Controllers PPO & SAC, Distillation & Gate F", "step2_rl_controllers_ppo_sac.py"),
        ("Step 3: Realistic Robustness, 16-Flow Scaling & Competition", "step3_realistic_robustness_and_competition.py")
    ]

    phase_timings = {}
    for name, script in phases:
        elapsed = run_phase(name, script)
        phase_timings[name] = elapsed

    total_time = time.time() - total_start
    print("\n" + "*" * 80)
    print("  ALL BENCHMARK PHASES COMPLETED CLEANLY!")
    print("*" * 80)
    print("\nExecution Timing Breakdown:")
    for name, t in phase_timings.items():
        print(f"  - {name}: {t:.1f}s ({t/60:.2f} min)")
    print(f"\nTotal Pipeline Execution Time: {total_time:.1f}s ({total_time/60:.2f} min)")
    print("All CSV artifacts, telemetry tables, and publication figures regenerated cleanly.")

if __name__ == "__main__":
    main()
