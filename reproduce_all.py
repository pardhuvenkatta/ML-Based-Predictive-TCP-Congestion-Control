"""
ML-Based Predictive TCP Congestion Control
Script: reproduce_all.py (Single-Command End-to-End Reproduction Runner)

Usage:
  python reproduce_all.py
"""

import os
import sys
import json
import subprocess

def main():
    print("=" * 80)
    print("  ML-TCP CONGESTION CONTROL: END-TO-END REPRODUCTION SUITE")
    print("=" * 80)
    
    with open("seeds.json", "r") as f:
        config = json.load(f)
    print(f"Loaded {len(config['seeds'])} fixed seeds from seeds.json: {config['seeds']}")
    
    scripts = [
        ("Phase 1: Baselines & Pacing Variants", "phase1_baselines_and_pacing.py"),
        ("Phase 2: Closed-Loop Predictor Ablation & Gate 1", "phase2_closed_loop_ablation.py"),
        ("Phase 6, 7 & 8: Fairness, Ablations & Deliverables", "phase6_7_8_final_suite.py")
    ]
    
    for title, script in scripts:
        print("\n" + "=" * 80)
        print(f"  RUNNING: {title} ({script})")
        print("=" * 80)
        cmd = [sys.executable, script]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"Error running {script}, exit code: {ret.returncode}")
            sys.exit(ret.returncode)
            
    print("\n" + "=" * 80)
    print("  [SUCCESS] All phases executed cleanly. All plots and CSVs generated.")
    print("=" * 80)

if __name__ == "__main__":
    main()
