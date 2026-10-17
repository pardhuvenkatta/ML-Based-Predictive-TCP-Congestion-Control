"""
ML-Based Predictive TCP Congestion Control:
Enhancing Network Reliability through Machine Learning-Driven Proactive Congestion Management

Team Members:
  - BAHAAR (Reg. No: 24BCE1760)
  - PARDHU (Reg. No: 24BCE1754)

Unified Master Execution Launcher (CLI Interface)
"""

import sys
import os
import argparse

def print_banner():
    banner = """
===========================================================================
  ML-BASED PREDICTIVE TCP CONGESTION CONTROL SYSTEM
  Course Project: Computer Networks
  Team: BAHAAR (24BCE1760) & PARDHU (24BCE1754)
===========================================================================
    """
    print(banner)

def menu():
    print_banner()
    print("Select an execution mode:")
    print("  [1] Run HALF IMPLEMENTATION (Baseline Reactive TCP Simulation & Dataset Generation)")
    print("  [2] Train Machine Learning Models (Random Forest & Multi-Model Comparison)")
    print("  [3] Run FULL IMPLEMENTATION (Proactive ML-TCP Closed-Loop Simulation)")
    print("  [4] Run Threshold Sensitivity Benchmarks (Slide 4 Table Validation)")
    print("  [5] Generate Comparative Dashboards & Plots (Side-by-Side Comparison)")
    print("  [6] Run COMPLETE PIPELINE (End-to-End Execution of All Modules)")
    print("  [0] Exit")
    print("-" * 75)

def main():
    parser = argparse.ArgumentParser(description="ML-Based Predictive TCP Congestion Control")
    parser.add_argument("--mode", choices=["half", "train", "full", "benchmark", "compare", "all"],
                        help="Execution mode (half, train, full, benchmark, compare, all)")
    args = parser.parse_args()

    mode_map = {
        "half": "1",
        "train": "2",
        "full": "3",
        "benchmark": "4",
        "compare": "5",
        "all": "6"
    }

    choice = mode_map.get(args.mode)

    while True:
        if choice is None:
            menu()
            try:
                choice = input("Enter choice (0-6): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break

        if choice == "1":
            print("\n>>> EXECUTING HALF IMPLEMENTATION (Baseline Simulation) <<<\n")
            import network_simulation
            network_simulation.run_baseline_simulation(total_steps=5000, save_csv=True, show_plot=False)
            print("\n[Done] Half implementation execution complete.")
        elif choice == "2":
            print("\n>>> TRAINING & EVALUATING ML MODELS <<<\n")
            import train_model
            train_model.train_and_evaluate()
            print("\n[Done] ML training & evaluation complete.")
        elif choice == "3":
            print("\n>>> EXECUTING FULL IMPLEMENTATION (ML Proactive Control) <<<\n")
            import ml_tcp_simulation
            ml_tcp_simulation.run_ml_tcp_simulation()
            print("\n[Done] Full ML-TCP simulation complete.")
        elif choice == "4":
            print("\n>>> RUNNING THRESHOLD BENCHMARKS <<<\n")
            import ml_tcp_simulation
            ml_tcp_simulation.run_threshold_benchmarks()
            print("\n[Done] Threshold benchmarks complete.")
        elif choice == "5":
            print("\n>>> GENERATING COMPARATIVE VISUALIZATIONS <<<\n")
            import compare_results
            compare_results.generate_comparisons()
            print("\n[Done] Visualizations generated.")
        elif choice == "6":
            print("\n>>> EXECUTING COMPLETE END-TO-END WORKFLOW <<<\n")
            import network_simulation
            import train_model
            import ml_tcp_simulation
            import compare_results
            
            print("\n--- Step 1: Baseline Simulation ---")
            network_simulation.run_baseline_simulation(total_steps=5000, save_csv=True, show_plot=False)
            
            print("\n--- Step 2: ML Model Training ---")
            train_model.train_and_evaluate()
            
            print("\n--- Step 3: Proactive ML-TCP Simulation ---")
            ml_tcp_simulation.run_ml_tcp_simulation()
            
            print("\n--- Step 4: Threshold Benchmarking ---")
            ml_tcp_simulation.run_threshold_benchmarks()
            
            print("\n--- Step 5: Comparative Analysis & Dashboards ---")
            compare_results.generate_comparisons()
            
            print("\n" + "=" * 75)
            print("  ALL MODULES SUCCESSFULLY EXECUTED!")
            print("  Generated Artifacts:")
            print("   - network_simulation.py (Baseline simulation engine)")
            print("   - network_data.csv (5,000 labeled telemetry rows)")
            print("   - train_model.py (Random Forest trainer & latency profiler)")
            print("   - rf_congestion_model.pkl (Saved trained ML model)")
            print("   - ml_tcp_simulation.py (Proactive rate controller)")
            print("   - benchmark_threshold_results.csv (Slide 4 benchmark data)")
            print("   - compare_results.py (Evaluation suite)")
            print("   - rtt_plot.png (Baseline RTT oscillation plot)")
            print("   - ml_tcp_rtt_plot.png (ML-TCP stabilized RTT plot)")
            print("   - comparison_dashboard.png (Multi-panel comparison dashboard)")
            print("   - threshold_sensitivity_chart.png (Slide 4 benchmark bar chart)")
            print("=" * 75)
        elif choice == "0":
            print("Exiting.")
            break
        else:
            print("[!] Invalid choice. Please enter a number between 0 and 6.")

        if args.mode:
            break
        choice = None

if __name__ == "__main__":
    main()
