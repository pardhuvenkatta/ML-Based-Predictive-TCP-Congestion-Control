"""
ML-Based Predictive TCP Congestion Control
Script: network_simulator.py (Alias for network_simulation.py)

Team Members:
  - BAHAAR (Reg. No: 24BCE1760)
  - PARDHU (Reg. No: 24BCE1754)
"""

from network_simulation import run_baseline_simulation

if __name__ == "__main__":
    run_baseline_simulation(total_steps=5000, save_csv=True, show_plot=False)
