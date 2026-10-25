# Changelog: Repository Consolidation & Architecture Refactoring

All changes, data consolidation mappings, script reorganizations, and reproducibility verification procedures for the ML-TCP repository are tracked in this document.

---

## [Unreleased] - 2026-10-25: Complete Repository Consolidation

### 1. Architectural Consolidation Overview
To streamline code review, package distribution, and long-term reproducibility without losing any scientific data or experimental history:
- **Repository Layout**:
  ```text
  .
  ├── README.md               # Primary project overview, benchmarks, quickstart
  ├── REPORT.md               # Comprehensive scientific audit and hardware report
  ├── CHANGELOG.md            # Detailed schema mappings and revision log
  ├── LICENSE                 # MIT Open-Source License
  ├── requirements.txt        # Exact Python dependencies
  ├── seeds.json              # Fixed random seeds for benchmarks and training
  ├── reproduce_all.py        # Single unified CLI reproducing all experiments
  ├── src/                    # Consolidated modular source code
  │   ├── sim.py              # Discrete-event network simulator (unified)
  │   ├── controllers.py      # Canonical controller implementations (heuristics, rules, ML)
  │   ├── models.py           # Neural network & ML architectures (RF, GRU, CNN, MLP, LOFO)
  │   ├── experiments.py      # Modular experiment runners matching audit phases
  │   └── analysis.py         # Statistical analysis, U_d utility, t-tests, Pareto frontiers
  ├── results/                # Complete consolidated experimental telemetry
  │   ├── results_all.csv.gz  # Long-format raw runs and traces (6,800 rows, 45 columns)
  │   ├── results_summary.csv # Consolidated summary & scorecard tables (1,499 rows, 304 columns)
  │   └── models.zip          # Trained model weights (RF, GRU, CNN, MLP, LightGBM)
  ├── figures/                # Presentation and audit figures
  │   ├── figures.pdf         # Multi-page compilation of all 31 experimental figures
  │   ├── pareto_frontier_audit_3domains.png
  │   ├── architecture_diagram.png
  │   ├── comparison_dashboard.png
  │   └── benchmark_barchart_with_ci.png
  └── archive/
      └── old_files.zip       # Full byte-for-byte pre-consolidation archive (169 files)
  ```

---

### 2. Dataset & CSV Consolidation Schema Mappings

#### A. Raw Simulation Runs (`results/results_all.csv.gz`)
The 4 wide-format per-seed experimental CSVs (`phase1_fair_baseline_train_tuning_raw.csv`, `phase2_loss_classification_raw.csv`, `phase3_competitor_classification_raw.csv`, `phase4_capacity_prediction_raw.csv`), plus diagnostic traces and training telemetry, have been unpivoted and merged into a canonical long-format table.

**Standard Long-Format Columns**:
- `experiment`: Name of the experimental benchmark (e.g., `fair_baseline_train_tuning`, `loss_classification`, `competitor_classification`, `capacity_prediction`, `diagnostic_trace`, `network_telemetry_dataset`).
- `phase`: Experimental phase (`phase1`, `phase2`, `phase3`, `phase4`, `audit`, `training`).
- `scenario`: Scenario identifier (e.g., `HO1_UnseenRTT300`, `L1_Rand01`, `C1_Alone`, `D1_StepDrop`, `RTT_20ms`, `synthetic_traces`).
- `controller`: Canonical controller name (`default_nonml`, `tuned_nonml`, `model_off`, `ml_guided`, `pacing_coincident_rise`, `rf_training_baseline`).
- `seed`: Integer simulation random seed (`42`, `43`, ...).
- `source_file`: Original filename from which the record originated.

**Metric Column Mappings**:
1. **Phase 1 (`phase1_fair_baseline_train_tuning_raw.csv`)**:
   - `def_goodput` $\rightarrow$ `goodput` (where `controller = 'default_nonml'`)
   - `def_rtt` $\rightarrow$ `avg_rtt` (where `controller = 'default_nonml'`)
   - `def_loss` $\rightarrow$ `loss` (where `controller = 'default_nonml'`)
   - `def_u10` $\rightarrow$ `u10` (where `controller = 'default_nonml'`)
   - `tun_goodput` $\rightarrow$ `goodput` (where `controller = 'tuned_nonml'`)
   - `tun_rtt` $\rightarrow$ `avg_rtt` (where `controller = 'tuned_nonml'`)
   - `tun_loss` $\rightarrow$ `loss` (where `controller = 'tuned_nonml'`)
   - `tun_u10` $\rightarrow$ `u10` (where `controller = 'tuned_nonml'`)
   - `diff_u10` $\rightarrow$ `diff_u10` (preserved directly)

2. **Phase 2 (`phase2_loss_classification_raw.csv`)**:
   - `gp_off`, `u_off` $\rightarrow$ `goodput`, `u10` (where `controller = 'model_off'`)
   - `gp_def`, `u_def` $\rightarrow$ `goodput`, `u10` (where `controller = 'default_nonml'`)
   - `gp_tun`, `u_tun` $\rightarrow$ `goodput`, `u10` (where `controller = 'tuned_nonml'`)
   - `gp_ml`, `u_ml` $\rightarrow$ `goodput`, `u10` (where `controller = 'ml_guided'`)
   - `delta_ml_vs_tun`, `delta_ml_vs_def`, `delta_ml_vs_off` $\rightarrow$ preserved directly for `ml_guided`

3. **Phase 3 (`phase3_competitor_classification_raw.csv`)**:
   - `gp_off`, `rtt_off`, `u_off` $\rightarrow$ `goodput`, `avg_rtt`, `u10` (where `controller = 'model_off'`)
   - `gp_def`, `rtt_def`, `u_def` $\rightarrow$ `goodput`, `avg_rtt`, `u10` (where `controller = 'default_nonml'`)
   - `gp_tun`, `rtt_tun`, `u_tun` $\rightarrow$ `goodput`, `avg_rtt`, `u10` (where `controller = 'tuned_nonml'`)
   - `gp_ml`, `rtt_ml`, `u_ml` $\rightarrow$ `goodput`, `avg_rtt`, `u10` (where `controller = 'ml_guided'`)
   - `delta_vs_tun` $\rightarrow$ preserved directly for `ml_guided`

4. **Phase 4 (`phase4_capacity_prediction_raw.csv`)**:
   - Mapped identically to Phase 3 (`gp_*`, `rtt_*`, `u_*`, and `delta_vs_tun`).

5. **Diagnostic Traces (`diagnostic_trace_rtt_{20,50,100,200}.csv`)**:
   - Preserves all 13 step-level columns: `step`, `curr_rtt`, `min_rtt`, `rtt_ratio`, `grad_1`, `grad_3`, `grad_5`, `est_delivery`, `est_capacity`, `pacing_rate`, `cwnd`, `inflight`, `queue_estimate`.

6. **Network Telemetry (`network_data.csv`)**:
   - Preserves all 18 feature columns: `RTT`, `RTT_Change`, `Queue_Size`, `Sending_Rate`, `Label`, `rtt_ratio`, `rtt_gradient_1`, `rtt_gradient_3`, etc.

#### B. Summary & Scorecard Tables (`results/results_summary.csv`)
All 73 summary CSVs have been aggregated into a unified table with:
- `table`: Base name of the originating summary table (e.g., `audit_master_scorecard`, `audit_lofo_loss_classification`, `phase5_global_synthesis_audit`, `audit_50ms_regression_check`).
- `source_file`: Filename of the originating file.
- All original metric columns and headers are preserved verbatim without altering any values.

---

### 3. Model Weights Archive (`results/models.zip`)
Trained neural and tree models have been packaged into `results/models.zip`:
- `model_cnn_h5.pt`: 1D Convolutional Neural Network for multi-step capacity prediction.
- `model_gru_h5.pt`: Gated Recurrent Unit for sequential RTT/rate dynamics.
- `model_mlp_h5.pt`: Multi-Layer Perceptron baseline.
- `model_lgbm_h5.joblib`: LightGBM gradient boosted decision tree.
- `model_rf_h5.joblib`: Random Forest regressor for multi-step capacity forecasting.
- `rf_congestion_model.pkl`: Random Forest classifier for proactive drop detection.
- `rf_iterative_model.pkl`: Iterative tree model for fine-grained pacing.

---

### 4. Codebase Modularization into `src/`
The 37 top-level experimental scripts were consolidated into 5 dedicated modules:
1. `src/sim.py`: Unified physical discrete-event network simulator supporting dynamic capacity walks, wireless fading (Rayleigh, Ricean, Markov Gilbert-Elliot), random packet loss, and multi-flow competing cross-traffic.
2. `src/controllers.py`: Deduplicated controller library containing base protocol agents, heuristics (`Reno`, `CUBIC`, `Vegas`, `BBR`, `Copa`), non-ML pacing rules (`DefaultNonML`, `TunedNonML`, `CoincidentRiseVariantD1`), and ML-guided agents (`LossClassifierAgent`, `CapacityForecasterAgent`, `OpponentDetectorAgent`).
3. `src/models.py`: Model architectures, feature extractors, and leave-one-family-out (LOFO) training pipelines ensuring zero ground-truth signal leakage.
4. `src/analysis.py`: Rigorous statistical scoring ($U_d$ utility across $d \in \{0.5, 1.0, 2.0\}$), paired Student's $t$-tests, Holm-Bonferroni corrections, Cohen's $d$ effect sizes, and Pareto frontier generation.
5. `src/experiments.py`: Executable harness exposing each audit and benchmark suite via clean Python functions.
6. `reproduce_all.py`: Unified CLI command runner supporting `--experiment <name>` and `--all`.
