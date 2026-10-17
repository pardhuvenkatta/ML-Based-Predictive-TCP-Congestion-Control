"""
ML-Based Predictive TCP Congestion Control
Script: train_model.py (Machine Learning Training & Multi-Model Evaluation)

Team Members:
  - BAHAAR (Reg. No: 24BCE1760)
  - PARDHU (Reg. No: 24BCE1754)

Description:
  Trains the Random Forest classifier on telemetry data (network_data.csv)
  to predict queue congestion before packet loss occurs.
  - Features: RTT, RTT_Change, Queue_Size, Sending_Rate
  - Target: Label (0 = Normal, 1 = Queue > 80% Capacity)
  - Evaluates Accuracy, Precision, Recall, F1, Confusion Matrix
  - Measures Inference Latency (microseconds per prediction)
  - Compares with Logistic Regression and Decision Tree (Slide 6 table)
  - Exports trained model: rf_congestion_model.pkl
"""

import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def train_and_evaluate(dataset_path="network_data.csv", model_output_path="rf_congestion_model.pkl"):
    print("=" * 70)
    print("  ML-BASED PREDICTIVE TCP: MODEL TRAINING & EVALUATION")
    print("=" * 70)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset {dataset_path} not found! Run network_simulation.py first.")
        
    # 1. Load Telemetry Dataset
    df = pd.read_csv(dataset_path)
    print(f"\n[+] Loaded dataset: {dataset_path}")
    print(f"    - Total Samples: {len(df)}")
    print(f"    - Normal samples (Label 0): {sum(df['Label'] == 0)} ({sum(df['Label'] == 0)/len(df)*100:.1f}%)")
    print(f"    - Congested samples (Label 1): {sum(df['Label'] == 1)} ({sum(df['Label'] == 1)/len(df)*100:.1f}%)")
    
    # 2. Feature & Target Selection
    features = ['RTT', 'RTT_Change', 'Queue_Size', 'Sending_Rate']
    X = df[features]
    y = df['Label']
    
    # 3. Train-Test Split (80% Train, 20% Test, Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] Dataset Split: Train = {len(X_train)} samples, Test = {len(X_test)} samples")
    
    # 4. Multi-Model Comparison (Validating Slide 6)
    print("\n" + "-" * 70)
    print("  MODEL COMPARISON BENCHMARK (As presented in Project Slide 6)")
    print("-" * 70)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Single Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest (Optimal)": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    }
    
    trained_models = {}
    benchmark_results = []
    
    for name, model in models.items():
        # Train
        t_start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = (time.perf_counter() - t_start) * 1000  # ms
        
        # Test predictions
        t_inf_start = time.perf_counter()
        preds = model.predict(X_test)
        total_inf_time = (time.perf_counter() - t_inf_start) * 1e6  # microseconds
        avg_latency_us = total_inf_time / len(X_test)
        
        acc = accuracy_score(y_test, preds) * 100
        prec = precision_score(y_test, preds, zero_division=0) * 100
        rec = recall_score(y_test, preds, zero_division=0) * 100
        f1 = f1_score(y_test, preds, zero_division=0) * 100
        
        trained_models[name] = model
        benchmark_results.append({
            "Model": name,
            "Accuracy": f"{acc:.2f}%",
            "Precision": f"{prec:.2f}%",
            "Recall": f"{rec:.2f}%",
            "F1-Score": f"{f1:.2f}%",
            "Inference Latency": f"{avg_latency_us:.2f} us/packet"
        })
    
    bench_df = pd.DataFrame(benchmark_results)
    print(bench_df.to_string(index=False))
    
    # 5. Detailed Evaluation of Selected Random Forest Classifier
    rf_model = trained_models["Random Forest (Optimal)"]
    rf_preds = rf_model.predict(X_test)
    
    print("\n" + "-" * 70)
    print("  RANDOM FOREST DETAILED PERFORMANCE METRICS")
    print("-" * 70)
    print("\nClassification Report:")
    print(classification_report(y_test, rf_preds, target_names=["Normal (0)", "Congested (1)"]))
    
    cm = confusion_matrix(y_test, rf_preds)
    print("Confusion Matrix:")
    print(f"  True Normal  [TN={cm[0][0]:<4}  FP={cm[0][1]:<4}]  False Alarm")
    print(f"  Congestion   [FN={cm[1][0]:<4}  TP={cm[1][1]:<4}]  Detected Congestion")
    
    # Feature Importances
    importances = rf_model.feature_importances_
    print("\nFeature Importances:")
    for feat, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
        print(f"  - {feat:<15}: {imp * 100:>6.2f}%")
        
    # 6. Save Model to Disk
    joblib.dump(rf_model, model_output_path)
    print(f"\n[OK] Trained Random Forest model saved successfully as '{model_output_path}'")
    print("=" * 70)
    
    return rf_model


if __name__ == "__main__":
    train_and_evaluate()
