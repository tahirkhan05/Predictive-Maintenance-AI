"""
Model Evaluation and Diagnostic Module for Predictive Maintenance AI.

Evaluates saved models, displays detailed confusion matrices, classification reports,
and generates diagnostic visualizations.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Add project root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.data_preprocessing import load_data, split_data
from src.feature_engineering import prepare_features_and_target

MODELS_DIR = 'models'


def load_model_and_metadata():
    """Load the trained model and associated metadata."""
    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    metrics_path = os.path.join(MODELS_DIR, 'metrics.json')
    feat_path = os.path.join(MODELS_DIR, 'feature_importance.json')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at '{model_path}'. Run src/train_model.py first.")
        
    model = joblib.load(model_path)
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    with open(feat_path, 'r') as f:
        feature_importances = json.load(f)
        
    return model, metrics, feature_importances


def evaluate_on_test_set():
    """Runs detailed evaluation on hold-out test set and prints a comprehensive report."""
    model, metrics, feature_importances = load_model_and_metadata()
    
    df_raw = load_data('data/ai4i2020.csv')
    _, test_raw = split_data(df_raw, test_size=0.2, random_state=42)
    X_test, y_test = prepare_features_and_target(test_raw)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("=" * 70)
    print("       PREDICTIVE MAINTENANCE AI - MODEL EVALUATION REPORT")
    print("=" * 70)
    print(f"\nBest Model Selected: {metrics['best_model']}")
    print("\nDetailed Classification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=['Normal Operation (0)', 'Machine Failure (1)'], digits=4))
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix Breakdown:")
    print(f"  * True Negatives  (Normal correctly identified):    {tn}")
    print(f"  * False Positives (False alarms / unnecessary stop): {fp}")
    print(f"  * False Negatives (Missed critical failures):        {fn}")
    print(f"  * True Positives  (Failures correctly predicted):    {tp}")
    
    print("\nModel Performance Metrics:")
    for model_name, res in metrics['models_comparison'].items():
        print(f"\n  [{model_name}]")
        for metric, val in res['test_metrics'].items():
            print(f"    - {metric.replace('_', ' ').title()}: {val}")
            
    print("\nTop Feature Importances:")
    for item in feature_importances[:7]:
        print(f"  * {item['feature']:<28}: {item['importance'] * 100:.2f}%")
        
    print("\n" + "=" * 70)


if __name__ == '__main__':
    evaluate_on_test_set()
