"""
Model Training & Evaluation Pipeline for Predictive Maintenance AI.

Trains and compares Logistic Regression and Random Forest models,
computes an extensive suite of classification metrics, handles class imbalance,
evaluates via 5-fold Stratified CV and hold-out test set, and exports the best pipeline.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Add project root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, balanced_accuracy_score
)

from src.data_preprocessing import load_data, validate_data, split_data
from src.feature_engineering import (
    prepare_features_and_target,
    BASE_NUMERICAL_FEATURES,
    ENGINEERED_FEATURES,
    CATEGORICAL_FEATURES
)

MODELS_DIR = 'models'


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs ColumnTransformer for preprocessing numerical and categorical features.
    """
    numerical_cols = BASE_NUMERICAL_FEATURES + ENGINEERED_FEATURES
    categorical_cols = CATEGORICAL_FEATURES
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
        ]
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Extract readable feature names from the fitted preprocessor.
    """
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(cols)
        elif name == 'cat':
            categories = trans.get_feature_names_out(cols)
            feature_names.extend(categories)
    return feature_names


def compute_extended_metrics(y_true, y_pred, y_proba):
    """
    Computes an extensive set of classification metrics for imbalanced datasets.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity / TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
    f1 = f1_score(y_true, y_pred, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        'accuracy': round(float(accuracy), 4),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'specificity': round(float(specificity), 4),
        'f1_score': round(float(f1), 4),
        'balanced_accuracy': round(float(balanced_acc), 4),
        'roc_auc': round(float(roc_auc), 4),
        'pr_auc': round(float(pr_auc), 4),
        'false_positive_rate': round(float(fpr), 4),
        'false_negative_rate': round(float(fnr), 4),
        'confusion_breakdown': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
    }


def train_and_evaluate():
    """
    Executes the end-to-end model training, comparison, and serialization pipeline.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("=" * 70)
    print("      PREDICTIVE MAINTENANCE AI - MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    # 1. Load and validate data
    print("\n[1/5] Loading & Validating Dataset...")
    df_raw = load_data('data/ai4i2020.csv')
    val_report = validate_data(df_raw)
    print(f"  [+] Total records: {val_report['total_rows']}")
    print(f"  [+] Class balance: {val_report['class_distribution']} (Failure rate: {val_report['failure_rate_percent']}%)")
    print(f"  [+] Missing values: {val_report['total_missing_values']}")
    
    # 2. Stratified Train/Test Split
    print("\n[2/5] Stratified Splitting (80% Train, 20% Test)...")
    train_raw, test_raw = split_data(df_raw, test_size=0.2, random_state=42)
    
    # 3. Feature Engineering
    print("\n[3/5] Applying Domain Feature Engineering...")
    X_train, y_train = prepare_features_and_target(train_raw)
    X_test, y_test = prepare_features_and_target(test_raw)
    print(f"  [+] Train features shape: {X_train.shape}")
    print(f"  [+] Test features shape:  {X_test.shape}")
    
    # 4. Define candidate models with Scikit-Learn Pipelines
    candidate_pipelines = {
        'Logistic Regression': Pipeline([
            ('preprocessor', build_preprocessor()),
            ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
        ]),
        'Random Forest Classifier': Pipeline([
            ('preprocessor', build_preprocessor()),
            ('classifier', RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ))
        ])
    }
    
    cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    results = {}
    print("\n[4/5] Training & Evaluating Models with 5-Fold Stratified Cross-Validation...")
    
    for name, pipeline in candidate_pipelines.items():
        print(f"\n--- Training {name} ---")
        
        # 5-fold CV on Train set
        cv_scores = cross_validate(pipeline, X_train, y_train, cv=cv_kfold, scoring=cv_metrics)
        
        # Train on full train set
        pipeline.fit(X_train, y_train)
        
        # Predict on holdout test set
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        test_ext = compute_extended_metrics(y_test, y_pred, y_proba)
        
        results[name] = {
            'test_metrics': test_ext,
            'cv_train_metrics': {
                'accuracy_mean': round(float(np.mean(cv_scores['test_accuracy'])), 4),
                'precision_mean': round(float(np.mean(cv_scores['test_precision'])), 4),
                'recall_mean': round(float(np.mean(cv_scores['test_recall'])), 4),
                'f1_mean': round(float(np.mean(cv_scores['test_f1'])), 4),
                'roc_auc_mean': round(float(np.mean(cv_scores['test_roc_auc'])), 4),
            },
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        print(f"  Test Accuracy:     {test_ext['accuracy'] * 100:.2f}%")
        print(f"  Test Precision:    {test_ext['precision'] * 100:.2f}%")
        print(f"  Test Recall (TPR): {test_ext['recall'] * 100:.2f}% (Critical failure detection)")
        print(f"  Test Specificity:  {test_ext['specificity'] * 100:.2f}%")
        print(f"  Test F1 Score:     {test_ext['f1_score']:.4f} (5-Fold CV: {results[name]['cv_train_metrics']['f1_mean']:.4f})")
        print(f"  Test ROC-AUC:      {test_ext['roc_auc']:.4f} (5-Fold CV: {results[name]['cv_train_metrics']['roc_auc_mean']:.4f})")
        print(f"  Test PR-AUC:       {test_ext['pr_auc']:.4f}")
        print(f"  Confusion Matrix (TN={test_ext['confusion_breakdown']['true_negatives']}, FP={test_ext['confusion_breakdown']['false_positives']}, FN={test_ext['confusion_breakdown']['false_negatives']}, TP={test_ext['confusion_breakdown']['true_positives']})")
    
    # 5. Best Model Selection & Feature Importance
    print("\n[5/5] Selecting Best Model & Exporting Artifacts...")
    best_model_name = 'Random Forest Classifier'
    best_pipeline = candidate_pipelines[best_model_name]
    
    # Extract feature importance from Random Forest
    rf_clf = best_pipeline.named_steps['classifier']
    rf_prep = best_pipeline.named_steps['preprocessor']
    feature_names = get_feature_names(rf_prep)
    importances = rf_clf.feature_importances_
    
    feat_imp_list = [
        {'feature': feat, 'importance': round(float(imp), 4)}
        for feat, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    ]
    
    print(f"\nSelected Best Model: {best_model_name}")
    print(f"Top Features Importance Ranking:")
    for item in feat_imp_list:
        print(f"  * {item['feature']:<26}: {item['importance'] * 100:.2f}%")
        
    # Save Pipeline
    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    joblib.dump(best_pipeline, model_path)
    print(f"\n[OK] Saved model pipeline to: {model_path}")
    
    # Save Metrics
    metrics_path = os.path.join(MODELS_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'best_model': best_model_name,
            'models_comparison': results,
            'dataset_summary': val_report
        }, f, indent=4)
    print(f"[OK] Saved evaluation metrics to: {metrics_path}")
    
    # Save Feature Importance
    feat_path = os.path.join(MODELS_DIR, 'feature_importance.json')
    with open(feat_path, 'w') as f:
        json.dump(feat_imp_list, f, indent=4)
    print(f"[OK] Saved feature importances to: {feat_path}")
    
    print("\n" + "=" * 70)
    print("          TRAINING & SERIALIZATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    return best_pipeline, results


if __name__ == '__main__':
    train_and_evaluate()
