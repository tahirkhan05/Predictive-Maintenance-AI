# Predictive Maintenance AI

> **Machine Failure Prediction & Diagnostics Using Machine Learning**  
> *An IBM BOB Internship Project*  
> **Repository:** [https://github.com/tahirkhan05/Predictive-Maintenance-AI.git](https://github.com/tahirkhan05/Predictive-Maintenance-AI.git)

---

## Overview

**Predictive Maintenance AI** is an end-to-end machine learning project designed to predict industrial milling machine failures from operating conditions and sensor telemetry. By forecasting potential equipment breakdowns in advance, the system helps plant operators minimize unexpected downtime, reduce maintenance expenditures, and improve workplace safety.

---

## Problem Statement

Unexpected equipment breakdowns in manufacturing cause severe production halts, tool breakage, workpiece scrap, and safety hazards. The goal of this project is to build a classification pipeline that reliably identifies machines at high risk of failure (`1 = Failure`, `0 = Normal Operation`) while prioritizing **high recall** (detecting true failures) and minimizing false alarms.

---

## Dataset

- **Source:** [UCI Machine Learning Repository - AI4I 2020 Dataset](https://archive.ics.uci.edu/dataset/601/ai4i)
- **Observations:** 10,000 instances (synthetic, designed to reflect realistic milling machine telemetry).
- **Target Variable:** `Machine failure` (0: Normal Operation [96.61%], 1: Machine Failure [3.39%]).
- **Telemetry Features:** Product Type (L, M, H), Air Temperature [K], Process Temperature [K], Rotational Speed [rpm], Torque [Nm], Tool Wear [min].

---

## Key Machine Learning Results

Evaluated on an independent hold-out test set (**2,000 samples**: 1,932 Normal, 68 Failures):

| Metric | Logistic Regression (Balanced) | Random Forest Classifier (Best) |
| :--- | :---: | :---: |
| **Accuracy** | 86.10% | **98.05%** |
| **Recall / Sensitivity** | 88.24% | **89.71%** (61 of 68 failures caught) |
| **Precision** | 18.18% | **65.59%** |
| **Specificity (TNR)** | 86.02% | **98.34%** |
| **F1-Score** | 0.3015 | **0.7578** (5-Fold CV: 0.7659) |
| **ROC-AUC Score** | 0.9385 | **0.9869** (5-Fold CV: 0.9815) |
| **PR-AUC (Avg Precision)** | 0.4183 | **0.8574** |
| **False Positive Rate** | 13.98% (270 false alarms) | **1.66%** (Only 32 false alarms) |

The consistent performance between 5-fold cross-validation (**0.7659**) and holdout test metrics (**0.7578**) confirms **no overfitting or underfitting**.

---

## Diagnostic Visualizations & Graphs

Generated plots are located in [`reports/figures/`](file:///c:/Users/mdkta/OneDrive/Desktop/ibmbob2/reports/figures/):
- **`roc_curves_comparison.png`:** ROC curves comparing Random Forest (AUC = 0.9869) vs. Logistic Regression (AUC = 0.9385).
- **`precision_recall_curves.png`:** Precision-Recall curves showing 0.8574 Average Precision on the minority class.
- **`confusion_matrices_side_by_side.png`:** Confusion matrix heatmaps comparing predictions.
- **`feature_importance_bar_chart.png`:** Feature importance ranking across all 11 features.
- **`sensor_distributions_and_failure_modes.png`:** Telemetry boxplots and failure mode breakdown counts.

---

## Project Structure

```
predictive-maintenance-ai/
│
├── data/
│   └── ai4i2020.csv                # Official UCI AI4I 2020 Dataset
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # EDA & Modeling Notebook
│
├── reports/
│   └── figures/                    # Generated ROC, PR, CM & EDA Graphs
│       ├── roc_curves_comparison.png
│       ├── precision_recall_curves.png
│       ├── confusion_matrices_side_by_side.png
│       ├── feature_importance_bar_chart.png
│       └── sensor_distributions_and_failure_modes.png
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Data loading, validation & splitting
│   ├── feature_engineering.py      # Domain feature calculations
│   ├── train_model.py              # Extended metrics training & CV pipeline
│   ├── evaluate_model.py           # Evaluation report generator
│   └── predict.py                  # Prediction engine & subsystem stress gauges
│
├── models/
│   ├── best_model.pkl              # Saved Scikit-Learn Pipeline
│   ├── metrics.json                # Complete metrics JSON
│   └── feature_importance.json     # Feature importance weights
│
├── templates/
│   └── index.html                  # Single-screen dashboard HTML
│
├── static/
│   ├── style.css                   # High-contrast dashboard CSS
│   └── script.js                   # Interactive controller & modal viewer
│
├── tests/
│   └── test_pipeline.py            # Automated unit and integration tests
│
├── app.py                          # Flask web server & figures endpoint
├── requirements.txt                # Python dependencies
├── README.md                       # Project overview
└── REPORT.md                       # Comprehensive internship report
```

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/tahirkhan05/Predictive-Maintenance-AI.git
cd Predictive-Maintenance-AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python tests/test_pipeline.py
```

### 4. (Optional) Re-train the Model
```bash
python src/train_model.py
```

### 5. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## Web Application Features

- **Single-Screen Layout:** Everything fits on one desktop screen for single-screenshot internship submission.
- **Real-Time Prediction Banner:** Instant status (`Normal Operation` vs. `Failure Risk Detected`) with probability percentage.
- **Subsystem Stress Gauges:** Visual progress bars for Overstrain Load, Tool Life Expended, Power Envelope, and Thermal Stress.
- **Model Comparison Table:** Side-by-side performance metrics comparison.
- **Diagnostic Curves Modal:** Built-in pop-up viewer for ROC, PR, Confusion Matrix, and Feature Importance graphs.
- **1-Click Presets:** Instant loading for *Normal Run*, *Overstrain Risk*, *Thermal Risk*, and *Power Overload*.
