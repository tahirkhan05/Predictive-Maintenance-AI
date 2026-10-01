# Predictive Maintenance AI

> **Machine Failure Prediction Using Machine Learning**  
> *An IBM BOB Internship Project*  
> **Repository:** [https://github.com/tahirkhan05/Predictive-Maintenance-AI.git](https://github.com/tahirkhan05/Predictive-Maintenance-AI.git)

---

## Overview

**Predictive Maintenance AI** is a clean, technically rigorous machine learning project built to predict industrial milling machine failures from sensor telemetry and operating conditions. By predicting potential equipment failures in advance, the system helps plant operators minimize unexpected downtime, lower maintenance expenditures, and enhance workplace safety.

---

## Problem Statement

Unexpected equipment breakdowns in industrial manufacturing lead to costly production stoppages, tool destruction, and safety hazards. The goal of this project is to build a binary classification system that reliably identifies machines at high risk of failure (`1 = Failure`, `0 = Normal Operation`) while prioritizing **high recall** (capturing true failures) and minimizing false alarms.

---

## Dataset

This project uses the official **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.

- **Source:** [UCI Machine Learning Repository - AI4I 2020 Dataset](https://archive.ics.uci.edu/dataset/601/ai4i)
- **Observations:** 10,000 instances (synthetic, designed to reflect realistic milling machine telemetry).
- **Target Variable:** `Machine failure` (0: Normal Operation [96.61%], 1: Machine Failure [3.39%]).
- **Features:** Product Type (L, M, H), Air Temperature [K], Process Temperature [K], Rotational Speed [rpm], Torque [Nm], Tool Wear [min].

---

## Technologies Used

- **Language:** Python 3.10+
- **Machine Learning & Preprocessing:** `scikit-learn`, `numpy`, `pandas`, `joblib`
- **Exploratory Data Analysis:** `matplotlib`, `seaborn`
- **Web Application:** `Flask`
- **Frontend:** Vanilla HTML5, CSS3, JavaScript (Responsive Single-Screen Dashboard)
- **Testing:** `unittest`

---

## Machine Learning Approach

1. **Data Preprocessing & Validation:** Verification of schema integrity and zero missing values.
2. **Domain-Specific Feature Engineering:**
   - **Temperature Difference (`Temp_Difference`):** `Process temperature [K] - Air temperature [K]` (detects heat dissipation issues).
   - **Mechanical Power (`Power_kW`):** Calculated from Torque and Rotational speed (detects power overload/underpower).
   - **Overstrain Index (`Overstrain_Index`):** `Tool wear [min] × Torque [Nm]` (detects heavy tool strain).
   - **Torque-Speed Ratio (`Torque_Speed_Ratio`):** Detects low-speed high-load stalling anomalies.
3. **Handling Class Imbalance:** Stratified splitting and `class_weight='balanced'` to prevent bias toward the majority class without causing data leakage.
4. **Model Comparison & Selection:** Trained Logistic Regression and Random Forest Classifier. Evaluated using 5-Fold Stratified Cross-Validation and a 20% hold-out test set. Random Forest was selected as the champion model.

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
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Data loading, validation & splitting
│   ├── feature_engineering.py      # Domain feature calculations
│   ├── train_model.py              # Cross-validation, training & model saving
│   ├── evaluate_model.py           # Detailed evaluation & metrics reporting
│   └── predict.py                  # Prediction service & explainability logic
│
├── models/
│   ├── best_model.pkl              # Saved Scikit-Learn Pipeline
│   ├── metrics.json                # Measured evaluation metrics
│   └── feature_importance.json     # Extracted feature importances
│
├── templates/
│   └── index.html                  # Single-screen dashboard HTML
│
├── static/
│   ├── style.css                   # Modern high-contrast dashboard styling
│   └── script.js                   # Telemetry form & presets controller
│
├── tests/
│   └── test_pipeline.py            # Automated unit and integration tests
│
├── app.py                          # Flask web server
├── requirements.txt                # Python dependencies
├── README.md                       # Project overview & documentation
└── REPORT.md                       # Formal internship report
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

### 3. (Optional) Re-train the Model
```bash
python src/train_model.py
```

### 4. Run Automated Tests
```bash
python tests/test_pipeline.py
```

### 5. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## Results

Evaluation on the hold-out test set (2,000 samples: 1,932 Normal, 68 Failures):

| Metric | Logistic Regression | Random Forest Classifier (Best) |
| :--- | :---: | :---: |
| **Accuracy** | 86.10% | **98.05%** |
| **Precision** | 18.18% | **65.59%** |
| **Recall (Failures Detected)** | 88.24% | **89.71%** (61 of 68 failures caught) |
| **F1-Score** | 0.3015 | **0.7578** (5-Fold CV: 0.7659) |
| **ROC-AUC** | 0.9385 | **0.9869** (5-Fold CV: 0.9815) |

The close agreement between 5-fold cross-validation scores and holdout test metrics confirms the model is robust with **no overfitting or underfitting**.

---

## Web Application

The project includes a single-page web dashboard designed to fit entirely on a single screen for quick demonstration and screenshot capture for the IBM BOB internship evaluation.

### Features:
- **Real-Time Machine Failure Prediction:** Immediate classification into `Normal Operation` (green) or `Failure Risk Detected` (red).
- **Failure Probability Gauge:** Continuous failure risk percentage (0% to 100%).
- **Operational Explainability:** Translates raw numbers into physical machine diagnosis insights.
- **One-Click Presets:** Test presets for instant demonstration (*Normal Run*, *Overstrain Risk*, *Thermal Risk*, *Power Overload*).
- **Model Performance & Feature Importance:** Live display of model metrics and top predictive features.

---

## Future Improvements

- Streaming telemetry ingestion via MQTT / OPC-UA.
- Remaining Useful Life (RUL) regression modeling.
- Multi-class classification for distinct failure modes (TWF, HDF, PWF, OSF).
