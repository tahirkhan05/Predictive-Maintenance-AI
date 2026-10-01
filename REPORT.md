# Project Report: Predictive Maintenance AI

**Project Title:** Predictive Maintenance AI  
**Assignment:** IBM BOB Internship Project  
**Repository:** [https://github.com/tahirkhan05/Predictive-Maintenance-AI.git](https://github.com/tahirkhan05/Predictive-Maintenance-AI.git)  
**Author:** IBM BOB Intern  

---

## 1. Introduction

Predictive Maintenance (PdM) is a proactive equipment reliability strategy that evaluates the operational condition of industrial machinery by performing continuous condition monitoring. Unlike reactive maintenance (repairing equipment after sudden failure) or preventative maintenance (servicing equipment on fixed calendar schedules regardless of actual wear), predictive maintenance uses data-driven machine learning algorithms to anticipate mechanical failures before they cause costly unplanned downtime.

---

## 2. Problem Statement

Modern manufacturing plants depend heavily on high-speed CNC milling machines and rotational tools. Unplanned tool breaks, overheating, and torque overstrains lead to broken workpieces, damaged equipment spindles, production bottlenecks, and workplace safety hazards.

The core challenge addressed by this project is:
> *Can we reliably predict whether an industrial machine is going to fail based on its operational sensor telemetry (temperatures, speed, torque, wear), while ensuring maximum recall for critical failures and minimizing false alarms?*

---

## 3. Why This Project Was Made

Predicting machine failures provides major industrial advantages:
- **Reducing Unplanned Downtime:** Early warning alerts allow maintenance teams to service equipment during scheduled changeovers.
- **Lowering Maintenance Expenditures:** Catching tool wear before catastrophic breakdown prevents secondary damage to expensive motor drives and bearings.
- **Minimizing Scrap & Production Losses:** Eliminates spoiled raw material batches caused by mid-operation machine stalls.
- **Worker Safety:** Mitigates severe machine seizures and tool flying fragmentation.

---

## 4. Dataset

This project uses the official **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.

- **Dataset Source:** [UCI Machine Learning Repository - AI4I 2020](https://archive.ics.uci.edu/dataset/601/ai4i)
- **Dataset Nature:** A synthetic dataset created by Matzka (2020) reflecting realistic milling machine operational telemetry.
- **Observations:** 10,000 records
- **Missing Values:** 0 (clean dataset verified)
- **Target Variable:** `Machine failure` (Binary: `0` = Normal Operation, `1` = Machine Failure)
- **Class Balance:** Highly imbalanced — 9,661 Normal instances (96.61%) vs. 339 Failure instances (3.39%).

### Key Telemetry Features:
1. **Product Type (`Type`):** Low (L - 50%), Medium (M - 30%), High (H - 20%) quality variants.
2. **Air Temperature (`Air temperature [K]`):** Ambient room temperature (~300 K).
3. **Process Temperature (`Process temperature [K]`):** Machine operating temperature (~10 K above ambient).
4. **Rotational Speed (`Rotational speed [rpm]`):** Spindle speed (1,168 rpm to 2,886 rpm).
5. **Torque (`Torque [Nm]`):** Cutting torque exerted (3.8 Nm to 76.6 Nm).
6. **Tool Wear (`Tool wear [min]`):** Accumulated cutting tool duration (0 to 253 minutes).

*(Note: Specific failure mode flags TWF, HDF, PWF, OSF, RNF were strictly excluded from input features to prevent data leakage).*

---

## 5. Methodology

```
Raw Telemetry Data (UCI AI4I 2020)
                ↓
Data Loading & Validation (10,000 rows, 0 nulls)
                ↓
Domain Feature Engineering (Physics-Based)
                ↓
Stratified Train/Test Split (80% Train, 20% Test)
                ↓
ColumnTransformer Pipeline (StandardScaler + OneHotEncoder)
                ↓
Class-Weighted Model Training & 5-Fold Stratified Cross-Validation
                ↓
Comprehensive Metrics & Diagnostic Graphs Generation
                ↓
Single-Page Web Dashboard (Flask + HTML5/CSS)
```

### Domain Feature Engineering:
1. **Temperature Difference (`Temp_Difference` = Process Temp - Air Temp):**
   - *Rationale:* Heat Dissipation Failure (HDF) occurs when process heat cannot dissipate fast enough into ambient air at lower spindle speeds.
2. **Mechanical Power (`Power_kW` = Torque × Rotational Speed × 2π / 60,000):**
   - *Rationale:* Power Failure (PWF) occurs when power output drops below 3.5 kW or exceeds 9.0 kW.
3. **Overstrain Index (`Overstrain_Index` = Tool Wear × Torque):**
   - *Rationale:* Overstrain Failure (OSF) is governed by the product of tool wear time and cutting torque.
4. **Torque-to-Speed Ratio (`Torque_Speed_Ratio` = Torque / Speed):**
   - *Rationale:* Identifies high-load low-speed bind-ups and abnormal mechanical drag.

---

## 6. Model Evaluation & Comparison

Models were evaluated using **5-Fold Stratified Cross-Validation** and tested on an independent hold-out test set (**2,000 samples**: 1,932 Normal, 68 Failures).

### Comprehensive Metrics Table:

| Metric | Logistic Regression (Balanced) | Random Forest Classifier (Best Model) |
| :--- | :---: | :---: |
| **Accuracy** | 86.10% | **98.05%** |
| **Recall / Sensitivity (TPR)** | 88.24% | **89.71%** (61/68 failures detected) |
| **Precision (PPV)** | 18.18% | **65.59%** |
| **Specificity (TNR)** | 86.02% | **98.34%** |
| **F1-Score** | 0.3015 (5-Fold CV: 0.2699) | **0.7578** (5-Fold CV: 0.7659) |
| **ROC-AUC Score** | 0.9385 (5-Fold CV: 0.9232) | **0.9869** (5-Fold CV: 0.9815) |
| **PR-AUC (Avg Precision)** | 0.4183 | **0.8574** |
| **Balanced Accuracy** | 87.13% | **94.03%** |
| **False Positive Rate (FPR)** | 13.98% (270 false alarms) | **1.66%** (Only 32 false alarms) |
| **False Negative Rate (FNR)** | 11.76% (8 missed) | **10.29%** (7 missed) |

### Test Set Confusion Matrix (Random Forest):
```
                       Predicted Normal (0)    Predicted Failure (1)
Actual Normal (0)            1,900 (TN)               32 (FP)
Actual Failure (1)               7 (FN)               61 (TP)
```

### Diagnostic Visualizations Generated (`reports/figures/`):
1. **`roc_curves_comparison.png`:** Receiver Operating Characteristic curves comparing Random Forest (AUC = 0.9869) vs. Logistic Regression (AUC = 0.9385).
2. **`precision_recall_curves.png`:** Precision-Recall curves showing Random Forest achieving an Average Precision of 0.8574 on minority failure detection.
3. **`confusion_matrices_side_by_side.png`:** Side-by-side heatmaps illustrating the drastic reduction in false alarms from 270 (Logistic Regression) to 32 (Random Forest).
4. **`feature_importance_bar_chart.png`:** Complete ranking of all 11 model features showing dominant contributions from `Torque_Speed_Ratio` (17.41%), `Rotational Speed` (16.68%), `Power_kW` (14.40%), and `Tool Wear` (13.32%).
5. **`sensor_distributions_and_failure_modes.png`:** Telemetry boxplots and failure mode breakdown counts across TWF, HDF, PWF, OSF, and RNF.

---

## 7. Web Application Architecture

The single-page web dashboard provides:
- **Telemetry Form & Quick Presets:** Allows manual sensor inputs or 1-click test scenarios (*Normal Run*, *Overstrain Risk*, *Thermal Risk*, *Power Overload*).
- **Instant Risk Status Banner:** Prominent color indicator (*Normal Operation* in green vs. *Failure Risk Detected* in crimson red) with probability percentages.
- **Subsystem Stress Gauges:** Real-time visual progress bars for Overstrain Load, Tool Life Expended, Power Envelope Stress, and Thermal Stress.
- **Model Comparison & Confusion Matrix:** Displays comparative accuracy, recall, precision, F1-score, and confusion matrix cell counts on the same screen.
- **Interactive Diagnostic Modal:** Provides 1-click pop-up viewing of high-resolution ROC, Precision-Recall, Confusion Matrix, and Distribution graphs.

---

## 8. Conclusion & Future Scope

The project demonstrates a production-grade machine learning pipeline for industrial predictive maintenance. By leveraging physics-based domain features and optimizing for Recall, the system detects ~90% of all machine failures while reducing false alarms to 1.66%.

### Future Improvements:
1. **IoT Stream Ingestion:** Live telemetry ingestion via MQTT / OPC-UA.
2. **Remaining Useful Life (RUL):** Regression modeling for hours remaining before replacement.
3. **Multi-class Failure Classification:** Simultaneously predicting specific failure modes (HDF, PWF, OSF, TWF).
