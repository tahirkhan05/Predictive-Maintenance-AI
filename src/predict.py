"""
Inference & Prediction Service for Predictive Maintenance AI.

Loads the trained pipeline, validates input parameters, computes engineered features,
and generates predictions with failure probabilities, subsystem risk gauges, and explainability summaries.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Add project root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_engineering import prepare_features_and_target, ALL_MODEL_FEATURES

MODELS_DIR = 'models'
_PIPELINE = None
_METRICS = None
_FEATURE_IMPORTANCE = None


def get_model():
    """Lazy load singleton model pipeline and metadata."""
    global _PIPELINE, _METRICS, _FEATURE_IMPORTANCE
    if _PIPELINE is None:
        model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
        metrics_path = os.path.join(MODELS_DIR, 'metrics.json')
        feat_path = os.path.join(MODELS_DIR, 'feature_importance.json')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file '{model_path}' not found. Train the model by running 'python src/train_model.py'."
            )
            
        _PIPELINE = joblib.load(model_path)
        
        with open(metrics_path, 'r') as f:
            _METRICS = json.load(f)
            
        with open(feat_path, 'r') as f:
            _FEATURE_IMPORTANCE = json.load(f)
            
    return _PIPELINE, _METRICS, _FEATURE_IMPORTANCE


def validate_input(params: dict) -> dict:
    """
    Validate input parameters against physical domain ranges.
    
    Args:
        params: Dictionary containing raw input variables.
        
    Returns:
        dict: Sanitized and validated input values.
        
    Raises:
        ValueError: If any input is invalid or missing.
    """
    product_type = str(params.get('type', 'L')).strip().upper()
    if product_type not in ['L', 'M', 'H']:
        raise ValueError(f"Invalid Product Type '{product_type}'. Allowed values: 'L', 'M', 'H'.")
        
    def parse_float(field_name, min_val, max_val):
        val = params.get(field_name)
        if val is None or str(val).strip() == '':
            raise ValueError(f"Field '{field_name}' is required.")
        try:
            num = float(val)
        except (ValueError, TypeError):
            raise ValueError(f"Field '{field_name}' must be a valid number.")
        if num < min_val or num > max_val:
            raise ValueError(f"'{field_name}' value {num} is out of realistic physical range [{min_val}, {max_val}].")
        return num

    air_temp = parse_float('air_temperature', 280.0, 320.0)
    process_temp = parse_float('process_temperature', 290.0, 335.0)
    
    if process_temp < air_temp:
        raise ValueError(
            f"Process temperature ({process_temp} K) cannot be strictly lower than ambient Air temperature ({air_temp} K)."
        )
        
    rot_speed = parse_float('rotational_speed', 500.0, 3500.0)
    torque = parse_float('torque', 0.0, 150.0)
    tool_wear = parse_float('tool_wear', 0.0, 350.0)

    return {
        'Type': product_type,
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rot_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear
    }


def compute_subsystem_gauges(data_dict: dict) -> dict:
    """
    Calculates percentage stress gauges across distinct mechanical subsystems.
    """
    air_temp = data_dict['Air temperature [K]']
    proc_temp = data_dict['Process temperature [K]']
    speed = data_dict['Rotational speed [rpm]']
    torque = data_dict['Torque [Nm]']
    wear = data_dict['Tool wear [min]']
    
    temp_diff = proc_temp - air_temp
    power_kw = (torque * (speed * 2 * np.pi / 60.0)) / 1000.0
    overstrain = wear * torque

    # 1. Thermal Dissipation Stress (%)
    # Safe delta is >= 10K with speed > 1400. Critical if delta < 8.6K and speed < 1380.
    if temp_diff < 8.6 and speed < 1380:
        thermal_gauge = min(100.0, 75.0 + (8.6 - temp_diff) * 15.0)
    elif temp_diff < 9.5:
        thermal_gauge = 40.0 + (9.5 - temp_diff) * 30.0
    else:
        thermal_gauge = max(5.0, 30.0 - (temp_diff - 9.5) * 5.0)

    # 2. Power Envelope Stress (%)
    # Normal is 3.5 to 9.0 kW.
    if power_kw > 9.0:
        power_gauge = min(100.0, 75.0 + (power_kw - 9.0) * 15.0)
    elif power_kw < 3.5:
        power_gauge = min(100.0, 65.0 + (3.5 - power_kw) * 15.0)
    else:
        power_gauge = max(8.0, abs(power_kw - 6.0) / 3.0 * 35.0)

    # 3. Tool Overstrain Stress (%)
    # Overstrain limit is ~11000 min*Nm for L
    overstrain_gauge = min(100.0, (overstrain / 12000.0) * 100.0)

    # 4. Tool Wear Life Expended (%)
    wear_gauge = min(100.0, (wear / 240.0) * 100.0)

    return {
        'thermal_stress': round(float(thermal_gauge), 1),
        'power_stress': round(float(power_gauge), 1),
        'overstrain_stress': round(float(overstrain_gauge), 1),
        'tool_wear_expended': round(float(wear_gauge), 1),
        'temp_diff_k': round(float(temp_diff), 2),
        'power_kw': round(float(power_kw), 2),
        'overstrain_val': round(float(overstrain), 1)
    }


def analyze_risk_factors(data_dict: dict, proba: float) -> list:
    """
    Generates human-understandable explanations for what operating conditions
    contributed to the prediction result.
    """
    factors = []
    
    temp_diff = data_dict['Process temperature [K]'] - data_dict['Air temperature [K]']
    power_kw = (data_dict['Torque [Nm]'] * (data_dict['Rotational speed [rpm]'] * 2 * np.pi / 60.0)) / 1000.0
    overstrain = data_dict['Tool wear [min]'] * data_dict['Torque [Nm]']
    
    if data_dict['Tool wear [min]'] > 200:
        factors.append(f"Tool wear is elevated ({data_dict['Tool wear [min]']:.0f} min), approaching maximum blade lifespan.")
    elif data_dict['Tool wear [min]'] < 80:
        factors.append("Cutting tool has low accumulated wear (< 80 min).")
        
    if overstrain > 11000:
        factors.append(f"High Overstrain Index ({overstrain:.0f} min*Nm) - heavy torque applied to high tool wear.")
        
    if temp_diff < 8.6 and data_dict['Rotational speed [rpm]'] < 1380:
        factors.append(f"Low thermal differential ({temp_diff:.1f} K) and low airflow speed indicates heat dissipation risk.")
    elif temp_diff >= 10.0:
        factors.append(f"Thermal dissipation is optimal (Temperature differential: {temp_diff:.1f} K).")
        
    if power_kw < 3.5:
        factors.append(f"Abnormally low mechanical power output ({power_kw:.2f} kW), potential under-power condition.")
    elif power_kw > 9.0:
        factors.append(f"Excessive mechanical power consumption ({power_kw:.2f} kW), risk of power overload.")
    else:
        factors.append(f"Mechanical power consumption ({power_kw:.2f} kW) is inside standard operational window.")

    if not factors:
        if proba >= 0.5:
            factors.append("Combination of torque, speed, and wear parameters presents anomalous risk patterns.")
        else:
            factors.append("All mechanical and thermodynamic parameters operate within safe thresholds.")
            
    return factors


def predict_failure(raw_params: dict) -> dict:
    """
    Predict machine failure from input parameter dictionary.
    
    Args:
        raw_params: Dictionary of sensor variables.
        
    Returns:
        dict: Complete prediction result with probabilities, status, gauges, and metadata.
    """
    pipeline, metrics, feature_importances = get_model()
    
    sanitized = validate_input(raw_params)
    df_single = pd.DataFrame([sanitized])
    X_input, _ = prepare_features_and_target(df_single)
    
    pred = int(pipeline.predict(X_input)[0])
    proba = float(pipeline.predict_proba(X_input)[0][1])
    
    status = "Failure Risk Detected" if pred == 1 else "Normal Operation"
    
    if proba < 0.25:
        risk_level = "Low"
    elif proba < 0.50:
        risk_level = "Moderate"
    elif proba < 0.75:
        risk_level = "High"
    else:
        risk_level = "Critical"
        
    risk_factors = analyze_risk_factors(sanitized, proba)
    subsystem_gauges = compute_subsystem_gauges(sanitized)
    
    return {
        'prediction': pred,
        'status': status,
        'failure_probability': round(proba * 100.0, 1),
        'failure_probability_raw': proba,
        'risk_level': risk_level,
        'input_data': sanitized,
        'risk_factors': risk_factors,
        'subsystem_gauges': subsystem_gauges,
        'model_name': metrics['best_model'],
        'model_metrics': metrics['models_comparison'][metrics['best_model']]['test_metrics'],
        'comparison_models': metrics['models_comparison'],
        'top_features': feature_importances[:7]
    }


if __name__ == '__main__':
    sample_normal = {
        'type': 'L',
        'air_temperature': 298.1,
        'process_temperature': 308.6,
        'rotational_speed': 1551,
        'torque': 42.8,
        'tool_wear': 50
    }
    res_norm = predict_failure(sample_normal)
    print("Normal Sample Test:")
    print(f"  Status: {res_norm['status']}")
    print(f"  Failure Probability: {res_norm['failure_probability']}%")
    print(f"  Gauges: {res_norm['subsystem_gauges']}")
