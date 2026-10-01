"""
Inference & Prediction Service for Predictive Maintenance AI.

Loads the trained pipeline, validates input parameters, computes engineered features,
and generates predictions with failure probabilities and explainability summaries.
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
    # 1. Product Type
    product_type = str(params.get('type', 'L')).strip().upper()
    if product_type not in ['L', 'M', 'H']:
        raise ValueError(f"Invalid Product Type '{product_type}'. Allowed values: 'L', 'M', 'H'.")
        
    # Helper to convert to float safely
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
        dict: Complete prediction result with probabilities, status, and metadata.
    """
    pipeline, metrics, feature_importances = get_model()
    
    # 1. Validate inputs
    sanitized = validate_input(raw_params)
    
    # 2. Convert to DataFrame
    df_single = pd.DataFrame([sanitized])
    
    # 3. Feature engineering
    X_input, _ = prepare_features_and_target(df_single)
    
    # 4. Predict
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
    
    return {
        'prediction': pred,
        'status': status,
        'failure_probability': round(proba * 100.0, 1),
        'failure_probability_raw': proba,
        'risk_level': risk_level,
        'input_data': sanitized,
        'risk_factors': risk_factors,
        'model_name': metrics['best_model'],
        'model_metrics': metrics['models_comparison'][metrics['best_model']]['test_metrics'],
        'top_features': feature_importances[:6]
    }


if __name__ == '__main__':
    # Test normal case
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
    print(f"  Risk Level: {res_norm['risk_level']}")
    
    # Test failure case (high torque + high tool wear)
    sample_fail = {
        'type': 'L',
        'air_temperature': 302.5,
        'process_temperature': 310.5,
        'rotational_speed': 1300,
        'torque': 75.0,
        'tool_wear': 230
    }
    res_fail = predict_failure(sample_fail)
    print("\nFailure Sample Test (High wear + Torque):")
    print(f"  Status: {res_fail['status']}")
    print(f"  Failure Probability: {res_fail['failure_probability']}%")
    print(f"  Risk Level: {res_fail['risk_level']}")
    print(f"  Explanations: {res_fail['risk_factors']}")
