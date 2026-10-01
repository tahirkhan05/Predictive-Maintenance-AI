"""
Feature Engineering Module for Predictive Maintenance AI.

Creates domain-informed mechanical and thermodynamic features based on
physics principles of machining tools and failure modes in the AI4I 2020 dataset.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Core base features from sensor telemetry
BASE_NUMERICAL_FEATURES = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

CATEGORICAL_FEATURES = ['Type']

# Engineered domain features
ENGINEERED_FEATURES = [
    'Temp_Difference',
    'Power_kW',
    'Overstrain_Index',
    'Torque_Speed_Ratio'
]

ALL_MODEL_FEATURES = CATEGORICAL_FEATURES + BASE_NUMERICAL_FEATURES + ENGINEERED_FEATURES
TARGET_COLUMN = 'Machine failure'


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate domain-specific features for predictive maintenance.
    
    Features engineered:
    1. Temp_Difference (K): Process Temperature - Air Temperature.
       - Rationale: Reflects heat dissipation efficiency. Insufficient temperature
         differential often triggers Heat Dissipation Failure (HDF).
    2. Power_kW (kW): Mechanical power calculated as Torque * Angular Velocity.
       - Formula: Torque [Nm] * (Rotational speed [rpm] * 2 * pi / 60) / 1000.
       - Rationale: Power Failure (PWF) occurs when power draws exceed or fall below
         safe operating envelopes.
    3. Overstrain_Index (min * Nm): Tool wear multiplied by Torque.
       - Formula: Tool wear [min] * Torque [Nm].
       - Rationale: Overstrain Failure (OSF) is directly caused by heavy torque applied
         on worn cutting tools.
    4. Torque_Speed_Ratio (Nm/rpm): Ratio of Torque to Rotational Speed.
       - Formula: Torque [Nm] / (Rotational speed [rpm] + 1e-6).
       - Rationale: Identifies heavy load at low speed (potential stalls or bind-ups).
       
    Args:
        df: Input DataFrame containing base sensor columns.
        
    Returns:
        pd.DataFrame: New DataFrame with engineered columns added.
    """
    data = df.copy()
    
    # 1. Temperature Difference (Heat dissipation indicator)
    data['Temp_Difference'] = data['Process temperature [K]'] - data['Air temperature [K]']
    
    # 2. Mechanical Power in Kilowatts (Torque * angular velocity)
    angular_velocity_rad_s = data['Rotational speed [rpm]'] * (2.0 * np.pi / 60.0)
    data['Power_kW'] = (data['Torque [Nm]'] * angular_velocity_rad_s) / 1000.0
    
    # 3. Overstrain Index (Tool Wear * Applied Torque)
    data['Overstrain_Index'] = data['Tool wear [min]'] * data['Torque [Nm]']
    
    # 4. Torque to Rotational Speed Ratio
    data['Torque_Speed_Ratio'] = data['Torque [Nm]'] / (data['Rotational speed [rpm]'] + 1e-6)
    
    return data


def prepare_features_and_target(df: pd.DataFrame):
    """
    Applies feature engineering and separates feature matrix X and target y.
    
    Args:
        df: Raw or preprocessed DataFrame.
        
    Returns:
        tuple: (X DataFrame, y Series if target column exists else None)
    """
    df_feat = engineer_features(df)
    X = df_feat[ALL_MODEL_FEATURES]
    y = df_feat[TARGET_COLUMN] if TARGET_COLUMN in df_feat.columns else None
    return X, y


if __name__ == '__main__':
    from src.data_preprocessing import load_data
    df_raw = load_data()
    X, y = prepare_features_and_target(df_raw)
    print(f"Features matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print("\nEngineered features sample:")
    print(X[ENGINEERED_FEATURES].head())
