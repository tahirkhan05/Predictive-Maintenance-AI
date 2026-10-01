"""
Unit and Integration Tests for Predictive Maintenance AI Pipeline.
Uses Python's standard unittest module for maximum portability without external tool dependencies.
"""

import os
import sys
import unittest
import pandas as pd

# Add project root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import load_data, validate_data, split_data
from src.feature_engineering import engineer_features, prepare_features_and_target
from src.predict import predict_failure, validate_input, get_model


class TestDataPipeline(unittest.TestCase):
    """Tests for dataset loading, validation, and splitting."""

    def test_load_data_exists(self):
        df = load_data('data/ai4i2020.csv')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 10000)
        self.assertIn('Machine failure', df.columns)

    def test_validate_data_integrity(self):
        df = load_data('data/ai4i2020.csv')
        report = validate_data(df)
        self.assertTrue(report['is_valid'])
        self.assertEqual(report['total_missing_values'], 0)
        self.assertEqual(report['total_rows'], 10000)
        self.assertGreater(report['failure_rate_percent'], 3.0)

    def test_split_data_stratification(self):
        df = load_data('data/ai4i2020.csv')
        train_df, test_df = split_data(df, test_size=0.2, random_state=42)
        self.assertEqual(len(train_df), 8000)
        self.assertEqual(len(test_df), 2000)
        # Check stratification
        train_fail_rate = train_df['Machine failure'].mean()
        test_fail_rate = test_df['Machine failure'].mean()
        self.assertAlmostEqual(train_fail_rate, test_fail_rate, places=2)


class TestFeatureEngineering(unittest.TestCase):
    """Tests for feature engineering calculations and shapes."""

    def test_engineer_features_math(self):
        sample = pd.DataFrame([{
            'Air temperature [K]': 300.0,
            'Process temperature [K]': 310.0,
            'Rotational speed [rpm]': 1500.0,
            'Torque [Nm]': 40.0,
            'Tool wear [min]': 100.0,
            'Type': 'M',
            'Machine failure': 0
        }])
        feat_df = engineer_features(sample)
        
        # Temp Difference
        self.assertEqual(feat_df['Temp_Difference'].iloc[0], 10.0)
        # Overstrain Index
        self.assertEqual(feat_df['Overstrain_Index'].iloc[0], 4000.0)
        # Power_kW should be positive
        self.assertGreater(feat_df['Power_kW'].iloc[0], 0)
        # Torque Speed Ratio
        self.assertGreater(feat_df['Torque_Speed_Ratio'].iloc[0], 0)

    def test_prepare_features_and_target_shape(self):
        df = load_data('data/ai4i2020.csv')
        X, y = prepare_features_and_target(df)
        self.assertEqual(X.shape[0], 10000)
        self.assertEqual(X.shape[1], 10)  # 1 cat + 5 base num + 4 engineered
        self.assertIsNotNone(y)
        self.assertEqual(len(y), 10000)


class TestPredictionService(unittest.TestCase):
    """Tests for inference, input validation, and risk analysis."""

    def test_model_loaded(self):
        pipeline, metrics, feat_imp = get_model()
        self.assertIsNotNone(pipeline)
        self.assertIn('Random Forest', metrics['best_model'])
        self.assertGreater(len(feat_imp), 0)

    def test_normal_machine_prediction(self):
        normal_params = {
            'type': 'L',
            'air_temperature': 298.1,
            'process_temperature': 308.6,
            'rotational_speed': 1551.0,
            'torque': 42.8,
            'tool_wear': 40.0
        }
        res = predict_failure(normal_params)
        self.assertEqual(res['prediction'], 0)
        self.assertEqual(res['status'], 'Normal Operation')
        self.assertLess(res['failure_probability'], 25.0)
        self.assertEqual(res['risk_level'], 'Low')

    def test_failure_machine_prediction(self):
        failure_params = {
            'type': 'L',
            'air_temperature': 302.5,
            'process_temperature': 310.5,
            'rotational_speed': 1300.0,
            'torque': 75.0,
            'tool_wear': 240.0
        }
        res = predict_failure(failure_params)
        self.assertEqual(res['prediction'], 1)
        self.assertEqual(res['status'], 'Failure Risk Detected')
        self.assertGreater(res['failure_probability'], 50.0)
        self.assertGreater(len(res['risk_factors']), 0)

    def test_input_validation_invalid_type(self):
        with self.assertRaises(ValueError):
            validate_input({'type': 'Z'})

    def test_input_validation_inverted_temperatures(self):
        with self.assertRaises(ValueError):
            validate_input({
                'type': 'L',
                'air_temperature': 310.0,
                'process_temperature': 295.0,
                'rotational_speed': 1500.0,
                'torque': 40.0,
                'tool_wear': 50.0
            })


if __name__ == '__main__':
    unittest.main()
