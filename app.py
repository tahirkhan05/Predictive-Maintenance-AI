"""
Flask Web Application for Predictive Maintenance AI.

Serves a single-screen dashboard interface for real-time machine failure prediction,
model metrics display, and explainability visualization.
"""

import os
import sys
import json
from flask import Flask, render_template, request, jsonify

# Add project root directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.predict import predict_failure, get_model, validate_input

app = Flask(__name__)

# Sample presets for quick testing and demonstration
PRESETS = {
    "normal": {
        "name": "Standard Normal Run",
        "description": "Safe operating limits across all parameters",
        "type": "L",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551.0,
        "torque": 42.8,
        "tool_wear": 45.0
    },
    "overstrain": {
        "name": "Overstrain Failure Risk",
        "description": "High tool wear combined with high cutting torque",
        "type": "L",
        "air_temperature": 302.5,
        "process_temperature": 310.8,
        "rotational_speed": 1300.0,
        "torque": 72.0,
        "tool_wear": 225.0
    },
    "thermal": {
        "name": "Thermal Dissipation Risk",
        "description": "Low temp differential between process & air at low speed",
        "type": "M",
        "air_temperature": 303.8,
        "process_temperature": 311.2,
        "rotational_speed": 1280.0,
        "torque": 52.0,
        "tool_wear": 120.0
    },
    "power": {
        "name": "Power Overload Risk",
        "description": "Excessive mechanical power at extreme torque & speed",
        "type": "H",
        "air_temperature": 299.0,
        "process_temperature": 309.5,
        "rotational_speed": 2650.0,
        "torque": 68.0,
        "tool_wear": 160.0
    }
}


@app.route('/')
def index():
    """Render the single-page web interface."""
    _, metrics, feature_importances = get_model()
    best_model_name = metrics['best_model']
    model_stats = metrics['models_comparison'][best_model_name]['test_metrics']
    
    # Run initial default prediction for instant page completeness
    default_pred = predict_failure(PRESETS['normal'])
    
    return render_template(
        'index.html',
        model_name=best_model_name,
        metrics=model_stats,
        feature_importances=feature_importances[:6],
        initial_prediction=default_pred,
        presets=PRESETS
    )


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict machine failure from JSON or form input."""
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()
        if not data:
            return jsonify({'success': False, 'error': 'No input data provided.'}), 400
            
        result = predict_failure(data)
        return jsonify({'success': True, 'data': result})
    except ValueError as ve:
        return jsonify({'success': False, 'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': f"Internal error: {str(e)}"}), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Return model performance and feature importance metadata."""
    try:
        _, metrics, feature_importances = get_model()
        return jsonify({
            'success': True,
            'metrics': metrics,
            'feature_importances': feature_importances
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    # Initialize model on startup
    get_model()
    print("\n* Predictive Maintenance AI Web App starting...")
    print("* Access the application in your browser at: http://127.0.0.1:5000\n")
    app.run(host='127.0.0.1', port=5000, debug=False)
