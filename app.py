from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Global variable to store the model
model = None

def load_model():
    """Load the trained model"""
    global model
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("Model loaded successfully")
        else:
            print("Model file not found. Please ensure model.pkl exists.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

def validate_input(data):
    """Validate input data according to dataset specifications"""
    required_fields = [
        'age', 'height', 'weight', 'gender', 'ap_hi', 'ap_lo',
        'cholesterol', 'gluc', 'smoke', 'alco', 'active'
    ]
    
    # Check if all required fields are present
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    try:
        # Validate age (in days, should be positive)
        age = int(data['age'])
        if age <= 0:
            return False, "Age must be a positive integer (in days)"
        
        # Validate height (in cm, reasonable range)
        height = int(data['height'])
        if height < 100 or height > 250:
            return False, "Height must be between 100 and 250 cm"
        
        # Validate weight (in kg, reasonable range)
        weight = float(data['weight'])
        if weight < 30 or weight > 300:
            return False, "Weight must be between 30 and 300 kg"
        
        # Validate gender (categorical code, typically 1 or 2)
        gender = int(data['gender'])
        if gender not in [1, 2]:
            return False, "Gender must be 1 or 2"
        
        # Validate blood pressure
        ap_hi = int(data['ap_hi'])
        ap_lo = int(data['ap_lo'])
        if ap_hi < 80 or ap_hi > 250:
            return False, "Systolic blood pressure must be between 80 and 250"
        if ap_lo < 40 or ap_lo > 150:
            return False, "Diastolic blood pressure must be between 40 and 150"
        if ap_hi <= ap_lo:
            return False, "Systolic pressure must be higher than diastolic pressure"
        
        # Validate cholesterol (1: normal, 2: above normal, 3: well above normal)
        cholesterol = int(data['cholesterol'])
        if cholesterol not in [1, 2, 3]:
            return False, "Cholesterol must be 1, 2, or 3"
        
        # Validate glucose (1: normal, 2: above normal, 3: well above normal)
        gluc = int(data['gluc'])
        if gluc not in [1, 2, 3]:
            return False, "Glucose must be 1, 2, or 3"
        
        # Validate binary features (0 or 1)
        binary_fields = ['smoke', 'alco', 'active']
        for field in binary_fields:
            value = int(data[field])
            if value not in [0, 1]:
                return False, f"{field} must be 0 or 1"
        
        return True, "Valid input"
        
    except (ValueError, TypeError) as e:
        return False, f"Invalid data type: {str(e)}"

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'Cardiovascular Disease Prediction API is running',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded. Please ensure model.pkl exists in the project directory.'
            }), 500
        
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        # Validate input
        is_valid, message = validate_input(data)
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': message
            }), 400
        
        # Prepare features for prediction
        features = np.array([
            int(data['age']),
            int(data['height']),
            float(data['weight']),
            int(data['gender']),
            int(data['ap_hi']),
            int(data['ap_lo']),
            int(data['cholesterol']),
            int(data['gluc']),
            int(data['smoke']),
            int(data['alco']),
            int(data['active'])
        ]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else None
        
        # Prepare response
        response = {
            'status': 'success',
            'prediction': int(prediction),
            'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
            'timestamp': datetime.now().isoformat(),
            'input_data': data
        }
        
        if probability is not None:
            response['probability'] = {
                'low_risk': float(probability[0]),
                'high_risk': float(probability[1])
            }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Prediction error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Detailed health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Cardiovascular Disease Prediction API',
        'version': '1.0.0',
        'model_status': 'loaded' if model is not None else 'not_loaded',
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'status': 'error',
        'message': 'Method not allowed'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

if __name__ == '__main__':
    load_model()
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port)