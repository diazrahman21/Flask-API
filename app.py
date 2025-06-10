from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import json
import datetime
import csv
import os
from tensorflow.keras.models import load_model
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
DATA_FOLDER = "saved_data"
CSV_FILE = os.path.join(DATA_FOLDER, "predictions.csv")

# Create data folder if not exists
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# Global variables for model and preprocessing objects
model = None
scaler = None
feature_info = None

def load_model_and_preprocessors():
    """Load model and preprocessing objects"""
    global model, scaler, feature_info
    
    try:
        # Check if files exist
        model_files = ['my_best_model.h5', 'scaler.pkl', 'feature_info.pkl']
        missing_files = [f for f in model_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            return False
        
        # Load model
        print("📁 Loading model...")
        model = load_model('my_best_model.h5')
        print("✅ Model loaded successfully")
        
        # Load scaler
        print("📁 Loading scaler...")
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Scaler loaded successfully")
        
        # Load feature info
        print("📁 Loading feature info...")
        with open('feature_info.pkl', 'rb') as f:
            feature_info = pickle.load(f)
        print("✅ Feature info loaded successfully")
        
        return True
    
    except Exception as e:
        print(f"❌ Error loading model/preprocessors: {str(e)}")
        print(f"📁 Current directory files: {os.listdir('.')}")
        return False

def convert_age_to_days(age_years):
    """Convert age from years to days"""
    return int(age_years * 365.25)

def convert_age_to_years(age_days):
    """Convert age from days to years"""
    return round(age_days / 365.25, 1)

def save_prediction_data(input_data, prediction, confidence):
    """Save prediction data to CSV"""
    try:
        # Create CSV header if file doesn't exist
        file_exists = os.path.exists(CSV_FILE)
        
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as file:
            fieldnames = ['id', 'timestamp', 'age_years', 'age_days', 'gender', 'height', 
                         'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 
                         'alco', 'active', 'prediction', 'confidence', 'risk_category']
            
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            # Generate ID
            data_id = int(datetime.datetime.now().timestamp() * 1000000) % 1000000
            
            # Determine risk category
            if prediction == 1:
                if confidence >= 0.8:
                    risk_category = "High Risk"
                elif confidence >= 0.6:
                    risk_category = "Medium-High Risk"
                else:
                    risk_category = "Medium Risk"
            else:
                if confidence >= 0.8:
                    risk_category = "Low Risk"
                elif confidence >= 0.6:
                    risk_category = "Low-Medium Risk"
                else:
                    risk_category = "Medium Risk"
            
            # Prepare data
            row_data = {
                'id': data_id,
                'timestamp': datetime.datetime.now().isoformat(),
                'age_years': input_data.get('age', 0),
                'age_days': convert_age_to_days(input_data.get('age', 0)),
                'gender': input_data.get('gender', 0),
                'height': input_data.get('height', 0),
                'weight': input_data.get('weight', 0),
                'ap_hi': input_data.get('ap_hi', 0),
                'ap_lo': input_data.get('ap_lo', 0),
                'cholesterol': input_data.get('cholesterol', 0),
                'gluc': input_data.get('gluc', 0),
                'smoke': input_data.get('smoke', 0),
                'alco': input_data.get('alco', 0),
                'active': input_data.get('active', 0),
                'prediction': int(prediction),
                'confidence': round(confidence, 4),
                'risk_category': risk_category
            }
            
            writer.writerow(row_data)
            return data_id
            
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        return None

def get_all_saved_data():
    """Get all saved prediction data"""
    try:
        if not os.path.exists(CSV_FILE):
            return []
        
        data = []
        with open(CSV_FILE, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Convert numeric fields
                for field in ['id', 'age_years', 'age_days', 'gender', 'height', 'weight',
                             'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco',
                             'active', 'prediction']:
                    if field in row and row[field]:
                        row[field] = int(row[field])
                
                if 'confidence' in row and row['confidence']:
                    row['confidence'] = float(row['confidence'])
                
                data.append(row)
        
        return data
    
    except Exception as e:
        print(f"Error reading data: {str(e)}")
        return []

# Load model and preprocessors on startup
print("🚀 Starting application...")
print(f"📁 Working directory: {os.getcwd()}")
print(f"📁 Files available: {os.listdir('.')}")

@app.route('/', methods=['GET'])
def home():
    """API information endpoint"""
    return jsonify({
        "message": "Cardiovascular Disease Prediction API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "POST /predict": "Make cardiovascular disease prediction",
            "GET /health": "Health check",
            "GET /model-info": "Model information",
            "GET /saved-data": "View saved predictions",
            "GET /saved-data/<id>": "Get specific prediction by ID",
            "GET /saved-data/stats": "Get prediction statistics",
            "POST /convert-age": "Convert age between years and days",
            "GET /export-data": "Export predictions to CSV"
        },
        "model_status": "loaded" if model is not None else "not_loaded"
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Try to reload model if not loaded
    if model is None:
        print("🔄 Attempting to reload model...")
        load_model_and_preprocessors()
    
    status = {
        "status": "healthy" if all([model is not None, scaler is not None, feature_info is not None]) else "unhealthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "feature_info_loaded": feature_info is not None,
        "current_directory": os.getcwd(),
        "files_in_directory": os.listdir('.') if os.path.exists('.') else []
    }
    
    if all([model is not None, scaler is not None, feature_info is not None]):
        return jsonify(status), 200
    else:
        return jsonify(status), 503

@app.route('/predict', methods=['POST'])
def predict():
    """Make cardiovascular disease prediction"""
    try:
        # Validasi model
        if model is None or scaler is None or feature_info is None:
            return jsonify({
                "error": "Model tidak tersedia",
                "status": "error"
            }), 500
        
        # Ambil data dari request
        data = request.get_json()
        
        # Validasi input
        required_fields = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                          'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "error": f"Field '{field}' wajib diisi",
                    "status": "error"
                }), 400
        
        # Convert age to days
        age_years = data['age']
        age_days = convert_age_to_days(age_years)
        
        # Prepare input data
        input_data = np.array([
            [
            age_days,
            data['gender'],
            data['height'],
            data['weight'],
            data['ap_hi'],
            data['ap_lo'],
            data['cholesterol'],
            data['gluc'],
            data['smoke'],
            data['alco'],
            data['active']
        ]])
        
        # Scale input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction_prob = model.predict(input_scaled)[0][0]
        prediction = 1 if prediction_prob > 0.5 else 0
        confidence = prediction_prob if prediction == 1 else (1 - prediction_prob)
        
        # Save prediction data
        saved_id = save_prediction_data(data, prediction, confidence)
        
        response = {
            "prediction": int(prediction),
            "confidence": round(float(confidence), 4),
            "probability": round(float(prediction_prob), 4),
            "result": "RISIKO TINGGI" if prediction == 1 else "RISIKO RENDAH",
            "saved_id": saved_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "success"
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

# Initialize model loading
if load_model_and_preprocessors():
    print("✅ All models loaded successfully")
else:
    print("⚠️ Model loading failed - will retry on first request")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=False, host='0.0.0.0', port=port)