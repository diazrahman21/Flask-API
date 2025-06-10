from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import datetime
import os
import sys
import traceback
from tensorflow.keras.models import load_model
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for model and preprocessing objects
model = None
scaler = None
feature_info = None

def load_model_and_preprocessors():
    """Load model and preprocessing objects"""
    global model, scaler, feature_info
    
    try:
        print("🔍 Starting model loading process...")
        print(f"📁 Current working directory: {os.getcwd()}")
        print(f"📁 Files in directory: {os.listdir('.')}")
        
        # Check if files exist
        model_files = ['my_best_model.h5', 'scaler.pkl', 'feature_info.pkl']
        missing_files = [f for f in model_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            return False
        
        print("✅ All required files found")
        
        # Load model with detailed error handling
        print("📁 Loading Keras model...")
        try:
            model = load_model('my_best_model.h5', compile=False)
            print(f"✅ Model loaded successfully - Type: {type(model)}")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            print(f"   Traceback: {traceback.format_exc()}")
            return False
        
        # Load scaler
        print("📁 Loading scaler...")
        try:
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            print(f"✅ Scaler loaded successfully - Type: {type(scaler)}")
        except Exception as e:
            print(f"❌ Error loading scaler: {str(e)}")
            return False
        
        # Load feature info
        print("📁 Loading feature info...")
        try:
            with open('feature_info.pkl', 'rb') as f:
                feature_info = pickle.load(f)
            print(f"✅ Feature info loaded successfully - Type: {type(feature_info)}")
        except Exception as e:
            print(f"❌ Error loading feature info: {str(e)}")
            return False
        
        print("🎉 All components loaded successfully!")
        return True
    
    except Exception as e:
        print(f"❌ Unexpected error in load_model_and_preprocessors: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def convert_age_to_days(age_years):
    """Convert age from years to days"""
    return int(age_years * 365.25)

def convert_age_to_years(age_days):
    """Convert age from days to years"""
    return round(age_days / 365.25, 1)

# Load model and preprocessors on startup
print("🚀 Starting Cardiovascular Disease Prediction API...")
print(f"🐍 Python version: {sys.version}")
print(f"🧠 TensorFlow version: {tf.__version__}")
print(f"📁 Working directory: {os.getcwd()}")

# Try to load models immediately
if load_model_and_preprocessors():
    print("✅ Initialization successful - All models loaded")
else:
    print("⚠️ Initialization failed - Models will be loaded on first request")

@app.route('/', methods=['GET'])
def home():
    """API information endpoint"""
    return jsonify({
        "message": "Cardiovascular Disease Prediction API",
        "version": "1.0.1",
        "status": "active",
        "endpoints": {
            "POST /predict": "Make cardiovascular disease prediction",
            "GET /health": "Health check",
            "GET /model-info": "Model information",
            "POST /convert-age": "Convert age between years and days",
            "GET /debug": "Debug information"
        },
        "model_status": "loaded" if model is not None else "not_loaded",
        "components": {
            "model": "loaded" if model is not None else "not_loaded",
            "scaler": "loaded" if scaler is not None else "not_loaded", 
            "feature_info": "loaded" if feature_info is not None else "not_loaded"
        }
    })

@app.route('/debug', methods=['GET'])
def debug_info():
    """Debug information endpoint"""
    try:
        return jsonify({
            "working_directory": os.getcwd(),
            "files_in_directory": os.listdir('.'),
            "python_version": sys.version,
            "tensorflow_version": tf.__version__,
            "model_loaded": model is not None,
            "scaler_loaded": scaler is not None,
            "feature_info_loaded": feature_info is not None,
            "model_type": str(type(model)) if model is not None else "None",
            "scaler_type": str(type(scaler)) if scaler is not None else "None",
            "feature_info_type": str(type(feature_info)) if feature_info is not None else "None",
            "file_sizes": {
                "my_best_model.h5": os.path.getsize("my_best_model.h5") if os.path.exists("my_best_model.h5") else "Not found",
                "scaler.pkl": os.path.getsize("scaler.pkl") if os.path.exists("scaler.pkl") else "Not found",
                "feature_info.pkl": os.path.getsize("feature_info.pkl") if os.path.exists("feature_info.pkl") else "Not found"
            }
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Try to reload model if not loaded
    if model is None or scaler is None or feature_info is None:
        print("🔄 Health check: Attempting to reload models...")
        load_model_and_preprocessors()
    
    status = {
        "status": "healthy" if all([model is not None, scaler is not None, feature_info is not None]) else "unhealthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "feature_info_loaded": feature_info is not None,
        "working_directory": os.getcwd(),
        "available_files": os.listdir('.') if os.path.exists('.') else []
    }
    
    if all([model is not None, scaler is not None, feature_info is not None]):
        return jsonify(status), 200
    else:
        return jsonify(status), 503

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        # Try to load model
        if not load_model_and_preprocessors():
            return jsonify({
                "error": "Model not loaded and failed to load",
                "status": "error"
            }), 503
    
    try:
        return jsonify({
            "model_type": "Neural Network (Keras)",
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "total_params": int(model.count_params()),
            "layers": len(model.layers),
            "features_count": len(feature_info) if feature_info else "unknown",
            "status": "loaded"
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "error"
        }), 500

@app.route('/convert-age', methods=['POST'])
def convert_age():
    """Convert age between years and days"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No data provided",
                "status": "error"
            }), 400
        
        if 'years' in data:
            years = data['years']
            if not isinstance(years, (int, float)) or years <= 0:
                return jsonify({
                    "error": "Years must be a positive number",
                    "status": "error"
                }), 400
            
            days = convert_age_to_days(years)
            return jsonify({
                "input": f"{years} tahun",
                "output": f"{days} hari",
                "conversion": "years to days",
                "status": "success"
            })
        elif 'days' in data:
            days = data['days']
            if not isinstance(days, (int, float)) or days <= 0:
                return jsonify({
                    "error": "Days must be a positive number",
                    "status": "error"
                }), 400
            
            years = convert_age_to_years(days)
            return jsonify({
                "input": f"{days} hari",
                "output": f"{years} tahun",
                "conversion": "days to years",
                "status": "success"
            })
        else:
            return jsonify({
                "error": "Provide either 'years' or 'days' in request",
                "status": "error"
            }), 400
            
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "error"
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make cardiovascular disease prediction"""
    try:
        # Validasi model - try to load if not loaded
        if model is None or scaler is None or feature_info is None:
            print("🔄 Predict: Models not loaded, attempting to load...")
            if not load_model_and_preprocessors():
                return jsonify({
                    "error": "Model tidak tersedia dan gagal dimuat",
                    "status": "error"
                }), 500
        
        # Ambil data dari request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No data provided",
                "status": "error"
            }), 400
        
        # Validasi input
        required_fields = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                          'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "error": f"Field '{field}' wajib diisi",
                    "status": "error"
                }), 400
        
        # Validasi ranges
        age_years = data['age']
        if not isinstance(age_years, (int, float)) or age_years < 1 or age_years > 120:
            return jsonify({
                "error": "Age harus antara 1-120 tahun",
                "status": "error"
            }), 400
        
        if data['gender'] not in [0, 1]:
            return jsonify({
                "error": "Gender harus 0 (female) atau 1 (male)",
                "status": "error"
            }), 400
        
        if data['height'] <= 0 or data['weight'] <= 0:
            return jsonify({
                "error": "Height dan weight harus lebih besar dari 0",
                "status": "error"
            }), 400
        
        if data['ap_hi'] <= data['ap_lo']:
            return jsonify({
                "error": "Systolic BP (ap_hi) harus lebih besar dari Diastolic BP (ap_lo)",
                "status": "error"
            }), 400
        
        # Validasi categorical fields
        for field in ['cholesterol', 'gluc']:
            if data[field] not in [1, 2, 3]:
                return jsonify({
                    "error": f"{field} harus 1, 2, atau 3",
                    "status": "error"
                }), 400
        
        for field in ['smoke', 'alco', 'active']:
            if data[field] not in [0, 1]:
                return jsonify({
                    "error": f"{field} harus 0 atau 1",
                    "status": "error"
                }), 400
        
        # Convert age to days
        age_days = convert_age_to_days(age_years)
        
        # Prepare input data
        input_data = np.array([[
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
        
        # Interpretasi hasil
        if prediction == 1:
            result_message = "RISIKO TINGGI - Disarankan konsultasi dengan dokter"
            interpretation = "Berdasarkan data yang diberikan, terdapat indikasi risiko penyakit kardiovaskular"
        else:
            result_message = "RISIKO RENDAH - Pertahankan gaya hidup sehat"
            interpretation = "Berdasarkan data yang diberikan, risiko penyakit kardiovaskular relatif rendah"
        
        # Calculate BMI
        bmi = round(data['weight'] / ((data['height']/100) ** 2), 2)
        
        response = {
            "prediction": int(prediction),
            "confidence": round(float(confidence), 4),
            "probability": round(float(prediction_prob), 4),
            "result": result_message,
            "interpretation": interpretation,
            "input_data": {
                "age_years": age_years,
                "age_days": age_days,
                "gender": "Male" if data['gender'] == 1 else "Female",
                "height": data['height'],
                "weight": data['weight'],
                "bmi": bmi,
                "blood_pressure": f"{data['ap_hi']}/{data['ap_lo']}",
                "cholesterol_level": ["Normal", "Above Normal", "Well Above Normal"][data['cholesterol']-1],
                "glucose_level": ["Normal", "Above Normal", "Well Above Normal"][data['gluc']-1],
                "smoking": "Yes" if data['smoke'] == 1 else "No",
                "alcohol": "Yes" if data['alco'] == 1 else "No",
                "physical_activity": "Yes" if data['active'] == 1 else "No"
            },
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "success"
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "error"
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist",
        "status": "error"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method not allowed",
        "message": "The request method is not allowed for this endpoint",
        "status": "error"
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "status": "error"
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=False, host='0.0.0.0', port=port)