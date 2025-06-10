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
load_model_and_preprocessors()

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

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({
            "error": "Model not loaded",
            "status": "error"
        }), 503
    
    try:
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        
        return jsonify({
            "model_type": "Neural Network (Keras)",
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "total_params": model.count_params(),
            "layers": len(model.layers),
            "features_count": len(feature_info) if feature_info else "unknown",
            "status": "loaded"
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

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
        
        # Validasi age range (dalam tahun)
        age_years = data['age']
        if age_years < 1 or age_years > 120:
            return jsonify({
                "error": "Age harus antara 1-120 tahun",
                "status": "error"
            }), 400
        
        # Validasi gender
        if data['gender'] not in [0, 1]:
            return jsonify({
                "error": "Gender harus 0 (female) atau 1 (male)",
                "status": "error"
            }), 400
        
        # Validasi blood pressure
        if data['ap_hi'] <= data['ap_lo']:
            return jsonify({
                "error": "Systolic BP (ap_hi) harus lebih besar dari Diastolic BP (ap_lo)",
                "status": "error"
            }), 400
        
        # Convert age to days (sesuai dengan training data)
        age_days = convert_age_to_days(age_years)
        
        # Prepare input data (gunakan age dalam hari)
        input_data = np.array([[
            age_days,  # age dalam hari
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
        
        # Save prediction data
        saved_id = save_prediction_data(data, prediction, confidence)
        
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
                "bmi": round(data['weight'] / ((data['height']/100) ** 2), 2),
                "blood_pressure": f"{data['ap_hi']}/{data['ap_lo']}",
                "cholesterol": data['cholesterol'],
                "glucose": data['gluc'],
                "smoking": "Yes" if data['smoke'] == 1 else "No",
                "alcohol": "Yes" if data['alco'] == 1 else "No",
                "physical_activity": "Yes" if data['active'] == 1 else "No"
            },
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

@app.route('/convert-age', methods=['POST'])
def convert_age():
    """Endpoint untuk convert age antara tahun dan hari"""
    try:
        data = request.get_json()
        
        if 'years' in data:
            years = data['years']
            days = convert_age_to_days(years)
            return jsonify({
                "input": f"{years} tahun",
                "output": f"{days} hari",
                "conversion": "years to days"
            })
        elif 'days' in data:
            days = data['days']
            years = convert_age_to_years(days)
            return jsonify({
                "input": f"{days} hari",
                "output": f"{years} tahun",
                "conversion": "days to years"
            })
        else:
            return jsonify({
                "error": "Provide either 'years' or 'days' in request",
                "status": "error"
            }), 400
            
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/saved-data', methods=['GET'])
def get_saved_data():
    """Get all saved prediction data"""
    try:
        all_data = get_all_saved_data()
        
        return jsonify({
            "data": all_data,
            "total_records": len(all_data),
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/saved-data/<int:data_id>', methods=['GET'])
def get_data_by_id(data_id):
    """Endpoint untuk melihat data berdasarkan ID"""
    try:
        all_data = get_all_saved_data()
        
        # Cari data berdasarkan ID
        for record in all_data:
            if record.get("id") == data_id:
                return jsonify({
                    "data": record,
                    "status": "success"
                })
        
        return jsonify({
            "error": f"Data dengan ID {data_id} tidak ditemukan",
            "status": "error"
        }), 404
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/saved-data/stats', methods=['GET'])
def get_data_stats():
    """Endpoint untuk melihat statistik data yang tersimpan"""
    try:
        all_data = get_all_saved_data()
        
        if not all_data:
            return jsonify({
                "message": "Belum ada data tersimpan",
                "stats": {
                    "total_records": 0
                },
                "status": "success"
            })
        
        # Hitung statistik
        total_records = len(all_data)
        high_risk = sum(1 for record in all_data if record.get('prediction') == 1)
        low_risk = total_records - high_risk
        
        # Statistik gender
        males = sum(1 for record in all_data if record.get('gender') == 1)
        females = total_records - males
        
        # Rata-rata age
        avg_age = sum(record.get('age_years', 0) for record in all_data) / total_records
        
        stats = {
            "total_records": total_records,
            "predictions": {
                "high_risk": high_risk,
                "low_risk": low_risk,
                "high_risk_percentage": round((high_risk / total_records) * 100, 2)
            },
            "demographics": {
                "males": males,
                "females": females,
                "average_age": round(avg_age, 1)
            }
        }
        
        return jsonify({
            "stats": stats,
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/export-data', methods=['GET'])
def export_data():
    """Endpoint untuk export data ke CSV"""
    try:
        if not os.path.exists(CSV_FILE):
            return jsonify({
                "error": "Belum ada data untuk di-export",
                "status": "error"
            }), 404
        
        return send_file(
            CSV_FILE,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'cardiovascular_predictions_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    
    except Exception as e:
        return jsonify({
            "error": str(e),
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
    # Load model on startup
    print("🚀 Starting application...")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📁 Files available: {os.listdir('.')}")

    if load_model_and_preprocessors():
        print("✅ All models loaded successfully")
        port = int(os.environ.get('PORT', 10000))
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        print("⚠️ Model loading failed - will retry on first request")