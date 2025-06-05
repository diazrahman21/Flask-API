from flask import Flask, request, jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import os
import json
import datetime
import csv
import pandas as pd

app = Flask(__name__)

# Load model dan preprocessing objects saat aplikasi start
try:
    model = load_model("my_best_model.h5")
    
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    with open("feature_info.pkl", "rb") as f:
        feature_info = pickle.load(f)
    
    print("Model dan preprocessing objects berhasil dimuat!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None
    feature_info = None

# Buat folder untuk menyimpan data jika belum ada
DATA_FOLDER = "saved_data"
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

def save_prediction_data(input_data, prediction_result):
    """
    Simpan data input dan hasil prediksi ke file
    """
    try:
        # Buat timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Gabungkan data input dan hasil prediksi
        record = {
            "timestamp": timestamp,
            "input": input_data,
            "prediction": prediction_result,
            "id": len(get_all_saved_data()) + 1
        }
        
        # Simpan ke JSON file
        json_file = os.path.join(DATA_FOLDER, "predictions.json")
        
        # Baca data existing atau buat list baru
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                data = json.load(f)
        else:
            data = []
        
        # Tambah record baru
        data.append(record)
        
        # Simpan kembali
        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)
        
        # Simpan juga ke CSV untuk analisis lebih mudah
        save_to_csv(record)
        
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

def save_to_csv(record):
    """
    Simpan data ke CSV file
    """
    try:
        csv_file = os.path.join(DATA_FOLDER, "predictions.csv")
        
        # Flatten data untuk CSV
        flattened_data = {
            "id": record["id"],
            "timestamp": record["timestamp"],
            "age_years": record["input"]["age"],
            "age_days": record["input"]["age"] * 365,
            "gender": record["input"]["gender"],
            "height": record["input"]["height"],
            "weight": record["input"]["weight"],
            "ap_hi": record["input"]["ap_hi"],
            "ap_lo": record["input"]["ap_lo"],
            "cholesterol": record["input"]["cholesterol"],
            "gluc": record["input"]["gluc"],
            "smoke": record["input"]["smoke"],
            "alco": record["input"]["alco"],
            "active": record["input"]["active"],
            "prediction_risk": record["prediction"]["risk"],
            "prediction_label": record["prediction"]["risk_label"],
            "probability": record["prediction"]["probability"],
            "confidence": record["prediction"]["confidence"]
        }
        
        # Cek apakah file sudah ada
        file_exists = os.path.exists(csv_file)
        
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=flattened_data.keys())
            
            # Tulis header jika file baru
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(flattened_data)
        
        return True
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return False

def get_all_saved_data():
    """
    Ambil semua data yang tersimpan
    """
    try:
        json_file = os.path.join(DATA_FOLDER, "predictions.json")
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error reading data: {e}")
        return []

def convert_age_to_days(age_years):
    """
    Convert age dari tahun ke hari untuk model
    """
    return age_years * 365

def convert_age_to_years(age_days):
    """
    Convert age dari hari ke tahun untuk display
    """
    return round(age_days / 365, 1)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Cardiovascular Disease Prediction API",
        "status": "active",
        "model_loaded": model is not None,
        "note": "Input age dalam tahun, akan dikonversi otomatis ke hari untuk model"
    })

@app.route('/predict', methods=['POST'])
def predict():
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
        
        # Convert age dari tahun ke hari untuk model
        age_days = convert_age_to_days(age_years)
        
        # Buat copy data dan replace age dengan hari
        model_data = data.copy()
        model_data['age'] = age_days
        
        # Preprocessing
        numeric_features = feature_info['numeric_features']
        categorical_features = feature_info['categorical_features']
        
        # Extract features (menggunakan age dalam hari)
        numeric_input = np.array([[model_data[feat] for feat in numeric_features]])
        categorical_input = np.array([[model_data[feat] for feat in categorical_features]])
        
        # Scale numeric features
        numeric_scaled = scaler.transform(numeric_input)
        
        # Combine features
        X_new = np.concatenate([numeric_scaled, categorical_input], axis=1).astype(np.float32)
        
        # Prediksi
        prediction = model.predict(X_new, verbose=0)
        probability = float(prediction[0][0])
        risk = int(probability > 0.5)
        
        # Hasil prediksi
        prediction_result = {
            "risk": risk,
            "risk_label": "High Risk" if risk == 1 else "Low Risk",
            "probability": round(probability, 4),
            "confidence": round(probability * 100, 2) if risk == 1 else round((1 - probability) * 100, 2)
        }
        
        # Simpan data input dan hasil prediksi
        save_success = save_prediction_data(data, prediction_result)
        
        # Response
        result = {
            "prediction": prediction_result,
            "input_data": {
                **data,  # Data asli dengan age dalam tahun
                "age_display": f"{age_years} tahun"
            },
            "model_input": {
                "age_in_days": age_days,
                "note": "Age dikonversi ke hari untuk model prediction"
            },
            "data_saved": save_success,
            "status": "success"
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_status": "loaded" if model is not None else "not loaded"
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    if feature_info is None:
        return jsonify({"error": "Feature info not available"}), 500
    
    return jsonify({
        "numeric_features": feature_info['numeric_features'],
        "categorical_features": feature_info['categorical_features'],
        "total_features": len(feature_info['numeric_features']) + len(feature_info['categorical_features']),
        "model_type": "Neural Network (MLP)",
        "framework": "TensorFlow/Keras",
        "input_format": {
            "age": "Input dalam tahun (1-120), akan dikonversi ke hari secara otomatis",
            "gender": "1=female, 2=male",
            "height": "Height in cm",
            "weight": "Weight in kg",
            "ap_hi": "Systolic blood pressure",
            "ap_lo": "Diastolic blood pressure",
            "cholesterol": "1=normal, 2=above normal, 3=well above normal",
            "gluc": "1=normal, 2=above normal, 3=well above normal",
            "smoke": "0=no, 1=yes",
            "alco": "0=no, 1=yes",
            "active": "0=no, 1=yes"
        }
    })

@app.route('/example', methods=['GET'])
def example():
    """
    Contoh data untuk testing dengan age dalam tahun
    """
    return jsonify({
        "example_data": {
            "age": 50,  # Dalam tahun
            "gender": 2,
            "height": 168,
            "weight": 62,
            "ap_hi": 110,
            "ap_lo": 80,
            "cholesterol": 1,
            "gluc": 1,
            "smoke": 0,
            "alco": 0,
            "active": 1
        },
        "note": "Age input dalam tahun, akan dikonversi otomatis ke hari untuk model",
        "conversion_example": {
            "input_age_years": 50,
            "converted_age_days": 50 * 365,
            "formula": "age_days = age_years * 365"
        }
    })

@app.route('/convert-age', methods=['POST'])
def convert_age():
    """
    Endpoint untuk convert age antara tahun dan hari
    """
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
    """
    Endpoint untuk melihat semua data yang tersimpan
    """
    try:
        # Parameter untuk pagination
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        # Ambil semua data
        all_data = get_all_saved_data()
        
        # Hitung pagination
        total = len(all_data)
        start = (page - 1) * per_page
        end = start + per_page
        
        # Ambil data sesuai pagination
        paginated_data = all_data[start:end]
        
        return jsonify({
            "data": paginated_data,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "pages": (total + per_page - 1) // per_page
            },
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/saved-data/<int:data_id>', methods=['GET'])
def get_data_by_id(data_id):
    """
    Endpoint untuk melihat data berdasarkan ID
    """
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
    """
    Endpoint untuk melihat statistik data yang tersimpan
    """
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
        high_risk_count = sum(1 for record in all_data if record["prediction"]["risk"] == 1)
        low_risk_count = total_records - high_risk_count
        
        # Statistik probability
        probabilities = [record["prediction"]["probability"] for record in all_data]
        avg_probability = sum(probabilities) / len(probabilities)
        
        return jsonify({
            "stats": {
                "total_records": total_records,
                "high_risk_predictions": high_risk_count,
                "low_risk_predictions": low_risk_count,
                "high_risk_percentage": round((high_risk_count / total_records) * 100, 2),
                "average_probability": round(avg_probability, 4),
                "date_range": {
                    "first_record": all_data[0]["timestamp"] if all_data else None,
                    "last_record": all_data[-1]["timestamp"] if all_data else None
                }
            },
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/export-data', methods=['GET'])
def export_data():
    """
    Endpoint untuk export data ke CSV
    """
    try:
        csv_file = os.path.join(DATA_FOLDER, "predictions.csv")
        
        if not os.path.exists(csv_file):
            return jsonify({
                "error": "Belum ada data untuk di-export",
                "status": "error"
            }), 404
        
        return jsonify({
            "message": "Data tersedia untuk di-download",
            "file_path": csv_file,
            "note": f"File CSV tersimpan di: {os.path.abspath(csv_file)}",
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)