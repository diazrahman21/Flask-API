# Cardiovascular Disease Prediction API

API untuk prediksi risiko penyakit kardiovaskular menggunakan model Neural Network dengan TensorFlow/Keras.

## 📋 Prerequisites

- Python 3.8 atau lebih tinggi
- pip (Python package installer)
- Model files: `my_best_model.h5`, `scaler.pkl`, `feature_info.pkl`

## 🚀 Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd Flask-API
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Struktur File
Pastikan struktur folder seperti berikut:
```
Flask-API/
├── app.py
├── requirements.txt
├── README.md
├── my_best_model.h5      # Model TensorFlow
├── scaler.pkl            # Scaler untuk preprocessing
├── feature_info.pkl      # Informasi features
└── saved_data/           # Folder otomatis dibuat untuk menyimpan data
    ├── predictions.json
    └── predictions.csv
```

### 4. Jalankan API
```bash
python app.py
```

API akan berjalan di: `http://localhost:5000`

## 📡 API Endpoints

### 1. Home - Cek Status API
- **URL**: `GET /`
- **Response**: Status API dan informasi model

### 2. Health Check
- **URL**: `GET /health`
- **Response**: Status kesehatan API

### 3. Model Information
- **URL**: `GET /model-info`
- **Response**: Informasi model dan format input

### 4. Example Data
- **URL**: `GET /example`
- **Response**: Contoh data untuk testing

### 5. Prediction (Main Endpoint)
- **URL**: `POST /predict`
- **Content-Type**: `application/json`
- **Body**: JSON dengan data pasien

### 6. Convert Age
- **URL**: `POST /convert-age`
- **Body**: `{"years": 50}` atau `{"days": 18250}`

### 7. View Saved Data
- **URL**: `GET /saved-data`
- **Parameters**: `?page=1&per_page=10`

### 8. Data Statistics
- **URL**: `GET /saved-data/stats`
- **Response**: Statistik prediksi yang tersimpan

## 🔬 Testing dengan Postman

### Test 1: Cek Status API
```http
GET http://localhost:5000/
```

### Test 2: Get Example Data
```http
GET http://localhost:5000/example
```

### Test 3: Prediksi Risiko Kardiovaskular
```http
POST http://localhost:5000/predict
Content-Type: application/json

{
  "age": 50,
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
}
```

### Test 4: Contoh Data High Risk
```http
POST http://localhost:5000/predict
Content-Type: application/json

{
  "age": 65,
  "gender": 1,
  "height": 160,
  "weight": 85,
  "ap_hi": 160,
  "ap_lo": 100,
  "cholesterol": 3,
  "gluc": 2,
  "smoke": 1,
  "alco": 1,
  "active": 0
}
```

### Test 5: Convert Age
```http
POST http://localhost:5000/convert-age
Content-Type: application/json

{
  "years": 50
}
```

### Test 6: View Saved Data
```http
GET http://localhost:5000/saved-data?page=1&per_page=5
```

### Test 7: Data Statistics
```http
GET http://localhost:5000/saved-data/stats
```

## 📊 Input Format

| Field | Type | Description | Values |
|-------|------|-------------|---------|
| age | int | Umur dalam tahun | 1-120 |
| gender | int | Jenis kelamin | 1=female, 2=male |
| height | int | Tinggi badan (cm) | > 0 |
| weight | int/float | Berat badan (kg) | > 0 |
| ap_hi | int | Tekanan darah sistolik | > 0 |
| ap_lo | int | Tekanan darah diastolik | > 0 |
| cholesterol | int | Level kolesterol | 1=normal, 2=above normal, 3=well above normal |
| gluc | int | Level glukosa | 1=normal, 2=above normal, 3=well above normal |
| smoke | int | Status merokok | 0=no, 1=yes |
| alco | int | Konsumsi alkohol | 0=no, 1=yes |
| active | int | Aktivitas fisik | 0=no, 1=yes |

## 📋 Response Format

### Successful Prediction
```json
{
  "prediction": {
    "risk": 0,
    "risk_label": "Low Risk",
    "probability": 0.2345,
    "confidence": 76.55
  },
  "input_data": {
    "age": 50,
    "gender": 2,
    "height": 168,
    "weight": 62,
    "ap_hi": 110,
    "ap_lo": 80,
    "cholesterol": 1,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1,
    "age_display": "50 tahun"
  },
  "model_input": {
    "age_in_days": 18250,
    "note": "Age dikonversi ke hari untuk model prediction"
  },
  "data_saved": true,
  "status": "success"
}
```

### Error Response
```json
{
  "error": "Field 'age' wajib diisi",
  "status": "error"
}
```

## 🧪 Postman Collection

### Import ke Postman:
1. Buka Postman
2. Click Import
3. Pilih "Raw Text"
4. Copy-paste collection berikut:

```json
{
  "info": {
    "name": "Cardiovascular Disease API",
    "description": "API untuk prediksi risiko penyakit kardiovaskular"
  },
  "item": [
    {
      "name": "1. Check API Status",
      "request": {
        "method": "GET",
        "header": [],
        "url": {
          "raw": "{{base_url}}/",
          "host": ["{{base_url}}"]
        }
      }
    },
    {
      "name": "2. Get Example Data",
      "request": {
        "method": "GET",
        "header": [],
        "url": {
          "raw": "{{base_url}}/example",
          "host": ["{{base_url}}"],
          "path": ["example"]
        }
      }
    },
    {
      "name": "3. Predict Low Risk",
      "request": {
        "method": "POST",
        "header": [
          {
            "key": "Content-Type",
            "value": "application/json"
          }
        ],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"age\": 50,\n  \"gender\": 2,\n  \"height\": 168,\n  \"weight\": 62,\n  \"ap_hi\": 110,\n  \"ap_lo\": 80,\n  \"cholesterol\": 1,\n  \"gluc\": 1,\n  \"smoke\": 0,\n  \"alco\": 0,\n  \"active\": 1\n}"
        },
        "url": {
          "raw": "{{base_url}}/predict",
          "host": ["{{base_url}}"],
          "path": ["predict"]
        }
      }
    },
    {
      "name": "4. Predict High Risk",
      "request": {
        "method": "POST",
        "header": [
          {
            "key": "Content-Type",
            "value": "application/json"
          }
        ],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"age\": 65,\n  \"gender\": 1,\n  \"height\": 160,\n  \"weight\": 85,\n  \"ap_hi\": 160,\n  \"ap_lo\": 100,\n  \"cholesterol\": 3,\n  \"gluc\": 2,\n  \"smoke\": 1,\n  \"alco\": 1,\n  \"active\": 0\n}"
        },
        "url": {
          "raw": "{{base_url}}/predict",
          "host": ["{{base_url}}"],
          "path": ["predict"]
        }
      }
    },
    {
      "name": "5. View Saved Data",
      "request": {
        "method": "GET",
        "header": [],
        "url": {
          "raw": "{{base_url}}/saved-data?page=1&per_page=5",
          "host": ["{{base_url}}"],
          "path": ["saved-data"],
          "query": [
            {"key": "page", "value": "1"},
            {"key": "per_page", "value": "5"}
          ]
        }
      }
    },
    {
      "name": "6. Data Statistics",
      "request": {
        "method": "GET",
        "header": [],
        "url": {
          "raw": "{{base_url}}/saved-data/stats",
          "host": ["{{base_url}}"],
          "path": ["saved-data", "stats"]
        }
      }
    }
  ],
  "variable": [
    {
      "key": "base_url",
      "value": "http://localhost:5000"
    }
  ]
}
```

## 🐛 Troubleshooting

### Error: Model tidak tersedia
- Pastikan file `my_best_model.h5`, `scaler.pkl`, dan `feature_info.pkl` ada di folder yang sama dengan `app.py`

### Error: Module not found
- Jalankan: `pip install -r requirements.txt`

### Error: Port sudah digunakan
- Ganti port di `app.py`: `app.run(debug=True, host='0.0.0.0', port=5001)`

### Error: TensorFlow compatibility
- Install versi TensorFlow yang sesuai dengan Python version

## 📝 Notes

- **Age Input**: Input age dalam tahun (1-120), akan dikonversi otomatis ke hari untuk model
- **Data Storage**: Semua prediksi disimpan otomatis ke `saved_data/`
- **File Format**: Data tersimpan dalam format JSON dan CSV
- **Pagination**: Endpoint `/saved-data` mendukung pagination

## 📞 Support

Jika ada masalah atau pertanyaan, silakan buat issue di repository ini.