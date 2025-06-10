# Cardiovascular Disease Prediction API - Docker

Flask API untuk prediksi penyakit kardiovaskular yang telah di-dockerize.

## Prerequisites

- Docker Desktop
- Docker Compose
- Python 3.9+ (untuk testing)
- Model files: `my_best_model.h5`, `scaler.pkl`, `feature_info.pkl`

## Quick Start

### Option 1: Using Build Script (Windows)

```batch
# Run the automated build and test script
build_and_test.bat
```

### Option 2: Manual Steps

#### 1. Build Docker Image

```bash
docker build -t cardio-prediction-api .
```

#### 2. Run with Docker Compose (Recommended)

```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

#### 3. Run Container Manually

```bash
docker run -d \
  --name cardio-prediction-api \
  -p 5000:5000 \
  -v $(pwd)/saved_data:/app/saved_data \
  cardio-prediction-api
```

## Testing the API

### Automated Testing

```bash
python test_api.py
```

### Manual Testing

```bash
# Health check
curl http://localhost:5000/health

# Model info
curl http://localhost:5000/model-info

# Make prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":50,"gender":1,"height":170,"weight":70,"ap_hi":120,"ap_lo":80,"cholesterol":1,"gluc":1,"smoke":0,"alco":0,"active":1}'
```

## API Endpoints

- `GET /` - API info
- `POST /predict` - Make prediction
- `GET /health` - Health check
- `GET /model-info` - Model information
- `GET /saved-data` - View saved predictions

## Docker Configuration

### Image Details

- Base: Python 3.9-slim
- Web Server: Gunicorn with 2 workers
- Port: 5000
- Health Check: Built-in every 30 seconds

### Volumes

- `./saved_data:/app/saved_data` - Persist prediction data

### Environment Variables

- `FLASK_ENV=production`
- `FLASK_DEBUG=False`

## File Structure

```
.
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose configuration
├── .dockerignore           # Docker ignore file
├── test_api.py             # API test script
├── build_and_test.bat      # Windows build script
├── my_best_model.h5        # Trained model (required)
├── scaler.pkl              # Feature scaler (required)
├── feature_info.pkl        # Feature information (required)
└── saved_data/             # Directory for saved predictions
```

## Troubleshooting

### Container Issues

```bash
# Check container status
docker ps

# View container logs
docker-compose logs cardio-api

# Restart container
docker-compose restart
```

### Port Issues

- Ensure port 5000 is not in use
- Change port mapping in docker-compose.yml if needed

### Model Files Missing

- Ensure all model files are in the project root
- Check file permissions

## Production Deployment

For production, consider:

- Using environment-specific docker-compose files
- Implementing proper logging
- Adding SSL/TLS termination
- Using container orchestration (Kubernetes, Docker Swarm)
- Implementing monitoring and alerting

## Performance

- Container uses Gunicorn with 2 workers
- Health checks every 30 seconds
- Automatic restart on failure
- Optimized Docker layers for faster builds