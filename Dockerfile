# Gunakan Python 3.9 slim sebagai base image
FROM python:3.9-slim

# Set working directory di dalam container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file aplikasi ke container
COPY app.py .

# Buat direktori untuk saved_data
RUN mkdir -p saved_data

# Copy model files (these should be present in your project directory)
COPY my_best_model.h5 .
COPY scaler.pkl .
COPY feature_info.pkl .

# Expose port 5000
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Command untuk menjalankan aplikasi dengan Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]