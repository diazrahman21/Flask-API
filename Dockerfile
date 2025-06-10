# Gunakan Python 3.9 slim sebagai base image
FROM python:3.9-slim

# Set working directory di dalam container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file aplikasi ke container
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=app.py

# Expose port (sesuaikan dengan Render)
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:10000/health || exit 1

# Start command
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--timeout", "120", "app:app"]