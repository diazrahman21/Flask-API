# Gunakan Python 3.9 untuk kompatibilitas TensorFlow
FROM python:3.9-slim

# Tetapkan direktori kerja
WORKDIR /app

# Atur environment variables untuk Flask dan Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Install dependensi sistem yang mungkin diperlukan oleh library Python
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Salin file requirements terlebih dahulu untuk caching yang lebih baik
COPY requirements.txt .

# Install dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi (termasuk .py, .h5, .pkl)
COPY . .

# Buat user non-root untuk keamanan
# 'adduser' tidak ada di slim, gunakan 'useradd'
RUN useradd --system appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port yang akan digunakan oleh Gunicorn
EXPOSE 5000

# Health check untuk memonitor status aplikasi
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# PERBAIKAN UTAMA: Jalankan aplikasi menggunakan Gunicorn untuk produksi
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "app:app"]