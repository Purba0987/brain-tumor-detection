FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies including ffmpeg for audio conversion
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Copy application code
COPY . /app

# Ensure upload and static folders exist
RUN mkdir -p /app/uploads /app/static/uploads /app/static/annotated

EXPOSE 8000

# Use PORT environment variable provided by App Service; default to 8000
ENTRYPOINT ["sh","-c","gunicorn main:app -b 0.0.0.0:${PORT:-8000}"]
