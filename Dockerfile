# ─────────────────────────────────────────────────────────────
# Dockerfile — Voice Interview Agent (FastAPI backend only)
# Used by Keploy to run the app in a container for test recording/replay
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Install OS-level audio/TTS deps
RUN apt-get update && apt-get install -y \
    espeak \
    espeak-data \
    libespeak-dev \
    libportaudio2 \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Create runtime directories
RUN mkdir -p uploads transcripts keploy

# Expose FastAPI port
EXPOSE 8000

# Start the FastAPI backend
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
