# Use Python 3.13 slim image for efficiency
FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# Copy all project files to the container
COPY . .

# Install system dependencies (for audio processing, if needed by groq)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port (Render will override with $PORT)
EXPOSE 8000

# Run the FastAPI app with uvicorn, using $PORT from environment with shell
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
