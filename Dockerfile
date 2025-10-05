# Use Python 3.13 slim image for efficiency
FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# Copy all project files to the container
COPY . .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port (Render will override with $PORT)
EXPOSE 8000

# Run the FastAPI app with uvicorn, using $PORT from environment
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]
