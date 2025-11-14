FROM python:3.10-slim

# Prevent Python buffering
ENV PYTHONUNBUFFERED=1

# Install system deps
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 7860

# Streamlit config (fixes headless/network issues)
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run the app
CMD ["streamlit", "run", "traffic_ai.py"]
