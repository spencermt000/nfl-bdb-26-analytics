# Start from the official Airflow image
FROM apache/airflow:2.7.3

# Switch to root to install system dependencies (if needed)
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the airflow user to install Python packages
USER airflow

# Copy requirements file to the container
COPY requirements.txt .

# Install Python dependencies (including matplotlib)
RUN pip install --no-cache-dir -r requirements.txt