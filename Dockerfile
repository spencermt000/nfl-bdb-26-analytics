# Use the Python 3.10 version of Airflow (REQUIRED for pandas 2.1.4)
FROM apache/airflow:2.7.3-python3.10

# 1. Switch to root to install compiler tools
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Switch back to airflow user
USER airflow

# 3. Copy requirements with correct ownership
COPY --chown=airflow:root requirements.txt /opt/airflow/requirements.txt

# 4. Install dependencies
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt