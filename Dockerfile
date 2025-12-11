# Dockerfile for NFL BigDataBowl 2026 Analytics Project
# Python 3.10 with scientific computing libraries

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/train \
    data/supplementary \
    outputs/dataframe_a \
    outputs/dataframe_b \
    outputs/dataframe_c \
    outputs/dataframe_d \
    model_outputs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["python", "--version"]