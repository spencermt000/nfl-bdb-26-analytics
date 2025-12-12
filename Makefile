# Makefile for NFL BigDataBowl 2026 Docker Workflows

.PHONY: help build up down shell clean process-all train

help:
	@echo "NFL BigDataBowl 2026 - Docker Commands"
	@echo "======================================="
	@echo ""
	@echo "Setup:"
	@echo "  make build          - Build Docker image"
	@echo "  make up             - Start container in background"
	@echo "  make down           - Stop and remove containers"
	@echo ""
	@echo "Development:"
	@echo "  make shell          - Open bash shell in container"
	@echo "  make python         - Open Python REPL in container"
	@echo ""
	@echo "Data Processing (v1 - original):"
	@echo "  make df-a           - Process dataframe A"
	@echo "  make df-b           - Process dataframe B"
	@echo "  make df-c           - Process dataframe C"
	@echo "  make df-d           - Process dataframe D"
	@echo "  make process-all    - Process all dataframes in sequence"
	@echo ""
	@echo "Data Processing (v2 - improved):"
	@echo "  make df-a-v2        - Process dataframe A v2"
	@echo "  make df-b-v2        - Process dataframe B v2"
	@echo "  make df-c-v2        - Process dataframe C v2"
	@echo "  make df-d-v2        - Process dataframe D v2"
	@echo "  make process-all-v2 - Process all v2 dataframes in sequence"
	@echo ""
	@echo "Training:"
	@echo "  make train          - Train adjacency matrix model"
	@echo "  make train-pilot    - Train in pilot mode (faster)"
	@echo ""
	@echo "Utilities:"
	@echo "  make logs           - View container logs"
	@echo "  make clean          - Remove outputs and caches"
	@echo "  make rebuild        - Clean rebuild of image"

# Build the Docker image
build:
	docker-compose build

# Start container in background
up:
	docker-compose up -d

# Stop and remove containers
down:
	docker-compose down

# Open interactive bash shell
shell:
	docker-compose run --rm nfl-analytics bash

# Open Python REPL
python:
	docker-compose run --rm nfl-analytics python

# Process individual dataframes (v1)
df-a:
	docker-compose run --rm nfl-analytics python scripts/dataframe_a.py

df-b:
	docker-compose run --rm nfl-analytics python scripts/dataframe_b.py

df-c:
	docker-compose run --rm nfl-analytics python scripts/dataframe_c.py

df-d:
	docker-compose run --rm nfl-analytics python scripts/dataframe_d.py

# Process individual dataframes (v2 - improved versions)
df-a-v2:
	docker-compose run --rm nfl-analytics python scripts/dataframe_a_v2.py

df-b-v2:
	docker-compose run --rm nfl-analytics python scripts/dataframe_b_v2.py

df-c-v2:
	docker-compose run --rm nfl-analytics python scripts/dataframe_c_v2.py

df-d-v2:
	docker-compose run --rm nfl-analytics python scripts/dataframe_d_v2.py

# Process all dataframes in sequence (v1)
process-all:
	@echo "Processing all dataframes..."
	docker-compose run --rm nfl-analytics bash -c "\
		python scripts/dataframe_a.py && \
		echo '✓ Dataframe A complete' && \
		python scripts/dataframe_b.py && \
		echo '✓ Dataframe B complete' && \
		python scripts/dataframe_c.py && \
		echo '✓ Dataframe C complete' && \
		python scripts/dataframe_d.py && \
		echo '✓ Dataframe D complete' && \
		echo '✓✓✓ All dataframes processed!'"

# Process all dataframes in sequence (v2 - improved versions)
process-all-v2:
	@echo "Processing all v2 dataframes..."
	docker-compose run --rm nfl-analytics bash -c "\
		python scripts/dataframe_a_v2.py && \
		echo '✓ Dataframe A v2 complete' && \
		python scripts/dataframe_b_v2.py && \
		echo '✓ Dataframe B v2 complete' && \
		python scripts/dataframe_c_v2.py && \
		echo '✓ Dataframe C v2 complete' && \
		python scripts/dataframe_d_v2.py && \
		echo '✓ Dataframe D v2 complete' && \
		echo '✓✓✓ All v2 dataframes processed!'"

# Train models
train:
	docker-compose run --rm nfl-analytics python scripts/train_adjacency_matrix_v1.py

train-pilot:
	docker-compose run --rm nfl-analytics python scripts/train_adjacency_matrix_v1.py --pilot

# View logs
logs:
	docker-compose logs -f

# Clean outputs and caches
clean:
	@echo "Cleaning Python caches..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaning output directories..."
	rm -rf outputs/dataframe_*/v*.parquet 2>/dev/null || true
	rm -rf model_outputs/*.pt 2>/dev/null || true
	@echo "✓ Cleaned"

# Rebuild image from scratch
rebuild: down
	docker-compose build --no-cache
	@echo "✓ Image rebuilt"

# Quick test
test:
	docker-compose run --rm nfl-analytics python -c "import pandas as pd; import numpy as np; import torch; print('✓ All imports successful')"