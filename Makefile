# ============================================================================
# NFL Big Data Bowl 2026 - Makefile
# ============================================================================
# 
# Simplifies running common commands
#
# USAGE:
#   make help          # Show available commands
#   make run           # Run entire pipeline (standalone)
#   make docker-up     # Start Airflow with Docker
#   make docker-down   # Stop Airflow
#   make clean         # Remove generated outputs
#
# ============================================================================

.PHONY: help run run-bash docker-up docker-down docker-logs clean check install

# Default target
.DEFAULT_GOAL := help

# Colors
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

# ============================================================================
# Help
# ============================================================================

help: ## Show this help message
	@echo ""
	@echo "$(BLUE)NFL Big Data Bowl 2026 - Pipeline Commands$(NC)"
	@echo "=============================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# ============================================================================
# Pipeline Execution
# ============================================================================

run: ## Run entire pipeline (Python standalone mode)
	@echo "$(BLUE)Running pipeline in standalone mode...$(NC)"
	python run_it.py --mode standalone

run-bash: ## Run entire pipeline (Bash script)
	@echo "$(BLUE)Running pipeline with bash script...$(NC)"
	chmod +x run_pipeline.sh
	./run_pipeline.sh

# ============================================================================
# Docker/Airflow Commands
# ============================================================================

docker-init: ## Initialize Airflow (first time setup)
	@echo "$(BLUE)Initializing Airflow...$(NC)"
	docker-compose up airflow-init

docker-up: docker-init ## Start Airflow with Docker Compose
	@echo "$(BLUE)Starting Airflow services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)Airflow UI available at: http://localhost:8080$(NC)"
	@echo "$(YELLOW)Username: airflow$(NC)"
	@echo "$(YELLOW)Password: airflow$(NC)"

docker-down: ## Stop Airflow services
	@echo "$(BLUE)Stopping Airflow services...$(NC)"
	docker-compose down

docker-restart: docker-down docker-up ## Restart Airflow services

docker-logs: ## View Airflow logs
	docker-compose logs -f

docker-clean: ## Stop and remove all containers and volumes
	@echo "$(YELLOW)This will delete all Airflow data!$(NC)"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ]
	docker-compose down -v

# ============================================================================
# Individual Dataframe Scripts
# ============================================================================

df-a: ## Run Dataframe A only
	@echo "$(BLUE)Running Dataframe A...$(NC)"
	python dataframe_a_v2.py

df-b: ## Run Dataframe B only
	@echo "$(BLUE)Running Dataframe B...$(NC)"
	python dataframe_b_v3.py

df-c: ## Run Dataframe C only
	@echo "$(BLUE)Running Dataframe C...$(NC)"
	python dataframe_c_v3.py

df-d: ## Run Dataframe D only
	@echo "$(BLUE)Running Dataframe D...$(NC)"
	python dataframe_d.py

# ============================================================================
# Utility Commands
# ============================================================================

check: ## Check prerequisites and show status
	@echo "$(BLUE)Checking prerequisites...$(NC)"
	@echo ""
	@echo "Required data files:"
	@test -f data/train/2023_input_all.parquet && echo "  $(GREEN)✓$(NC) data/train/2023_input_all.parquet" || echo "  $(YELLOW)✗$(NC) data/train/2023_input_all.parquet"
	@test -f data/supplementary_data.csv && echo "  $(GREEN)✓$(NC) data/supplementary_data.csv" || echo "  $(YELLOW)✗$(NC) data/supplementary_data.csv"
	@test -f data/sumer_bdb/sumer_coverages_player_play.parquet && echo "  $(GREEN)✓$(NC) data/sumer_bdb/sumer_coverages_player_play.parquet" || echo "  $(YELLOW)✗$(NC) data/sumer_bdb/sumer_coverages_player_play.parquet"
	@test -f data/sumer_bdb/sumer_coverages_frame.parquet && echo "  $(GREEN)✓$(NC) data/sumer_bdb/sumer_coverages_frame.parquet" || echo "  $(YELLOW)✗$(NC) data/sumer_bdb/sumer_coverages_frame.parquet"
	@echo ""
	@echo "Required scripts:"
	@test -f dataframe_a_v2.py && echo "  $(GREEN)✓$(NC) dataframe_a_v2.py" || echo "  $(YELLOW)✗$(NC) dataframe_a_v2.py"
	@test -f dataframe_b_v3.py && echo "  $(GREEN)✓$(NC) dataframe_b_v3.py" || echo "  $(YELLOW)✗$(NC) dataframe_b_v3.py"
	@test -f dataframe_c_v3.py && echo "  $(GREEN)✓$(NC) dataframe_c_v3.py" || echo "  $(YELLOW)✗$(NC) dataframe_c_v3.py"
	@test -f dataframe_d.py && echo "  $(GREEN)✓$(NC) dataframe_d.py" || echo "  $(YELLOW)✗$(NC) dataframe_d.py"
	@echo ""
	@echo "Output files:"
	@test -f outputs/dataframe_a/v2.parquet && echo "  $(GREEN)✓$(NC) outputs/dataframe_a/v2.parquet" || echo "  $(YELLOW)✗$(NC) outputs/dataframe_a/v2.parquet (not generated yet)"
	@test -f outputs/dataframe_b/v3.parquet && echo "  $(GREEN)✓$(NC) outputs/dataframe_b/v3.parquet" || echo "  $(YELLOW)✗$(NC) outputs/dataframe_b/v3.parquet (not generated yet)"
	@test -f outputs/dataframe_c/v3.parquet && echo "  $(GREEN)✓$(NC) outputs/dataframe_c/v3.parquet" || echo "  $(YELLOW)✗$(NC) outputs/dataframe_c/v3.parquet (not generated yet)"
	@test -f outputs/dataframe_d/v1.parquet && echo "  $(GREEN)✓$(NC) outputs/dataframe_d/v1.parquet" || echo "  $(YELLOW)✗$(NC) outputs/dataframe_d/v1.parquet (not generated yet)"

status: check ## Alias for check

install: ## Install Python dependencies
	@echo "$(BLUE)Installing Python dependencies...$(NC)"
	pip install -r requirements.txt

clean: ## Remove generated output files
	@echo "$(YELLOW)Removing generated outputs...$(NC)"
	rm -rf outputs/dataframe_a/*.parquet
	rm -rf outputs/dataframe_b/*.parquet
	rm -rf outputs/dataframe_c/*.parquet
	rm -rf outputs/dataframe_d/*.parquet
	@echo "$(GREEN)Clean complete!$(NC)"

clean-all: clean ## Remove all generated files and Docker volumes
	@echo "$(YELLOW)Removing all generated files and Docker volumes...$(NC)"
	docker-compose down -v 2>/dev/null || true
	@echo "$(GREEN)Clean complete!$(NC)"

# ============================================================================
# Development Helpers
# ============================================================================

dirs: ## Create necessary directories
	@echo "$(BLUE)Creating directories...$(NC)"
	mkdir -p data/train data/sumer_bdb
	mkdir -p outputs/dataframe_a outputs/dataframe_b outputs/dataframe_c outputs/dataframe_d
	mkdir -p dags logs plugins
	@echo "$(GREEN)Directories created!$(NC)"

env: ## Create .env file from template
	@if [ -f .env ]; then \
		echo "$(YELLOW).env file already exists!$(NC)"; \
	else \
		cp .env.example .env; \
		echo "$(GREEN).env file created from template$(NC)"; \
		echo "$(YELLOW)Remember to update AIRFLOW_UID in .env!$(NC)"; \
	fi
