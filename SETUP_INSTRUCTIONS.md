# NFL Big Data Bowl 2026 - Pipeline Orchestration Setup

## Overview

This setup provides **two ways** to run your data processing pipeline:

1. **Standalone Python** (Recommended for quick runs)
2. **Airflow + Docker** (Recommended for production, scheduling, monitoring)

---

## Quick Start - Standalone Mode (No Docker)

### Prerequisites
```bash
# Install Python dependencies
pip install -r requirements.txt
```

### Run Pipeline
```bash
# Run entire pipeline
python run_it.py --mode standalone

# Pipeline will execute:
# 1. Check prerequisites
# 2. Run dataframe_a_v2.py
# 3. Run dataframe_b_v3.py  
# 4. Run dataframe_d.py
# 5. Run dataframe_c_v3.py
# 6. Generate summary report
```

**Advantages:**
- ✅ Simple, no Docker required
- ✅ Fast setup
- ✅ Easy debugging
- ✅ Good for development

**Disadvantages:**
- ❌ No task scheduling
- ❌ No retry logic
- ❌ No monitoring UI
- ❌ No parallel execution

---

## Advanced Setup - Airflow + Docker

### Prerequisites

1. **Docker & Docker Compose installed**
   ```bash
   # Check installation
   docker --version
   docker-compose --version
   ```

2. **Create required directories**
   ```bash
   mkdir -p dags logs plugins outputs
   ```

3. **Set up environment variables**
   ```bash
   # Copy example file
   cp .env.example .env
   
   # Edit .env and set your user ID
   # Run: id -u
   # Then update AIRFLOW_UID in .env
   ```

4. **Copy DAG file**
   ```bash
   # Copy run_it.py to dags folder
   cp run_it.py dags/
   ```

### Start Airflow

```bash
# Initialize Airflow (first time only)
docker-compose up airflow-init

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Access Airflow UI

1. Open browser: **http://localhost:8080**
2. Login:
   - Username: `airflow`
   - Password: `airflow`
3. Find DAG: `nfl_bdb_data_pipeline`
4. Click "Trigger DAG" to run pipeline

### Monitor Pipeline

The Airflow UI shows:
- ✅ Task status (success/failure/running)
- ⏱️ Task duration
- 📊 Logs for each task
- 🔄 Retry attempts
- 📈 Historical runs

### Stop Airflow

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

**Advantages:**
- ✅ Task scheduling and monitoring
- ✅ Automatic retries on failure
- ✅ Visual pipeline monitoring
- ✅ Can run tasks in parallel
- ✅ Production-ready

**Disadvantages:**
- ❌ Requires Docker
- ❌ More complex setup
- ❌ Slower initial startup

---

## Project Structure

```
nfl-bdb-26-analytics/
├── run_it.py                   # Pipeline orchestrator
├── docker-compose.yml          # Docker services config
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
│
├── data/                       # Input data (not tracked)
│   ├── train/
│   │   └── 2023_input_all.parquet
│   ├── supplementary_data.csv
│   └── sumer_bdb/
│       ├── sumer_coverages_player_play.parquet
│       └── sumer_coverages_frame.parquet
│
├── outputs/                    # Generated data
│   ├── dataframe_a/
│   │   └── v2.parquet
│   ├── dataframe_b/
│   │   ├── v3.parquet
│   │   └── v3_pilot_3games.parquet
│   ├── dataframe_c/
│   │   ├── v3.parquet
│   │   └── v3_pilot_3games.parquet
│   └── dataframe_d/
│       └── v1.parquet
│
├── dags/                       # Airflow DAGs (copy run_it.py here)
├── logs/                       # Airflow logs
└── plugins/                    # Airflow plugins
```

---

## Pipeline Flow

```
┌─────────────────────┐
│ Check Prerequisites │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Dataframe A (v2)  │  Node-level features per frame
└──────────┬──────────┘
           │
           ├────────────────────┐
           │                    │
           ▼                    ▼
┌─────────────────────┐  ┌─────────────────────┐
│   Dataframe B (v3)  │  │   Dataframe D (v2)  │
│  Play-level + Ball  │  │   Frame-level       │
└──────────┬──────────┘  └──────────┬──────────┘
           │                        │
           ├────────────────────────┘
           │
           ▼
┌─────────────────────┐
│   Dataframe C (v3)  │  Edge-level + Ball trajectory
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Summary Report     │
└─────────────────────┘
```

**Dependencies:**
- Dataframe B depends on A (needs passer position)
- Dataframe C depends on A and B (needs both node + play features)
- Dataframe D is independent (can run parallel to B)

---

## Pilot Mode

Both dataframe_b_v3.py and dataframe_c_v3.py support **PILOT_MODE**:

```python
# At top of each script
PILOT_MODE = True   # Process only 3 games
PILOT_N_GAMES = 3
```

**When to use:**
- ✅ Development and testing
- ✅ Quick pipeline validation
- ✅ Attention model prototyping

**When to disable:**
- ✅ Final submission run
- ✅ Full model training
- ✅ Production deployment

---

## Troubleshooting

### Docker Issues

**Problem:** Port 8080 already in use
```bash
# Change port in docker-compose.yml
ports:
  - "8081:8080"  # Use 8081 instead
```

**Problem:** Permission denied
```bash
# Set correct AIRFLOW_UID
id -u  # Get your user ID
# Update .env file with your UID
```

**Problem:** Container won't start
```bash
# Check logs
docker-compose logs airflow-webserver
docker-compose logs airflow-scheduler

# Restart services
docker-compose restart
```

### Standalone Mode Issues

**Problem:** Module not found
```bash
# Ensure you're in project root
pip install -r requirements.txt
```

**Problem:** File not found
```bash
# Verify data structure
ls data/train/
ls data/sumer_bdb/
```

**Problem:** Script fails
```bash
# Run individual scripts to isolate issue
python dataframe_a_v2.py
python dataframe_b_v3.py
# etc.
```

---

## Next Steps

### After Pipeline Completes:

1. **Verify outputs**
   ```bash
   ls -lh outputs/dataframe_*/
   ```

2. **Check summary report** (printed at end)

3. **Begin model development**
   - Load pilot data for quick iteration
   - Build attention mechanism
   - Test with 3-game subset

4. **Full production run**
   - Set PILOT_MODE = False
   - Run complete pipeline
   - Train final model

---

## Performance Tips

### For Faster Processing:

1. **Use Pilot Mode** during development (3 games vs 270+)
2. **Parallel execution** with Airflow (B and D run together)
3. **Chunked processing** in scripts (already implemented)
4. **Monitor memory** usage during large runs

### Expected Times (Full Dataset):

- Dataframe A: ~30-60 minutes
- Dataframe B: ~5-10 minutes  
- Dataframe C: ~2-3 hours (most intensive)
- Dataframe D: ~2-5 minutes

### Expected Times (Pilot Mode - 3 games):

- Dataframe A: ~2-5 minutes
- Dataframe B: ~1 minute
- Dataframe C: ~5-10 minutes
- Dataframe D: ~1 minute

---

## Support

For issues with:
- **Pipeline scripts**: Check individual script documentation
- **Airflow setup**: See Airflow docs (airflow.apache.org)
- **Docker issues**: See Docker docs (docs.docker.com)

**Deadline: December 16, 2025**

Good luck with your Big Data Bowl submission! 🏈
