# 🚀 QUICK START GUIDE - Pipeline Orchestration

## TL;DR - Run the Pipeline NOW

### Option 1: Makefile (Easiest)
```bash
make run
```

### Option 2: Python Script
```bash
python run_it.py --mode standalone
```

### Option 3: Bash Script
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

That's it! The pipeline will run all 4 dataframe scripts in order.

---

## What You Get

### 7 Files Created for You:

1. **run_it.py** - Main orchestrator (Python + Airflow DAG)
2. **run_pipeline.sh** - Bash alternative
3. **Makefile** - Simplest commands
4. **docker-compose.yml** - Airflow setup
5. **.env.example** - Environment config
6. **requirements.txt** - Python dependencies  
7. **SETUP_INSTRUCTIONS.md** - Full documentation

### Three Ways to Run:

| Method | Best For | Setup Time | Features |
|--------|----------|------------|----------|
| **Makefile** | Quick runs | None | ✅ Simple commands |
| **Python Script** | Development | `pip install -r requirements.txt` | ✅ Error handling, ✅ Reports |
| **Airflow + Docker** | Production | 5 min setup | ✅ Monitoring, ✅ Scheduling, ✅ Retries |

---

## All Available Commands (Makefile)

```bash
# Run pipeline
make run              # Run everything (Python)
make run-bash         # Run everything (Bash)

# Run individual scripts
make df-a             # Just Dataframe A
make df-b             # Just Dataframe B
make df-c             # Just Dataframe C
make df-d             # Just Dataframe D

# Check status
make check            # Verify files and status
make status           # Same as check

# Setup
make install          # Install dependencies
make dirs             # Create directories
make env              # Create .env file

# Airflow
make docker-up        # Start Airflow → http://localhost:8080
make docker-down      # Stop Airflow
make docker-logs      # View logs

# Cleanup
make clean            # Remove outputs
make clean-all        # Remove everything + Docker volumes

# Help
make help             # Show all commands
```

---

## Pipeline Flow

```
Check Prerequisites
        ↓
  Dataframe A (v2)
        ↓
        ├─────────────┐
        ↓             ↓
  Dataframe B    Dataframe D
    (v3)           (v2)
        ↓             ↓
        └─────────────┤
                      ↓
              Dataframe C
                (v3)
                      ↓
            Summary Report
```

**Time Estimates:**
- **Pilot Mode** (3 games): ~10-15 minutes total
- **Full Mode** (all games): ~3-4 hours total

---

## Pilot Mode (Recommended First Run)

Both dataframe_b and dataframe_c default to **PILOT_MODE = True**

This processes only 3 games instead of 270+ → **much faster testing!**

**To switch to full mode:**
```python
# Edit these lines in dataframe_b_v3.py and dataframe_c_v3.py
PILOT_MODE = False  # Change True to False
```

---

## Expected Outputs

After running, you'll have:

```
outputs/
├── dataframe_a/
│   └── v2.parquet
├── dataframe_b/
│   ├── v3.parquet                    # Full
│   └── v3_pilot_3games.parquet       # Pilot
├── dataframe_c/
│   ├── v3.parquet                    # Full
│   └── v3_pilot_3games.parquet       # Pilot
└── dataframe_d/
    └── v1.parquet
```

---

## Troubleshooting

**Pipeline fails?**
```bash
# Check what's missing
make check

# Try individual scripts
make df-a
make df-b
# etc.
```

**Need to start over?**
```bash
make clean
make run
```

**Dependencies missing?**
```bash
make install
```

---

## Next Steps

1. ✅ **Run pipeline with pilot mode** (fast iteration)
2. ✅ **Develop attention mechanism** using pilot data
3. ✅ **Test and validate** your model
4. ✅ **Switch to full mode** for final submission
5. ✅ **Train complete model** for Dec 16 deadline

---

## Full Documentation

See **SETUP_INSTRUCTIONS.md** for:
- Detailed Airflow setup
- Docker troubleshooting
- Performance optimization
- Advanced configuration

---

## Questions?

**Which method should I use?**
- During development: `make run` (fastest)
- For monitoring: Airflow + Docker (best visibility)
- For simplicity: Bash script (no dependencies)

**Should I use pilot mode?**
- YES for development and testing (3 games)
- NO for final submission (all games)

**How do I know it worked?**
- Check `outputs/` directory for .parquet files
- Look for "PIPELINE COMPLETE!" message
- Run `make check` to verify outputs

---

**Ready? Let's go!**

```bash
make run
```

🏈 Good luck with your Big Data Bowl submission!
