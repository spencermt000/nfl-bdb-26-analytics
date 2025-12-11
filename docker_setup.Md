# Docker Setup Guide for NFL BigDataBowl 2026

## 📋 Prerequisites

1. **Install Docker Desktop**
   - **macOS**: [Download Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
   - **Windows**: [Download Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
   - **Linux**: [Install Docker Engine](https://docs.docker.com/engine/install/)

2. **Verify Installation**
   ```bash
   docker --version
   docker-compose --version
   ```

## 🚀 Quick Start

### 1. Update Your `requirements.txt`

First, export your current venv dependencies:
```bash
# Activate your current venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Export dependencies
pip freeze > requirements.txt
```

### 2. Build the Docker Image

```bash
# Build the image (takes 2-5 minutes first time)
docker-compose build
```

### 3. Run the Container

**Option A: Interactive Shell** (recommended for development)
```bash
docker-compose run --rm nfl-analytics bash
```

**Option B: Run Specific Script**
```bash
docker-compose run --rm nfl-analytics python scripts/dataframe_a.py
```

**Option C: Keep Container Running in Background**
```bash
docker-compose up -d
docker exec -it nfl-bdb-2026 bash
```

## 📁 Project Structure

```
nfl-bdb-26-analytics/
├── Dockerfile                 # Docker image definition
├── docker-compose.yml         # Container orchestration
├── .dockerignore             # Files to exclude from image
├── requirements.txt          # Python dependencies
├── data/                     # Mounted volume
│   ├── train/
│   └── supplementary/
├── scripts/                  # Mounted volume
│   ├── utils.py
│   ├── dataframe_a.py
│   └── ...
├── outputs/                  # Mounted volume
│   ├── dataframe_a/
│   └── ...
└── model_outputs/            # Mounted volume
```

## 🔄 Common Workflows

### Run Data Processing Scripts

```bash
# Process dataframe A
docker-compose run --rm nfl-analytics python scripts/dataframe_a.py

# Process dataframe B
docker-compose run --rm nfl-analytics python scripts/dataframe_b.py

# Run all processing in sequence
docker-compose run --rm nfl-analytics bash -c "
  python scripts/dataframe_a.py && \
  python scripts/dataframe_b.py && \
  python scripts/dataframe_c.py && \
  python scripts/dataframe_d.py
"
```

### Train Models

```bash
# Run training script
docker-compose run --rm nfl-analytics python scripts/train_adjacency_matrix_v1.py

# With GPU support (if configured)
docker-compose run --rm nfl-analytics python scripts/train_adjacency_matrix_v1.py
```

### Interactive Development

```bash
# Start interactive session
docker-compose run --rm nfl-analytics bash

# Inside container:
python scripts/dataframe_a.py
python
>>> import pandas as pd
>>> df = pd.read_parquet('outputs/dataframe_a/v2.parquet')
```

### Install Additional Packages

**Option 1: Update requirements.txt and rebuild**
```bash
# Edit requirements.txt
echo "new-package>=1.0.0" >> requirements.txt

# Rebuild image
docker-compose build
```

**Option 2: Install temporarily in running container**
```bash
docker-compose run --rm nfl-analytics bash
pip install new-package
# (lost when container stops)
```

## 🗂️ Data Volume Management

Your data is **mounted as volumes** - changes persist outside the container:

```yaml
volumes:
  - ./data:/app/data              # Your data stays on host
  - ./scripts:/app/scripts        # Edit scripts on host
  - ./outputs:/app/outputs        # Outputs saved on host
```

**This means:**
- ✅ Edit scripts in VSCode/your IDE normally
- ✅ Data processing outputs save to your local machine
- ✅ No need to copy files in/out of container
- ✅ Container is disposable - rebuild anytime

## 🐛 Troubleshooting

### Container won't start
```bash
# View logs
docker-compose logs

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
```

### Permission errors
```bash
# Fix on Linux/macOS
sudo chown -R $USER:$USER ./outputs ./model_outputs
```

### Out of disk space
```bash
# Clean up old images
docker system prune -a

# Remove specific containers
docker-compose down --volumes
```

### Package import errors
```bash
# Verify requirements installed
docker-compose run --rm nfl-analytics pip list

# Rebuild with updated requirements
docker-compose build --no-cache
```

## 🎯 Advantages Over Venv

| Feature | Venv | Docker |
|---------|------|--------|
| **Reproducibility** | ❌ OS-dependent | ✅ Consistent everywhere |
| **Sharing** | ❌ Manual setup | ✅ `docker-compose up` |
| **Isolation** | ⚠️ Python only | ✅ Full system |
| **GPU Support** | ⚠️ Complex setup | ✅ Built-in |
| **Multiple Versions** | ❌ One at a time | ✅ Unlimited containers |
| **Cleanup** | ⚠️ Manual | ✅ `docker-compose down` |

## 🔧 Advanced: GPU Support (Optional)

If you have an NVIDIA GPU:

1. **Install NVIDIA Container Toolkit**
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

2. **Update Dockerfile**
   ```dockerfile
   # Use CUDA base image instead
   FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
   ```

3. **Uncomment GPU section in docker-compose.yml**

4. **Verify GPU access**
   ```bash
   docker-compose run --rm nfl-analytics nvidia-smi
   ```

## 🆘 Still Using Venv?

You can **gradually transition**:

```bash
# Method 1: Use both during transition
source venv/bin/activate  # For quick tests
docker-compose run ...    # For formal runs

# Method 2: One-time migration
docker-compose run --rm nfl-analytics python scripts/dataframe_a.py
# Check outputs look good
# Then delete venv: rm -rf venv/
```

## 📝 Next Steps

1. ✅ Build image: `docker-compose build`
2. ✅ Test run: `docker-compose run --rm nfl-analytics python --version`
3. ✅ Process data: `docker-compose run --rm nfl-analytics python scripts/dataframe_a.py`
4. ✅ Check outputs: `ls outputs/dataframe_a/`
5. ✅ Delete venv once confident: `rm -rf venv/`

## 🤝 Working with Me (Claude)

You can still work with me the same way:
- Edit files locally in your IDE
- Run processing in Docker
- Share results/errors with me
- I'll help debug or modify scripts

The Docker setup doesn't change our workflow - it just makes your environment cleaner and more reproducible!