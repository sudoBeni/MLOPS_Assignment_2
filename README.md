# MLOps Project 2: Containerization

This repository contains the implementation for **MLOps Project 2**. The project fine-tunes **DistilBERT** on the **MRPC paraphrase detection task** using **PyTorch Lightning** and tracks experiments with **Weights & Biases**.

## Project Overview

| Task | Description |
|------|-------------|
| **Task 1** | Convert Jupyter notebook to CLI-based Python scripts
| **Task 2** | Docker image with best hyperparameters running locally
| **Task 3** | Deploy on GitHub Codespaces/Docker Playground
| **Task 4** | Public GitHub repo with documentation

## Quick Start

### Prerequisites
- Docker Desktop (for containerized training)
- Python 3.10+ with UV (for local development)
- Weights & Biases account ([get one free](https://wandb.ai))

### Option 1: Run with Docker (Recommended)

```powershell
.\run_docker.ps1
```

Follow the prompts to configure W&B logging. The script will build and run training with the best hyperparameters from Project 1.

### Option 2: Run Locally

```bash
# Install dependencies with UV (fast!)
uv sync

# Or with pip
pip install -e .

# Run training
python train.py \
    --project_name my_project \
    --run_name my_experiment \
    --tags training,local \
    --checkpoint_dir checkpoints
```

## Repository Structure

```
├── GLUEDataModule.py      # DataModule for GLUE datasets
├── GLUETransformer.py     # Model with training logic
├── train.py               # CLI training script
├── Dockerfile             # Docker image definition
├── run_docker.ps1         # Automated Docker script
├── pyproject.toml         # Dependencies (UV compatible)
└── README.md              # This file
```

## Task 1: CLI Training

Convert Jupyter notebook to modular Python scripts with CLI arguments.

### Usage

```bash
# Basic run
python train.py \
    --project_name my_project \
    --run_name experiment_1 \
    --tags training

# With hyperparameters
python train.py \
    --project_name my_project \
    --run_name high_lr \
    --tags training,tuning \
    --lr 5e-5 \
    --scheduler cosine \
    --max_seq_length 192 \
    --epochs 3
```

### Key Arguments

- `--project_name`, `--run_name`, `--tags`: Required for W&B tracking
- `--lr`: Learning rate (default: 2e-5)
- `--scheduler`: linear/cosine/constant (default: linear)
- `--max_seq_length`: Token length (default: 128)
- `--epochs`: Training epochs (default: 3)
- `--checkpoint_dir`: Save directory (default: checkpoints)
- `--no_wandb`: Disable W&B logging

Run `python train.py --help` for all options.

## Task 2: Docker Training

Docker image with best hyperparameters from Project 1.

### Best Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 2e-5 |
| Scheduler | linear |
| Sequence Length | 128 |
| Batch Size | 16 |
| Warmup Steps | 50 |
| Weight Decay | 0.05 |
| Epochs | 3 |

### Run with PowerShell Script

```powershell
.\run_docker.ps1
```

The script:
- Checks Docker status
- Builds the image
- Prompts for W&B configuration
- Runs training
- Saves checkpoints to `./checkpoints/`

### Manual Docker Commands

```bash
# Build
docker build -t mlops-project2 .

# Run (no W&B)
docker run --name training \
    -v ${PWD}/checkpoints:/app/checkpoints \
    mlops-project2

# Run (with W&B)
docker run --name training \
    -v ${PWD}/checkpoints:/app/checkpoints \
    mlops-project2 \
    --wandb_key YOUR_KEY \
    --project_name my_project \
    --run_name docker_run \
    --tags docker
```

**Training time:** 25-60 minutes

## Task 3: Cloud Deployment

Tested on:
- Local Windows machine (Docker Desktop)
- GitHub Codespaces

### GitHub Codespaces

1. Fork/clone this repository
2. Open in Codespaces: "Code" → "Codespaces" → "Create"
3. Run: `.\run_docker.ps1`

### Dependencies

Install with UV (fast):
```bash
uv sync
```

Or with pip:
```bash
pip install -e .
```

Main packages:
- pytorch-lightning==2.1.0
- transformers==4.36.0
- datasets==2.16.0
- wandb==0.16.1

Full list in `pyproject.toml`

## How to Use This Repository

**For Quick Testing:**
1. Clone: `git clone https://github.com/sudoBeni/MLOPS_Assignment_2.git`
2. Run: `.\run_docker.ps1`
3. Check W&B dashboard for results

**For Development:**
1. Install: `uv sync` (or `pip install -e .`)
2. Modify hyperparameters in `train.py` or `Dockerfile`
3. Run locally: `python train.py [args]`

## Assignment Context

Created for MLOps course assignment focusing on:
- Converting notebooks to production code
- Containerizing ML workflows
- Cloud deployment
- Documentation

---

**Author:** Benjamin Amhof, Hochschule Luzern  
**Repository:** https://github.com/sudoBeni/MLOPS_Assignment_2
