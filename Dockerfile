# Dockerfile for MLOps Project 2 - Task 2
# Runs training with best hyperparameters from Project 1

# Use official Python 3.10 slim image for smaller size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for better caching)
COPY pyproject.toml ./

# Install Python dependencies
# Note: Installing without GPU support (torch CPU version)
RUN pip install --no-cache-dir \
    datasets==4.2.0 \
    evaluate==0.4.6 \
    pytorch-lightning==2.1.0 \
    torch==2.5.1 \
    transformers==4.57.1 \
    wandb==0.22.2 \
    scikit-learn \
    scipy

# Copy application code
COPY GLUEDataModule.py .
COPY GLUETransformer.py .
COPY train.py .

# Create checkpoint directory
RUN mkdir -p /app/checkpoints

# Best hyperparameters from Project 1
ENV LEARNING_RATE=0.00002 \
    LR_SCHEDULER_TYPE=linear \
    MAX_SEQ_LENGTH=128 \
    OPTIMIZER_TYPE=adamw \
    TRAIN_BATCH_SIZE=16 \
    WARMUP_STEPS=50 \
    WEIGHT_DECAY=0.05 \
    EPOCHS=3

# Default command: Run training with best hyperparameters
# Can be overridden with docker run arguments
CMD ["sh", "-c", "python train.py \
    --lr ${LEARNING_RATE} \
    --scheduler ${LR_SCHEDULER_TYPE} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --optimizer ${OPTIMIZER_TYPE} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --warmup_steps ${WARMUP_STEPS} \
    --weight_decay ${WEIGHT_DECAY} \
    --epochs ${EPOCHS} \
    --checkpoint_dir /app/checkpoints \
    --run_name docker_best_params \
    --tags docker task2 best_params \
    --no_wandb"]

# Expose port for potential future use (e.g., tensorboard)
EXPOSE 6006

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"
