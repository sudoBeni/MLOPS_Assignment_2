# PowerShell script to build and run Docker container for Task 2
# MLOps Project 2 - Best Hyperparameters Training

Write-Host "========================================"
Write-Host "MLOps Project 2: Easy Build & Run"
Write-Host "========================================"

# Configuration
$IMAGE_NAME = "mlops-project2"
$CONTAINER_NAME = "mlops-training"

# Best Hyperparameters (from Project 1)
$LEARNING_RATE = "0.00002"
$SCHEDULER = "linear"
$MAX_SEQ_LENGTH = "128"
$OPTIMIZER = "adamw"
$BATCH_SIZE = "16"
$WARMUP_STEPS = "50"
$WEIGHT_DECAY = "0.05"
$EPOCHS = "3"

# Check if Docker is running
Write-Host "Checking Docker status."
$dockerCheck = docker version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker is not running. Please start Docker Desktop."
    exit 1
}
Write-Host "Docker is running"

# Remove old container if exists
Write-Host "Cleaning up old containers."
docker rm -f $CONTAINER_NAME 2>$null

# Build Docker image
Write-Host "Building Docker image."
Write-Host "This may take a few minutes (downloading dependencies)."
docker build -t ${IMAGE_NAME}:latest .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker build failed!"
    exit 1
}

Write-Host "Docker image built successfully!"

# Ask for W&B configuration
Write-Host ""
Write-Host "========== W&B Configuration =========="
Write-Host "Do you want to enable Weights & Biases logging? (y/n): "
$wandb_choice = Read-Host

$wandb_args = "--no_wandb"
if ($wandb_choice -eq "y" -or $wandb_choice -eq "Y") {
    Write-Host ""
    Write-Host "Enter your W&B API key: "
    $wandb_key = Read-Host
    
    Write-Host "Enter W&B project name: "
    $project_name = Read-Host
    while (-not $project_name) {
        Write-Host "Project name is required. Please enter a project name: "
        $project_name = Read-Host
    }
    
    Write-Host "Enter run name: "
    $run_name = Read-Host
    while (-not $run_name) {
        Write-Host "Run name is required. Please enter a run name: "
        $run_name = Read-Host
    }
    
    Write-Host "Enter tags (comma-separated): "
    $tags = Read-Host
    while (-not $tags) {
        Write-Host "Tags are required. Please enter at least one tag: "
        $tags = Read-Host
    }
    
    if ($wandb_key) {
        $wandb_args = "--wandb_key $wandb_key --project_name $project_name --run_name $run_name --tags $($tags -replace ',', ' ')"
    }
    else {
        Write-Host "No API key provided, W&B logging disabled."
        $wandb_args = "--no_wandb"
    }
}

# Run container
Write-Host "Starting container."
Write-Host "Container name: $CONTAINER_NAME"
Write-Host "Checkpoints will be saved to: ./checkpoints/"

# Create checkpoints directory if it doesn't exist
New-Item -ItemType Directory -Force -Path ./checkpoints | Out-Null

# Run with custom command if W&B is enabled
if ($wandb_args -ne "--no_wandb") {
    docker run --name $CONTAINER_NAME -v "${PWD}/checkpoints:/app/checkpoints" $IMAGE_NAME sh -c "python train.py --lr $LEARNING_RATE --scheduler $SCHEDULER --max_seq_length $MAX_SEQ_LENGTH --optimizer $OPTIMIZER --train_batch_size $BATCH_SIZE --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY --epochs $EPOCHS --checkpoint_dir /app/checkpoints --run_name docker_best_params --tags docker task2 best_params $wandb_args"
}
else {
    docker run --name $CONTAINER_NAME -v "${PWD}/checkpoints:/app/checkpoints" $IMAGE_NAME
}

# Check exit status
if ($LASTEXITCODE -eq 0) {
    Write-Host "========================================"
    Write-Host "Training completed successfully!"
    Write-Host "========================================"
    Write-Host "Checkpoints saved to: ./checkpoints/"
    Write-Host "To view logs: docker logs $CONTAINER_NAME"
}
else {
    Write-Host "========================================"
    Write-Host "Training failed!"
    Write-Host "========================================"
    Write-Host "View logs: docker logs $CONTAINER_NAME"
    exit 1
}

# Clean up
Write-Host "Cleaning up container."
docker rm $CONTAINER_NAME 2>$null
Write-Host "All cleaned up!"
