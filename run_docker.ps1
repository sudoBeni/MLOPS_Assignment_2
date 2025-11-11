# PowerShell script to build and run Docker container for Task 2
# MLOps Project 2 - Best Hyperparameters Training

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "MLOps Project 2 - Task 2: Docker Training" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Configuration
$IMAGE_NAME = "mlops-project2"
$CONTAINER_NAME = "mlops-training-task2"

# Best hyperparameters from Project 1
Write-Host "Best Hyperparameters:" -ForegroundColor Yellow
Write-Host "  Learning Rate: 0.00002" -ForegroundColor White
Write-Host "  Scheduler: linear" -ForegroundColor White
Write-Host "  Max Seq Length: 128" -ForegroundColor White
Write-Host "  Optimizer: adamw" -ForegroundColor White
Write-Host "  Batch Size: 16" -ForegroundColor White
Write-Host "  Warmup Steps: 50" -ForegroundColor White
Write-Host "  Weight Decay: 0.05`n" -ForegroundColor White

# Check if Docker is running
Write-Host "Checking Docker..." -ForegroundColor Yellow
$dockerCheck = docker version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "X Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    Write-Host "`nSteps to fix:" -ForegroundColor Yellow
    Write-Host "1. Open Docker Desktop application" -ForegroundColor White
    Write-Host "2. Wait for it to fully start (whale icon should be stable)" -ForegroundColor White
    Write-Host "3. Run this script again`n" -ForegroundColor White
    exit 1
}
Write-Host "OK Docker is running`n" -ForegroundColor Green

# Remove old container if exists
Write-Host "Cleaning up old containers..." -ForegroundColor Yellow
docker rm -f $CONTAINER_NAME 2>$null

# Build Docker image
Write-Host "`nBuilding Docker image..." -ForegroundColor Yellow
Write-Host "This may take a few minutes (downloading dependencies)...`n" -ForegroundColor Gray
docker build -t ${IMAGE_NAME}:latest .

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nX Docker build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`nOK Docker image built successfully!`n" -ForegroundColor Green

# Ask user if they want to enable W&B logging
Write-Host "Do you want to enable Weights & Biases logging? (y/N): " -ForegroundColor Yellow -NoNewline
$wandb_choice = Read-Host

$wandb_args = "--no_wandb"
if ($wandb_choice -eq "y" -or $wandb_choice -eq "Y") {
    Write-Host "Enter your W&B API key (or press Enter to use cached login): " -ForegroundColor Yellow -NoNewline
    $wandb_key = Read-Host
    if ($wandb_key) {
        $wandb_args = "--wandb_key $wandb_key"
    }
    else {
        $wandb_args = ""
    }
}

# Run container
Write-Host "`nStarting training container..." -ForegroundColor Yellow
Write-Host "Container name: $CONTAINER_NAME" -ForegroundColor Gray
Write-Host "Checkpoints will be saved to: ./checkpoints/`n" -ForegroundColor Gray

# Create checkpoints directory if it doesn't exist
New-Item -ItemType Directory -Force -Path ./checkpoints | Out-Null

# Run with custom command if W&B is enabled
if ($wandb_args -ne "--no_wandb") {
    docker run --name $CONTAINER_NAME -v "${PWD}/checkpoints:/app/checkpoints" $IMAGE_NAME sh -c "python train.py --lr 0.00002 --scheduler linear --max_seq_length 128 --optimizer adamw --train_batch_size 16 --warmup_steps 50 --weight_decay 0.05 --epochs 3 --checkpoint_dir /app/checkpoints --run_name docker_best_params --tags docker task2 best_params $wandb_args"
}
else {
    docker run --name $CONTAINER_NAME -v "${PWD}/checkpoints:/app/checkpoints" $IMAGE_NAME
}

# Check exit status
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "OK Training completed successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Checkpoints saved to: ./checkpoints/" -ForegroundColor White
    Write-Host "`nTo view logs: docker logs $CONTAINER_NAME" -ForegroundColor Gray
}
else {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "X Training failed!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "View logs: docker logs $CONTAINER_NAME" -ForegroundColor Gray
    exit 1
}

# Clean up
Write-Host "`nCleaning up container..." -ForegroundColor Yellow
docker rm $CONTAINER_NAME 2>$null
Write-Host "OK Done!`n" -ForegroundColor Green
