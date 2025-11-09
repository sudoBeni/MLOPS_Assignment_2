# MLOps Project 2 - GLUE Task Fine-tuning

This project contains code for fine-tuning transformer models on GLUE tasks using PyTorch Lightning and Weights & Biases for experiment tracking.

## Project Structure

```
├── GLUEDataModule.py      # Lightning DataModule for GLUE datasets
├── GLUETransformer.py     # Lightning Module for transformer model training
├── train.py              # Single training run with CLI arguments (Task 1)
├── sweep.py              # W&B hyperparameter sweep script
├── pyproject.toml        # Project dependencies
└── README.md            # This file
```

## Files Overview

### GLUEDataModule.py
Contains the `GLUEDataModule` class that handles:
- Loading GLUE datasets (cola, sst2, mrpc, qqp, etc.)
- Tokenization and preprocessing
- Creating train/validation/test dataloaders

### GLUETransformer.py
Contains the `GLUETransformer` class that handles:
- Model initialization (DistilBERT, BERT, etc.)
- Training and validation steps
- Metrics computation
- Optimizer and scheduler configuration
- Support for multiple optimizers (AdamW, Adam, SGD)
- Support for multiple schedulers (linear, cosine, constant)

### train.py (Task 1)
Command-line training script for running single experiments:
- Accepts hyperparameters as CLI arguments
- Configurable checkpoint directory
- W&B logging with custom run names
- Full control over all hyperparameters

### sweep.py
Automated hyperparameter optimization script:
- W&B sweep configuration
- Bayesian optimization
- Runs multiple trials (default: 20)
- Uses modular GLUEDataModule and GLUETransformer classes

## Installation

Install dependencies using:
```bash
pip install -e .
```

Or install specific packages:
```bash
pip install datasets evaluate ipywidgets lightning torch transformers wandb
```

## Usage

### Task 1: Single Training Run

Run a single training with specific hyperparameters:

```bash
# Basic usage
python train.py --checkpoint_dir models --lr 1e-3

# Advanced usage with all options
python train.py \
    --lr 3e-5 \
    --max_seq_length 192 \
    --scheduler cosine \
    --optimizer adamw \
    --epochs 3 \
    --train_batch_size 32 \
    --checkpoint_dir checkpoints \
    --run_name my_experiment
```

**Available arguments:**
- `--lr, --learning_rate`: Learning rate (default: 2e-5)
- `--max_seq_length`: Maximum sequence length (default: 128)
- `--scheduler`: LR scheduler type [linear, cosine, constant] (default: linear)
- `--optimizer`: Optimizer type [adamw, adam, sgd] (default: adamw)
- `--epochs`: Number of training epochs (default: 3)
- `--train_batch_size`: Training batch size (default: 32)
- `--checkpoint_dir`: Directory to save checkpoints (default: checkpoints)
- `--run_name`: Custom W&B run name (auto-generated if not provided)
- `--tags`: Tags for W&B run (default: ['training', 'task1'])
- `--no_wandb`: Disable W&B logging
- See `python train.py --help` for all options

### Hyperparameter Sweep

Run automated hyperparameter optimization:

```bash
python sweep.py
```

This executes a W&B sweep with 20 trials optimizing:
- Learning rate (2e-5 to 5e-5)
- Learning rate scheduler (linear, cosine)
- Max sequence length (128, 160, 192, 224)

## Hyperparameters

The following hyperparameters can be configured in `train.py`:
1. **Learning rate** (`--lr`): Controls the step size during optimization
2. **Warmup steps** (`--warmup_steps`): Number of steps for learning rate warmup
3. **Weight decay** (`--weight_decay`): L2 regularization strength
4. **Train batch size** (`--train_batch_size`): Number of samples per training batch
5. **Max sequence length** (`--max_seq_length`): Maximum token length for inputs
6. **LR scheduler type** (`--scheduler`): Learning rate schedule (linear, cosine, constant)
7. **Optimizer type** (`--optimizer`): Optimization algorithm (adamw, adam, sgd)
8. **Adam beta1** (`--adam_beta1`): First momentum parameter for Adam
9. **Adam beta2** (`--adam_beta2`): Second momentum parameter for Adam
10. **Adam epsilon** (`--adam_epsilon`): Numerical stability constant for Adam

## Examples

```bash
# Conservative training with low learning rate
python train.py --lr 2e-5 --scheduler linear --checkpoint_dir models/conservative

# Aggressive training with high learning rate and cosine schedule
python train.py --lr 5e-5 --scheduler cosine --max_seq_length 192 --checkpoint_dir models/aggressive

# SGD optimizer with longer sequences
python train.py --optimizer sgd --lr 1e-4 --max_seq_length 224 --checkpoint_dir models/sgd

# Disable W&B logging for local testing
python train.py --no_wandb --epochs 1 --checkpoint_dir models/test
```

## Task 1 Completion

✅ **Task 1 is now complete:**
- ✅ Jupyter notebook adapted to Python scripts (modular design)
- ✅ Single training run with `train.py`
- ✅ Command-line arguments for all hyperparameters
- ✅ `--checkpoint_dir` argument for saving models
- ✅ W&B experiment tracking integrated
- ✅ Example usage: `python train.py --checkpoint_dir models --lr 1e-3`
