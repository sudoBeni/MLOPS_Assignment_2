# Example Usage for Task 1

## Quick Start

The simplest way to run training as specified in Task 1:

```bash
python train.py --checkpoint_dir models --lr 1e-3
```

This will:
- Train on the MRPC task with DistilBERT
- Use learning rate of 1e-3
- Save checkpoints to the `models/` directory
- Log to Weights & Biases

## Common Use Cases

### 1. Replicate Baseline Experiment
```bash
python train.py \
    --lr 2e-5 \
    --scheduler linear \
    --max_seq_length 128 \
    --checkpoint_dir checkpoints/baseline \
    --run_name baseline_experiment
```

### 2. High Learning Rate Experiment
```bash
python train.py \
    --lr 5e-5 \
    --scheduler cosine \
    --max_seq_length 160 \
    --checkpoint_dir checkpoints/high_lr \
    --run_name high_lr_experiment
```

### 3. SGD Optimizer Experiment
```bash
python train.py \
    --optimizer sgd \
    --lr 1e-4 \
    --scheduler linear \
    --max_seq_length 192 \
    --checkpoint_dir checkpoints/sgd \
    --run_name sgd_experiment
```

### 4. Quick Test Run (No W&B, 1 epoch)
```bash
python train.py \
    --no_wandb \
    --epochs 1 \
    --checkpoint_dir test_models \
    --run_name quick_test
```

### 5. Long Sequences with Regularization
```bash
python train.py \
    --lr 3e-5 \
    --max_seq_length 224 \
    --weight_decay 0.01 \
    --checkpoint_dir checkpoints/long_seq \
    --run_name long_sequences_regularized
```

## All Available Arguments

Run `python train.py --help` to see all options:

```
--model_name            HuggingFace model (default: distilbert-base-uncased)
--task_name             GLUE task (default: mrpc)
--lr                    Learning rate (default: 2e-5)
--warmup_steps          Warmup steps (default: 0)
--weight_decay          Weight decay (default: 0.005)
--train_batch_size      Training batch size (default: 32)
--eval_batch_size       Eval batch size (default: 32)
--max_seq_length        Max sequence length (default: 128)
--epochs                Number of epochs (default: 3)
--optimizer             Optimizer [adamw, adam, sgd] (default: adamw)
--scheduler             LR scheduler [linear, cosine, constant] (default: linear)
--adam_beta1            Adam beta1 (default: 0.9)
--adam_beta2            Adam beta2 (default: 0.999)
--adam_epsilon          Adam epsilon (default: 1e-8)
--checkpoint_dir        Checkpoint directory (default: checkpoints)
--project_name          W&B project name
--run_name              W&B run name (auto-generated if not provided)
--tags                  W&B tags (default: training task1)
--seed                  Random seed (default: 42)
--no_wandb              Disable W&B logging
--wandb_key             W&B API key
```

## Output

After training completes, you'll see:

```
🎉 TRAINING COMPLETED!
======================================================================
Best checkpoint saved to: checkpoints/your_run_name-epoch=02-accuracy=0.8578.ckpt
Last checkpoint saved to: checkpoints/your_run_name-last.ckpt
W&B Run: https://wandb.ai/your-username/project/runs/xyz123
======================================================================
```

## Checkpoints

Checkpoints are saved in the specified `--checkpoint_dir`:
- **Best model**: Highest validation accuracy
- **Last model**: Final epoch checkpoint

Filename format: `{run_name}-epoch={epoch}-accuracy={accuracy}.ckpt`
