"""
Training script for GLUE task fine-tuning with command-line arguments.
Task 1: Single training run with configurable hyperparameters.

Usage:
    python train.py --checkpoint_dir models --lr 1e-3
    python train.py --lr 3e-5 --max_seq_length 192 --scheduler cosine
"""

import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
from GLUEDataModule import GLUEDataModule
from GLUETransformer import GLUETransformer


def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='Train a transformer model on GLUE tasks')
    
    # Model and task configuration
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased',
                        help='HuggingFace model name (default: distilbert-base-uncased)')
    parser.add_argument('--task_name', type=str, default='mrpc',
                        help='GLUE task name (default: mrpc)')
    
    # Training hyperparameters
    parser.add_argument('--lr', '--learning_rate', type=float, default=2e-5, dest='learning_rate',
                        help='Learning rate (default: 2e-5)')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Number of warmup steps (default: 0)')
    parser.add_argument('--weight_decay', type=float, default=0.005,
                        help='Weight decay (default: 0.005)')
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='Training batch size (default: 32)')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='Evaluation batch size (default: 32)')
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='Maximum sequence length (default: 128)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs (default: 3)')
    
    # Optimizer and scheduler
    parser.add_argument('--optimizer', type=str, default='adamw', 
                        choices=['adamw', 'adam', 'sgd'],
                        help='Optimizer type (default: adamw)')
    parser.add_argument('--scheduler', type=str, default='linear',
                        choices=['linear', 'cosine', 'constant'],
                        help='Learning rate scheduler type (default: linear)')
    parser.add_argument('--adam_beta1', type=float, default=0.9,
                        help='Adam beta1 parameter (default: 0.9)')
    parser.add_argument('--adam_beta2', type=float, default=0.999,
                        help='Adam beta2 parameter (default: 0.999)')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8,
                        help='Adam epsilon parameter (default: 1e-8)')
    
    # Paths and logging
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints (default: checkpoints)')
    parser.add_argument('--project_name', type=str, required=True,
                        help='W&B project name')
    parser.add_argument('--run_name', type=str, required=True,
                        help='W&B run name')
    parser.add_argument('--tags', type=str, nargs='+', required=True,
                        help='Tags for W&B run')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable W&B logging')
    parser.add_argument('--wandb_key', type=str, default=None,
                        help='W&B API key (optional)')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed for reproducibility
    pl.seed_everything(args.seed)
    
    # Login to W&B if key provided
    if args.wandb_key and not args.no_wandb:
        wandb.login(key=args.wandb_key, relogin=True)
    
    # Generate run name if not provided
    if args.run_name is None:
        args.run_name = (f"{args.task_name}___lr{args.learning_rate:.0e}___"
                        f"{args.scheduler}___seq{args.max_seq_length}___"
                        f"bs{args.train_batch_size}")
    
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Model: {args.model_name}")
    print(f"Task: {args.task_name}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Batch Size: {args.train_batch_size}")
    print(f"Max Seq Length: {args.max_seq_length}")
    print(f"Epochs: {args.epochs}")
    print(f"Checkpoint Dir: {args.checkpoint_dir}")
    print(f"Run Name: {args.run_name}")
    print("="*70 + "\n")
    
    # ========================================================================
    # INITIALIZE DATA MODULE
    # ========================================================================
    print("📊 Loading dataset...")
    dm = GLUEDataModule(
        model_name_or_path=args.model_name,
        task_name=args.task_name,
        max_seq_length=args.max_seq_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    dm.setup("fit")
    print(f"✅ Dataset loaded: {len(dm.dataset['train'])} train, "
          f"{len(dm.dataset['validation'])} validation examples\n")
    
    # ========================================================================
    # INITIALIZE MODEL
    # ========================================================================
    print("🤖 Initializing model...")
    model = GLUETransformer(
        model_name_or_path=args.model_name,
        num_labels=dm.num_labels,
        task_name=args.task_name,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        lr_scheduler_type=args.scheduler,
        optimizer_type=args.optimizer,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        eval_splits=dm.eval_splits,
    )
    print(f"✅ Model initialized with {dm.num_labels} labels\n")
    
    # ========================================================================
    # SETUP LOGGER AND CALLBACKS
    # ========================================================================
    logger = None
    if not args.no_wandb:
        print("📝 Setting up W&B logger...")
        logger = WandbLogger(
            project=args.project_name,
            name=args.run_name,
            tags=args.tags,
            config=vars(args)
        )
        print("✅ W&B logger configured\n")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f'{args.run_name}-{{epoch:02d}}-{{accuracy:.4f}}',
        monitor='accuracy',
        mode='max',
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    
    # ========================================================================
    # SETUP TRAINER AND TRAIN
    # ========================================================================
    print("🚀 Starting training...\n")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator='auto',
        devices=1,
        enable_progress_bar=True,
        deterministic=True,
        log_every_n_steps=10,
    )
    
    trainer.fit(model, datamodule=dm)
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETED!")
    print("="*70)
    print(f"Best checkpoint saved to: {checkpoint_callback.best_model_path}")
    print(f"Last checkpoint saved to: {checkpoint_callback.last_model_path}")
    if logger:
        print(f"W&B Run: {wandb.run.url}")
    print("="*70 + "\n")
    
    # Finish W&B run
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
