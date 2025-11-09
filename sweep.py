import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from GLUEDataModule import GLUEDataModule
from GLUETransformer import GLUETransformer

wandb.login(key="3b8fb613ce4af5ffb82486f87379678bd7550244", relogin=True)

# Define the dataset constant
DATASET = "mrpc"

sweep_config = {
    'method': 'bayes',  # Bayesian optimization
    'metric': {
        'name': 'accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        # === TUNE THESE 3 HYPERPARAMETERS (same as Week 2) ===
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 2e-5,
            'max': 5e-5
        },
        'lr_scheduler_type': {
            'values': ['linear', 'cosine']
        },
        'max_seq_length': {
            'values': [128, 160, 192, 224]
        },

        # === FIXED HYPERPARAMETERS (exactly like Week 2) ===
        'warmup_steps': {'value': 0},
        'weight_decay': {'value': 0.005},
        'train_batch_size': {'value': 32},
        'optimizer_type': {'value': 'adamw'},
    }
}

# ----------------------------------------------------------------------------
# TRAINING FUNCTION - EVERYTHING DEFINED INSIDE
# ----------------------------------------------------------------------------
sweep_run_counter = {'count': 0}

def train_sweep():
    """Modular training function for W&B sweep using GLUEDataModule and GLUETransformer"""

    # Increment counter
    sweep_run_counter['count'] += 1
    run_number = sweep_run_counter['count']

    # Initialize wandb
    run = wandb.init()
    config = wandb.config

    # Create run name
    run_name = (f"{DATASET}___week3_sweep___"
                f"run{run_number:02d}___"
                f"lr{config.learning_rate:.0e}_"
                f"{config.lr_scheduler_type}_"
                f"seq{config.max_seq_length}")

    wandb.run.name = run_name
    wandb.run.tags = ["week3", "sweep", "automated"]

    print(f"\n{'='*60}")
    print(f"Sweep Run {run_number}/20: {run_name}")
    print(f"{'='*60}")

    pl.seed_everything(42)

    # ========================================================================
    # INITIALIZE DATA MODULE
    # ========================================================================
    dm = GLUEDataModule(
        model_name_or_path="distilbert-base-uncased",
        task_name=DATASET,
        max_seq_length=config.max_seq_length,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.train_batch_size,
    )
    dm.setup("fit")

    # ========================================================================
    # INITIALIZE MODEL
    # ========================================================================
    model = GLUETransformer(
        model_name_or_path="distilbert-base-uncased",
        num_labels=dm.num_labels,
        task_name=DATASET,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.train_batch_size,
        lr_scheduler_type=config.lr_scheduler_type,
        optimizer_type=config.optimizer_type,
        eval_splits=dm.eval_splits,
    )

    # ========================================================================
    # SETUP TRAINER AND TRAIN
    # ========================================================================
    wandb_logger = WandbLogger(
        project="MLOPS___Sem_5___Project_01",
        name=run_name,
        tags=["week3", "sweep", "automated"]
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='accuracy',
        mode='max',
        save_top_k=1,
        filename=f'{run_name}-{{epoch:02d}}-{{accuracy:.4f}}'
    )

    trainer = pl.Trainer(
        max_epochs=3,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accelerator='auto',
        devices=1,
        enable_progress_bar=True,
        deterministic=True,
    )

    trainer.fit(model, datamodule=dm)

    print(f"✅ Completed Run {run_number}/20: {run_name}")
    wandb.finish()

# ----------------------------------------------------------------------------
# RUN THE SWEEP
# ----------------------------------------------------------------------------
print("\n" + "="*70)
print("WEEK 3: STARTING AUTOMATED HYPERPARAMETER SWEEP")
print("="*70)
print(f"Method: {sweep_config['method']}")
print(f"Metric: {sweep_config['metric']['name']} (maximize)")
print(f"Number of runs: 20 (same as Week 2)")
print("="*70 + "\n")

sweep_id = wandb.sweep(sweep=sweep_config, project="MLOPS___Sem_5___Project_01")

print(f"✅ Sweep initialized!")
print(f"📊 Sweep ID: {sweep_id}")
print(f"🔗 View live at: https://wandb.ai/benjamin-amhof-hochschule-luzern/MLOPS___Sem_5___Project_01/sweeps/{sweep_id}")
print("\n⏳ Starting sweep agent (this will run 20 trials)...\n")

wandb.agent(sweep_id, function=train_sweep, count=20)

print("\n" + "="*70)
print("🎉 WEEK 3 SWEEP COMPLETED!")
print("="*70)
print(f"Results: https://wandb.ai/benjamin-amhof-hochschule-luzern/MLOPS___Sem_5___Project_01/sweeps/{sweep_id}")
print("="*70)