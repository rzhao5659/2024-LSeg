import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from Lseg.lseg_trainer import LSegModule
from Lseg.lseg_net import LSegNet

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.loggers import WandbLogger

from Lseg.data.util import get_labels, get_dataset


# Path to the latest checkpoint. Set to None if you don't have.
# latest_checkpoint_path = "checkpoints/checkpoint_epoch=3-val_loss=4.9235.ckpt"
latest_checkpoint_path = "checkpoints/lastest-epoch=5-step=54000.ckpt"
# latest_checkpoint_path = None

# Change these as required
train_dataset = get_dataset(dataset_name="coco", get_train=True)
val_dataset = get_dataset(dataset_name="coco", get_train=False)
labels = get_labels()

# Configuration
config = {
    "batch_size": 12,  # 6
    "base_lr": 0.04,
    "max_epochs": 50,
    "num_features": 512,
}

train_dataloaders = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8)
val_dataloaders = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=8)

net = LSegNet(
    labels=labels,
    features=config["num_features"],
)

# Initialize model
model = LSegModule(
    max_epochs=config["max_epochs"],
    model=net,
    num_classes=len(labels),
    batch_size=config["batch_size"],
    base_lr=config["base_lr"],
)

summary = ModelSummary(model, max_depth=-1)
print(summary)

best_val_checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    monitor="val_loss",  # Metric to monitor
    mode="min",  # Save the model with the minimum loss
    save_top_k=1,  # Only keep the best model
    filename="checkpoint_{epoch}-{val_loss:.4f}",  # Filename format
    verbose=False,
    save_on_train_epoch_end=True,
)

last_checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    monitor="step",  
    mode="max", 
    every_n_train_steps = 3000, 
    save_top_k=1,  # Only keep one model
    filename="lastest-{epoch}-{step}",  # Filename format
)

# Wandb logger
wandb_logger = WandbLogger(
    project="LSeg",
    log_model="all",
)

# Trainer
trainer = pl.Trainer(
    max_epochs=config["max_epochs"],
    devices=1 if torch.cuda.is_available() else "auto",  # Use GPUs if available
    accelerator="cuda" if torch.cuda.is_available() else "auto",  # Specify GPU usage
    precision=16 if torch.cuda.is_available() else 32,  # Use mixed precision if using GPU
    callbacks=[best_val_checkpoint_callback, last_checkpoint_callback],
    logger=wandb_logger,
    # limit_train_batches=1,  # For testing purposes. 
    # limit_val_batches=1,
)

# Continue training
trainer.fit(
    model,
    train_dataloaders=train_dataloaders,
    val_dataloaders=val_dataloaders,
    ckpt_path=latest_checkpoint_path,  # Resume from the latest checkpoint
)
