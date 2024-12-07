import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from Lseg.lseg_trainer import LSegModule
from Lseg.lseg_net import LSegNet

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.loggers import WandbLogger

from Lseg.data.util import get_labels, get_dataset

# Change these as required
train_dataset = get_dataset(dataset_name="coco", get_train=True)
val_dataset = get_dataset(dataset_name="coco", get_train=False)
labels = get_labels()

# Configuration
config = {
    "batch_size": 2,  # 6
    "base_lr": 0.004,
    "max_epochs": 2,
    "num_features": 512,
}

train_dataloaders = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=8)
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

# summary = ModelSummary(model, max_depth=-1)
# print(summary)

checkpoint_callback = ModelCheckpoint(
    monitor="train_loss",  # Metric to monitor
    mode="min",  # Save the model with the minimum training loss
    save_top_k=1,  # Only keep the best model
    filename="epoch={epoch}-train_loss={train_loss:.4f}",  # Filename format
    verbose=False,
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
    callbacks=[checkpoint_callback],
    logger=wandb_logger,
)

trainer.fit(model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)
