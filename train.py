import PIL.Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import os
import PIL
from torchvision.transforms import v2

import pathlib
import torch
import pytorch_lightning as pl

from lseg_train import LSegModule
from Lseg.lseg_net import LSegNet

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.model_summary import ModelSummary

from Lseg.Lseg.coco_dataloader import load_coco_dataset


# class SegmentationDataset(Dataset):
#     def __init__(self, folder_path):
#         super().__init__()
#         self.img_files = glob.glob(os.path.join(folder_path, "semantic_img", "*.jpg"))
#         self.mask_files = []
#         for img_path in self.img_files:
#             self.mask_files.append(os.path.join(folder_path, "semantic_img_mask", pathlib.Path(img_path).stem + ".png"))

#         norm_mean = [0.5, 0.5, 0.5]
#         norm_std = [0.5, 0.5, 0.5]
#         print(f"** Using normalization mean={norm_mean} and std={norm_std} **")

#         self.transforms = v2.Compose(
#             [
#                 v2.Resize(size=(480, 480)),
#                 v2.ToTensor(),
#                 v2.ToDtype(torch.float32, scale=True),
#                 v2.Normalize(norm_mean, norm_std),
#             ]
#         )

#     def __getitem__(self, index):
#         img_path = self.img_files[index]
#         mask_path = self.mask_files[index]
#         img = PIL.Image.open(img_path)
#         label = PIL.Image.open(mask_path)
#         img = self.transforms(img)
#         label = self.transforms(label)
#         return img, label

#     def __len__(self):
#         return len(self.img_files)


def get_labels(dataset):
    labels = []
    path = "data/label_files/{}_objectInfo150.txt".format(dataset)
    assert os.path.exists(path), "*** Error : {} not exist !!!".format(path)
    f = open(path, "r")
    lines = f.readlines()
    for line in lines:
        label = line.strip().split(",")[-1].split(";")[0]
        labels.append(label)
    f.close()
    if dataset in ["ade20k"]:
        labels = labels[1:]
    return labels


# Change these as required
# train_dataset = SegmentationDataset(folder_path="data")
# val_dataset = SegmentationDataset(folder_path="data")
# labels = get_labels("ade20k")
train_dataset = load_coco_dataset(get_train=True)
val_dataset = load_coco_dataset(get_train=False)
# TODO:  get_labels from cocodataset.

# Configuration
config = {
    "batch_size": 2,  # 6
    "base_lr": 0.004,
    "max_epochs": 2,
    "num_features": 512,
}

train_dataloaders = DataLoader(
    train_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=8, pin_memory=True
)
val_dataloaders = DataLoader(
    val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=8, pin_memory=True
)

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

# Trainer
trainer = pl.Trainer(
    max_epochs=config["max_epochs"],
    devices=1 if torch.cuda.is_available() else "auto",  # Use GPUs if available
    accelerator="cuda" if torch.cuda.is_available() else "auto",  # Specify GPU usage
    precision=16 if torch.cuda.is_available() else 32,  # Use mixed precision if using GPU
    callbacks=[checkpoint_callback],
)

trainer.fit(model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)
