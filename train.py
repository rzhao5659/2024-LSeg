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

from pytorch_lightning.utilities.model_summary import ModelSummary


class SegmentationDataset(Dataset):
    def __init__(self, folder_path):
        super().__init__()
        self.img_files = glob.glob(os.path.join(folder_path, "semantic_img", "*.jpg"))
        self.mask_files = []
        for img_path in self.img_files:
            self.mask_files.append(os.path.join(folder_path, "semantic_img_mask", pathlib.Path(img_path).stem + ".png"))

        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]
        print(f"** Using normalization mean={norm_mean} and std={norm_std} **")

        self.transforms = v2.Compose(
            [
                v2.Resize(size=(480, 480)),
                v2.ToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(norm_mean, norm_std),
            ]
        )

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        img = PIL.Image.open(img_path)
        label = PIL.Image.open(mask_path)
        img = self.transforms(img)
        label = self.transforms(label)
        return img, label

    def __len__(self):
        return len(self.img_files)


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


train_dataset = SegmentationDataset(folder_path="data")
labels = get_labels("ade20k")
# print(f"{len(labels)=}")

# Configuration
config = {
    "data_path": "./data",
    "dataset_name": "ade20k",
    "batch_size": 1,  # 16
    "base_lr": 0.004,
    "max_epochs": 5,
    "num_features": 512,
}

train_dataloaders = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False)

net = LSegNet(
    labels=labels,
    features=config["num_features"],
)

# Initialize model
model = LSegModule(
    data_path=config["data_path"],
    dataset=config["dataset_name"],
    batch_size=config["batch_size"],
    base_lr=config["base_lr"],
    max_epochs=config["max_epochs"],
    model=net,
)

summary = ModelSummary(model, max_depth=-1)
print(summary)

# for name, param in model.named_parameters():
#     print(f"Parameter: {name}, dtype: {param.dtype}, device: {param.device}")

# Trainer
trainer = pl.Trainer(
    max_epochs=config["max_epochs"],
    devices=1 if torch.cuda.is_available() else "auto",  # Use GPUs if available
    accelerator="cuda" if torch.cuda.is_available() else "auto",  # Specify GPU usage
    precision=16 if torch.cuda.is_available() else 32,  # Use mixed precision if using GPU
    log_every_n_steps=2,
)
trainer.fit(model, train_dataloaders=train_dataloaders)
