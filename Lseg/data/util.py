import torch
import pandas as pd
from mseg.taxonomy.taxonomy_converter import TaxonomyConverter
from Lseg.data.dataset import SemData
import os
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from PIL import Image

# PATHS
DATASETS = ["coco", "ade20k"]

semantic_label_tsv_path = "mseg-api/mseg/class_remapping_files/MSeg_master.tsv"
coco_images_dir = "data/mseg_dataset/COCOPanoptic/"
coco_train_text_path = "mseg-api/mseg/dataset_lists/coco-panoptic-133-relabeled/list/train.txt"
coco_val_text_path = "mseg-api/mseg/dataset_lists/coco-panoptic-133-relabeled/list/val.txt"
ade20k_images_dir = "data/mseg_dataset/ADE20K/"
ade20k_train_text_path = "mseg-api/mseg/dataset_lists/ade20k-150-relabeled/list/train.txt"
ade20k_val_text_path = "mseg-api/mseg/dataset_lists/ade20k-150-relabeled/list/train.txt"


# This is a Callable object similar to pytorch transforms.
class ToUniversalLabel:
    def __init__(self, dataset):
        self.dataset = dataset
        self.tax_converter = TaxonomyConverter()

    def __call__(self, image, label):
        return image, self.tax_converter.transform_label(label, self.dataset)

    @staticmethod
    def read_MSeg_master(file_path):
        """
        Reads the MSeg master TSV file and returns the 'universal' column.
        """
        # Read the TSV file into a pandas DataFrame
        df = pd.read_csv(file_path, sep="\t")
        pd.set_option("display.max_rows", None)  # Set to display all rows if necessary
        return df["universal"]


# A custom transform to map 255 (unlabeled) in the label tensor, to the correct label number 194 (which is unlabeled as well)
def change_255_to_194(tensor):
    return torch.where(tensor == 255, torch.tensor(194, dtype=tensor.dtype), tensor)


def get_dataset(dataset_name: str, get_train: bool):
    """Gets validation set if get_train = False.  dataset_name must be coco or ade20k"""
    assert dataset_name in DATASETS, "Must be either coco or ade20k"
    if dataset_name == "coco":
        img_dir = coco_images_dir
        train_text_path = coco_train_text_path
        val_text_path = coco_val_text_path
        dataset_actual_name = "coco-panoptic-133-relabeled"
    else:
        img_dir = ade20k_images_dir
        train_text_path = ade20k_train_text_path
        val_text_path = ade20k_val_text_path
        dataset_actual_name = "ade20k-150-relabeled"

    img_transform = v2.Compose(
        [
            v2.ToTensor(),
            v2.Resize(size=(320, 320)),
            lambda x: v2.functional.permute_channels(x, permutation=(2, 0, 1)),  # (H,W,C) to (C,H,W)
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    label_transform = v2.Compose(
        [
            v2.ToTensor(),
            v2.Resize(size=(320, 320), interpolation=InterpolationMode.NEAREST),
            lambda x: change_255_to_194(x),  # Using lambda to apply the custom transform
        ]
    )

    together_transform = ToUniversalLabel(dataset_actual_name)

    if get_train is True:
        dataset = SemData(
            split="train",
            data_root=img_dir,
            data_list=train_text_path,
            together_transform=together_transform,
            img_transform=img_transform,
            label_transform=label_transform,
        )
    else:
        dataset = SemData(
            split="val",
            data_root=img_dir,
            data_list=val_text_path,
            together_transform=together_transform,
            img_transform=img_transform,
            label_transform=label_transform,
        )
    return dataset


def get_labels():
    """Returns universal labels as a List of strings"""
    universal_labels = ToUniversalLabel.read_MSeg_master(semantic_label_tsv_path)
    labels_list = list(universal_labels)
    return labels_list
