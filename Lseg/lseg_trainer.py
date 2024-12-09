import torch
import torch.nn as nn

import pytorch_lightning as pl
from torchmetrics import Accuracy, JaccardIndex


class LSegModule(pl.LightningModule):
    def __init__(self, max_epochs, model, num_classes, batch_size=1, base_lr=0.04, **kwargs):
        super().__init__()
        self.base_lr = base_lr / 16 * batch_size
        self.lr = self.base_lr
        self.max_epochs = max_epochs
        self.model = model
        self.ignore_index = 194  # `unlabeled` class index.
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_iou = JaccardIndex(task="multiclass", num_classes=num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # Optimize pretrained parameters, if built from scratch self.base_lr*10
        optimizer = torch.optim.SGD(self.model.dpt.parameters(), lr=self.base_lr, momentum=0.9, weight_decay=1e-4)
        # polynomial learning rate scheduler with decay factor 0:9.
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: pow(1.0 - epoch / self.max_epochs, 0.9)
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        img, target_val = batch

        # Pytorch's CE Loss can accept target_val with classes indices in the range [0, num_classes). 
        # No need to one-hot encode it. Note that it ignores unlabeled class. 
        out = self(img)
        loss = self.loss_fn(out, target_val)
        
        # Predictions to measure accuracy. Remove unlabeled classes cases before evaluating metrics
        preds = torch.argmax(out, dim=1)
        preds, target_val = self._filter_invalid_labels_from_predictions(preds, target_val)
        
        # Update and log training metrics
        self.train_accuracy.update(preds, target_val)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", self.train_accuracy.compute(), prog_bar=True)
        return loss

    def on_training_epoch_end(self, outputs):
        # Log epoch-level metrics
        self.log("train_acc_epoch", self.train_accuracy.compute())
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        img, target_val = batch

        # Pytorch's CE Loss can accept target_val with classes indices in the range [0, num_classes). 
        # No need to one-hot encode it. Note that it ignores unlabeled class. 
        out = self(img)
        val_loss = self.loss_fn(out, target_val)

        # Predictions to measure accuracy. Remove unlabeled classes cases before evaluating metrics
        preds = torch.argmax(out, dim=1)
        preds, target_val = self._filter_invalid_labels_from_predictions(preds, target_val)

        # Update and log validation metrics
        self.val_accuracy.update(preds, target_val)
        self.val_iou.update(preds, target_val)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_accuracy", self.val_accuracy.compute(), prog_bar=True)
        self.log("val_iou", self.val_iou.compute(), prog_bar=True)

    def on_validation_epoch_end(self):
        # Log epoch-level metrics
        self.log("val_acc_epoch", self.val_accuracy.compute())
        self.log("val_iou_epoch", self.val_iou.compute())
        self.val_accuracy.reset()
        self.val_iou.reset()

    def _filter_invalid_labels_from_predictions(self, hard_model_predictions, target_val):
        # Ignore any label indexed at 194 from target for loss computation, which is for unlabeled. 
        valid_pixels = target_val != self.ignore_index
        return hard_model_predictions[valid_pixels], target_val[valid_pixels]

