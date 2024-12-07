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
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_iou = JaccardIndex(task="multiclass", num_classes=num_classes)

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
        img, target = batch
        out = self(img)
        loss = self.loss_fn(out, target)
        preds = torch.argmax(out, dim=1)
        target_val = torch.argmax(target, dim=1)

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
        img, target = batch
        out = self(img)
        val_loss = self.loss_fn(out, target)
        preds = torch.argmax(out, dim=1)
        target_val = torch.argmax(target, dim=1)

        # Update and log validation metrics
        self.val_accuracy.update(preds, target_val)
        self.val_iou.update(preds, target_val)
        # Pixel accuracy not included
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_accuracy", self.val_accuracy.compute(), prog_bar=True)
        self.log("val_iou", self.val_iou.compute(), prog_bar=True)

    def on_validation_epoch_end(self):
        # Log epoch-level metrics
        self.log("val_acc_epoch", self.val_accuracy.compute())
        self.log("val_iou_epoch", self.val_iou.compute())
        self.val_accuracy.reset()
        self.val_iou.reset()
