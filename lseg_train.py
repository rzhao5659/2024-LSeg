import torch
import torch.nn as nn

import pytorch_lightning as pl


class LSegModule(pl.LightningModule):
    def __init__(self, data_path, dataset, batch_size, base_lr, max_epochs, model, **kwargs):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.base_lr = base_lr / 16 * batch_size
        self.lr = self.base_lr
        self.max_epochs = max_epochs
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=4e-3, momentum=0.9, weight_decay=1e-4)
        # polynomial learning rate scheduler with decay factor 0:9.
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: pow(1.0 - epoch / self.max_epochs, 0.9)
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        img, target = batch
        out = self(img)
        loss = self.loss_fn(out, target)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss
