import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

import pytorch_lightning as pl

from .metrics import compute_accuracy


class ResnetModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=None, num_classes=10)
        resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model = resnet
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def compute_loss_and_acc(self, batch):
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = compute_accuracy(y, torch.argmax(y_hat, dim=1))
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.compute_loss_and_acc(batch)
        self.log('loss', loss)
        self.log('acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.compute_loss_and_acc(batch)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0003)
        return optimizer