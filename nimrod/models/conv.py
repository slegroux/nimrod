# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/models.conv.ipynb.

# %% auto 0
__all__ = ['logger', 'ConvLayer', 'ConvNet', 'ConvNetX']

# %% ../../nbs/models.conv.ipynb 4
import torch.nn as nn
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from lightning import LightningModule, Trainer
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.callbacks import LearningRateFinder
from lightning.pytorch.loggers import CSVLogger

from torch_lr_finder import LRFinder
from torchmetrics import Accuracy
from hydra.utils import instantiate
from omegaconf import OmegaConf

from matplotlib import pyplot as plt
import pandas as pd

from ..image.datasets import MNISTDataModule
from ..utils import get_device
from .core import Classifier, find_optimal_lr

import logging
logger = logging.getLogger(__name__)

# %% ../../nbs/models.conv.ipynb 6
class ConvLayer(nn.Module):
    def __init__(self,
                in_channels:int=3, # input channels
                out_channels:int=16, # output channels
                kernel_size:int=3, # kernel size
                activation:bool=True
                ):

        super().__init__()
        self.activation = activation
        # use stride 2 for downsampling instead of max or average pooling with stride 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 2, kernel_size//2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            x = self.relu(x)
        return x

# %% ../../nbs/models.conv.ipynb 10
class ConvNet(nn.Module):
    def __init__(
            self,
            in_channels:int=1, # input channels
            out_channels:int=10 # num_classes
            ):
        super().__init__()
        self.net = nn.Sequential(
            ConvLayer(in_channels, 8, kernel_size=5), #14x14
            nn.BatchNorm2d(8),
            ConvLayer(8, 16), #7x7
            nn.BatchNorm2d(16),
            ConvLayer(16, 32), #4x4
            nn.BatchNorm2d(32),
            ConvLayer(32, 64), #2x2
            nn.BatchNorm2d(64),
            ConvLayer(64, 10, activation=False), #1x1
            nn.BatchNorm2d(10),
            nn.Flatten()

        )

    def forward(self, x:torch.Tensor # input image tensor of dimension (B, C, W, H)
                ) -> torch.Tensor: # output probs (B, N_classes)

        return self.net(x)

# %% ../../nbs/models.conv.ipynb 28
class ConvNetX(Classifier, LightningModule):
    def __init__(
            self,
            nnet:ConvNet,
            num_classes:int,
            optimizer:torch.optim.Optimizer,
            scheduler:torch.optim.lr_scheduler,
            ):
        logger.info("ConvNetX: init")
        super().__init__(num_classes, optimizer, scheduler)
        self.save_hyperparameters(logger=False, ignore=['nnet'])
        self.lr = optimizer.keywords['lr'] # for lr finder
        self.nnet = nnet

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.nnet(x)
    
    def _step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        preds = y_hat.argmax(dim=1)
        return loss, preds, y
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.forward(x)
        return y_hat.argmax(dim=1)