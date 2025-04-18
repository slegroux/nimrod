# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/models.attention.ipynb.

# %% auto 0
__all__ = ['logger', 'SelfAttention', 'MultiheadAttention']

# %% ../../nbs/models.attention.ipynb 3
import torch.nn as nn
import torch
from torch_lr_finder import LRFinder
from torchinfo import summary
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from einops import rearrange


from hydra.utils import instantiate
from omegaconf import OmegaConf

from matplotlib import pyplot as plt
import pandas as pd
import math

from ..utils import get_device, set_seed
from .core import Classifier

from pprint import pprint
import logging
from typing import List, Optional, Type


# %% ../../nbs/models.attention.ipynb 4
logger = logging.getLogger(__name__)
set_seed()

# %% ../../nbs/models.attention.ipynb 12
class SelfAttention(nn.Module):
    """
    Self-Attention layer for image data.

    Parameters
    ----------
    n_features : int
        Number of input features (channels).
    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Forward pass through the self-attention layer.    
    Notes
    -----
    This implementation uses a single linear layer to project the input tensor
    into query, key, and value tensors. The attention mechanism is applied to
    these tensors, and the result is projected back to the original feature space.
    """

    def __init__(
            self,
            n_features: int # number of features
            ):
        super().__init__()
        self.scale = math.sqrt(n_features)

        # TODO: check why GroupNorm is used
        # self.norm = nn.GroupNorm(1, n_features)
        self.norm = nn.BatchNorm2d(n_features)

        # project to focus on specific channel areas
        # self.k_projection = nn.Linear(n_features, n_features)
        # self.q_projection = nn.Linear(n_features, n_features)
        # self.v_projection = nn.Linear(n_features, n_features)

        # the 3 previous projections can be replaced by 1 larger projection
        self.kqv = nn.Linear(n_features, n_features*3)
        self.proj = nn.Linear(n_features, n_features)
        
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        inp = x
        t = self.norm(x).view(B, C, -1).transpose(1, 2) # B, T=H*W, C
        # k = self.k_projection(t)
        # q = self.q_projection(t)
        # v = self.v_projection(t)
        kqv = self.kqv(t)
        q, k, v = kqv.chunk(3, dim=-1)
        s = (q@k.transpose(1,2))/self.scale
        x = s.softmax(dim=-1)@v
        x = self.proj(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x + inp #specific to stable diffusion ~resnet style

# %% ../../nbs/models.attention.ipynb 15
class MultiheadAttention(nn.Module):
    """
    Multihead Attention module.
    
    Parameters
    ----------
    n_features : int
        Number of features.
    n_heads : int
        Number of heads or groups of features.
    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Forward pass of the multihead attention module.
    """

    def __init__(
            self,
            n_features: int, #number of features
            n_heads: int #number of heads or groups of features
            ):
        
        super().__init__()
        self.n_heads = n_heads # number of heads
        self.scale = math.sqrt(n_features)
        self.norm = nn.BatchNorm2d(n_features)
        self.kqv = nn.Linear(n_features, n_features*3)
        self.proj = nn.Linear(n_features, n_features)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        B, C, H, W = x.shape
        inp = x
        t = self.norm(x).view(B, C, -1).transpose(1, 2) # B, T=H*W, C
        kqv = self.kqv(t)
        # split channels into n_heads and reshape
        kqv = rearrange(kqv, 'b t (h d) -> (b h) t d', h=self.n_heads)
        q, k, v = kqv.chunk(3, dim=-1)
        s = (q@k.transpose(1,2))/self.scale
        x = s.softmax(dim=-1)@v
        x = rearrange(x, '(b h) t d -> b t (h d)', h=self.n_heads)
        x = self.proj(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x + inp #specific to stable diffusion ~resnet style
        
    
