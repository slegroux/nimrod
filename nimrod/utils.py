"""Utilities to avoid re-inventing the wheel"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/utils.ipynb.

# %% auto 0
__all__ = ['FORMAT', 'get_device', 'set_seed', 'get_logger']

# %% ../nbs/utils.ipynb 4
import torch
import numpy as np
import random
import os
import logging
from rich.logging import RichHandler
import matplotlib
import lightning as L

# %% ../nbs/utils.ipynb 5
def get_device():
    return 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

# %% ../nbs/utils.ipynb 9
def set_seed(seed: int = 42) -> None:
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # # When running on the CuDNN backend, two further options must be set
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # # Set a fixed value for the hash seed
    # os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")
    L.seed_everything(seed, workers=True)

# %% ../nbs/utils.ipynb 13
# Configure the logger

FORMAT = "%(asctime)s"
logging.basicConfig(
    level=logging.DEBUG, format=FORMAT, datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, tracebacks_suppress=[matplotlib, L])]
)
# Create a logger
# logger = logging.getLogger(__name__)
def get_logger(name=__name__):
    return logging.getLogger(name)
