# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/utils.ipynb.

# %% auto 0
__all__ = ['get_device']

# %% ../nbs/utils.ipynb 4
import torch

# %% ../nbs/utils.ipynb 5
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device