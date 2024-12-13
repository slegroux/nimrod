#!/usr/bin/env python

from nimrod.models.lm import NNBigramL
from omegaconf import OmegaConf
from hydra.utils import instantiate
import torch

# config
cfg = OmegaConf.load('config/train.yaml')
model = instantiate(cfg.model)
dm = instantiate(cfg.datamodule, num_workers=0, persistent_workers=False)
dm.prepare_data()
dm.setup()

# checkpoint
CKPT = "logs/runs/2023-10-31_14-38-56/checkpoints/epoch=8-step=13950.ckpt"
model = NNBigramL.load_from_checkpoint(CKPT).to(torch.device('cpu'))

print(dm.dataset.from_tokens(model.predict(idx=torch.zeros((1,1), dtype=torch.long), max_new_tokens=500)[0].tolist()))
