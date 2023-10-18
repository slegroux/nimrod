#!/usr/bin/env python
from nimrod.models.mlp import MLP_PL
from omegaconf import OmegaConf
from hydra.utils import instantiate 
import pytorch_lightning as pl
import torch
import wandb
from pathlib import Path


# instantiate from configs
cfg = OmegaConf.load('conf/train.yaml')
model = instantiate(cfg.model)
# default train.yaml is setup for ...training! change num_workers for predict
datamodule = instantiate(cfg.datamodule, num_workers=0, persistent_workers=False)
datamodule.prepare_data()
datamodule.setup()


# local checkpoints
PATH = "logs/multiruns/2023-10-17_17-37-15/5/checkpoints/last.ckpt"
model = MLP_PL.load_from_checkpoint(PATH).to(torch.device('cpu'))

# WANDB to retrive checkpoints from cloud
# run = wandb.init()
# artifact = run.use_artifact('slegroux/MNIST-HP/model-girtmnkf:v0', type='model')
# artifact_dir = artifact.download()
# model = MLP_PL.load_from_checkpoint(Path(artifact_dir) / "model.ckpt").to(torch.device('cpu'))
artifact_dir = "artifacts/model-girtmnkf:v0"
model = MLP_PL.load_from_checkpoint(Path(artifact_dir) / "model.ckpt").to(torch.device('cpu'))

# # fake batch data 
# n_batch, n_channel, w, h = 2, 1, 28,28
# x = torch.rand((n_batch, n_channel, w*h))

# # straight pytorch
# model.eval()
# with torch.no_grad():
#     # model forward method calls mlp which is (B,C,W*H) unlike datamodule which is (B,C,W,H)
#     y_hat = model(x).argmax(dim=2)
# print(y_hat)

# with trainer predict
trainer = pl.Trainer()
preds = trainer.predict(model, datamodule.test_dataloader())

