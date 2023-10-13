#!/usr/bin/env python
from nimrod.models.mlp import MLP_PL
from omegaconf import OmegaConf
from hydra.utils import instantiate 
import pytorch_lightning as pl
import torch

# checkpoints
PATH = "logs/runs/2023-10-11_21-50-57/checkpoints/last.ckpt"
model = MLP_PL.load_from_checkpoint(PATH).to(torch.device('cpu'))

# fake batch data 
n_batch, n_channel, data = 2, 1, 28*28
x = torch.rand((n_batch, n_channel, data))

# straight pytorch
model.eval()
with torch.no_grad():
    y_hat = model(x).argmax(dim=2)
print(y_hat)

# datamodule
cfg = OmegaConf.load('../../../config/data/image/mnist.yaml')
datamodule = instantiate(cfg.datamodule)
datamodule.prepare_data()
datamodule.setup()

# with trainer predict
trainer = pl.Trainer()
preds = trainer.predict(model, datamodule.test_dataloader())
