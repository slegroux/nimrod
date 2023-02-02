#!/usr/bin/env python

from nimrod.models.autoencoders import AutoEncoderPL, AutoEncoder
from nimrod.modules import Encoder, Decoder
from nimrod.data.datasets import MNISTDataset
from matplotlib import pyplot as plt
import torch
import os


run_id = "hp"
ckpt_path = os.path.join("logs/runs/", run_id, "checkpoints", "last.ckpt")


a = AutoEncoder(Encoder(), Decoder())
model = AutoEncoderPL.load_from_checkpoint(ckpt_path, autoencoder=a)
model.eval()

ds = MNISTDataset(train=False)
imgs = []
for index in torch.randint(0,100,(5,)):
    x = ds[index][0].flatten().unsqueeze(0)
    imgs.append(x)
    with torch.no_grad():
        y_hat = model(x)
    imgs.append(y_hat)

ds.show_grid(imgs)

# TODO: pure pytorch inference (to leverage tooling)
# enc = Encoder()
# dec = Decoder()
# mdl = AutoEncoder(enc, dec)
# checkpoint = torch.load(ckpt_path)
# hyper_parameters = checkpoint["hyper_parameters"]
# mdl = AutoEncoder(**hyper_parameters)
# mdl_weights = checkpoint["state_dict"]
# mdl.load_state_dict(mdl_weights)
# mdl.eval()