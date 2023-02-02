#!/usr/bin/env python

from nimrod.models import AutoEncoderPL, AutoEncoder
from nimrod.modules import Encoder, Decoder
from nimrod.data.datasets import MNISTDataset
from matplotlib import pyplot as plt
import torch
import os


run_id = "v0.0.0"
ckpt_path = os.path.join("logs/runs/", run_id, "checkpoints", "last.ckpt")


a = AutoEncoder(Encoder(), Decoder())
model = AutoEncoderPL.load_from_checkpoint(ckpt_path, autoencoder=a)
model.eval()

ds = MNISTDataset(train=False)
index = 128
ds.show_idx(index)
# ds.show_random()

# x = ds[index][0].flatten().unsqueeze(0)
# with torch.no_grad():
#     y_hat = model(x)
# print(x.shape, y_hat.shape)
# plt.imshow(y_hat.reshape(28,28).numpy(), cmap='gray')


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