#!/usr/bin/env python
from nimrod.models import AutoEncoderPL, AutoEncoder
from nimrod.modules import Encoder, Decoder
from nimrod.data.datasets import MNISTDataset
import torch
import os
script_dir = os.path.dirname(__file__)
ckpt_path = os.path.join(script_dir, "checkpoints/last-v3.ckpt")

a = AutoEncoder(Encoder(), Decoder())
mdl_pl = AutoEncoderPL(a)
model = mdl_pl.load_from_checkpoint(ckpt_path)
# model.eval()
# x = torch.randn(1, 28*28)
# with torch.no_grad():
#     y_hat = model(x)
# print(x.shape, y_hat.shape)

# ds = MNISTDataset(train=False)
# print(len(ds))

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