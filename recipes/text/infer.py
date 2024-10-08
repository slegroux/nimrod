#!/usr/bin/env python

from nimrod.models.lm import NNLM_L
from omegaconf import OmegaConf
from hydra.utils import instantiate 
import torch
from IPython import embed
import os

def main():
    # instantiate from configs
    cfg = OmegaConf.load('conf/train.yaml')

    lm = instantiate(cfg.model)
    # default train.yaml is setup for ...training! change num_workers for predict
    dm = instantiate(cfg.datamodule, num_workers=0, persistent_workers=False)

    # local checkpoints
    run_id = "batch_size=512-lr=0.001-context_len=3"
    ckpt = "epoch=020-val_loss=1.74.ckpt"
    PATH = os.path.join("logs", "runs", run_id, "checkpoints", ckpt)
    
    lm = NNLM_L.load_from_checkpoint(PATH)
    lm.to("cpu")
    sequences = lm.sample(
        n_iterations=20,
        bos=dm.v.stoi('<bos>'),
        eos=dm.v.stoi('<eos>'),
        pad=dm.v.stoi('<pad>')
        )
    for seq in sequences:
        print(''.join(dm.v.itos(i) for i in seq))


if __name__ == "__main__":
    main()
