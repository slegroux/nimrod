#!/usr/bin/env python

from nimrod.models.lm import NNLM_X
from omegaconf import OmegaConf
from hydra.utils import instantiate 
import torch
from IPython import embed
import os

def main():
    # instantiate from configs
    cfg = OmegaConf.load('conf/train.yaml')

    # default train.yaml is setup for ...training! change num_workers for predict
    dm = instantiate(cfg.datamodule, num_workers=0, persistent_workers=False)

    # the model config is a partial (optimizer/scheduler are added by the trainer): build the nnet only
    nnlm_cfg = OmegaConf.load('../../config/text/model/nnlm.yaml')
    nnet = instantiate(nnlm_cfg.nnet, n_vocab=len(dm.v))

    # local checkpoints
    run_id = "batch_size=512-lr=0.001-context_len=3"
    ckpt = "epoch=020-val_loss=1.74.ckpt"
    PATH = os.path.join("logs", "runs", run_id, "checkpoints", ckpt)
    
    # nnet is not stored in the checkpoint hparams: pass the instantiated one
    lm = NNLM_X.load_from_checkpoint(PATH, nnet=nnet, weights_only=False)
    lm.to("cpu")
    print(lm.sample("the ", dm.v, max_new_tokens=100))


if __name__ == "__main__":
    main()
