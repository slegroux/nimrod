#!/usr/bin/env python

from nimrod.text.datasets import CharDataModule
from nimrod.models.lm import NNLM_L
from omegaconf import OmegaConf
from hydra.utils import instantiate
from lightning import Trainer

def main():
    # cfg1 = OmegaConf.load("../../config/text/data/tinyshakespeare.yaml")
    # cfg2 = OmegaConf.load("../../config/text/model/nnlm.yaml")
    # cfg = OmegaConf.merge(cfg1, cfg2)
    # print(cfg)
    cfg = OmegaConf.load("conf/train_test.yaml")
    dm = instantiate(cfg.datamodule)
    # dm.setup()
    lm  = instantiate(cfg.model)

    trainer = Trainer(accelerator="auto", fast_dev_run=True)
    trainer.fit(lm, dm)

if __name__ == "__main__":
    main()