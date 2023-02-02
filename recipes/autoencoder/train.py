#!/usr/bin/env python

from nimrod.modules import Encoder, Decoder
from nimrod.models.autoencoders import AutoEncoder, AutoEncoderPL
from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler
from nimrod.data.datasets import MNISTDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
import wandb
import os

@hydra.main(version_base="1.3",config_path="conf", config_name="train.yaml")
def main(cfg: DictConfig) -> None:

    # PARAMS
    # print(OmegaConf.to_yaml(cfg))

    # SEED
    pl.seed_everything(cfg.seed, workers=True)

    # MODEL
    enc = Encoder()
    dec = Decoder()
    autoencoder = AutoEncoder(enc, dec)
    autoencoder_pl = AutoEncoderPL(autoencoder)

    # DATA
    full_train = instantiate(cfg.datasets.train)
    test = instantiate(cfg.datasets.test)
    train, dev = full_train.train_dev_split(0.8)
    train_dl = instantiate(cfg.dataloaders.train, dataset=train)
    dev_dl = instantiate(cfg.dataloaders.dev, dataset=dev)
    test_dl = instantiate(cfg.dataloaders.dev, dataset=test)

    # TRAIN
    callbacks = []
    for _, cb_conf in cfg.callbacks.items():
        callbacks.append(instantiate(cb_conf))

    logger = instantiate(cfg.logger)
    profiler = instantiate(cfg.profiler)

    trainer = instantiate(cfg.trainer, callbacks=callbacks, profiler=profiler, logger=[logger])

    if cfg.get("train"):
        trainer.fit(model=autoencoder_pl, train_dataloaders=train_dl, val_dataloaders=dev_dl, ckpt_path=cfg.get("ckpt_path"))

    # TEST
    if cfg.get("test"):
        trainer.test(autoencoder_pl, dataloaders=test_dl)

    wandb.finish()

if __name__ == "__main__":
    main()