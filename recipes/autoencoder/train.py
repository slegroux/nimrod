#!/usr/bin/env python

from nimrod.modules import Encoder, Decoder
from nimrod.models import AutoEncoder, AutoEncoderPL
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
    print(OmegaConf.to_yaml(cfg))

    # SEED
    pl.seed_everything(cfg.seed, workers=True)

    # MODEL
    enc = Encoder()
    dec = Decoder()
    autoencoder = AutoEncoder(enc, dec)
    autoencoder_pl = AutoEncoderPL(autoencoder)

    # DATA
    full_train = MNISTDataset('~/Data', train=True)
    test = MNISTDataset('~/Data', train=False)
    train, dev = full_train.train_dev_split(0.8)
    # train_dl = instantiate(cfg.dataloaders.train)
    # print(cfg.dataloaders.train)
    train_l = DataLoader(train)
    dev_l = DataLoader(dev)
    test_l = DataLoader(test)

    # # TRAIN
    callbacks = []
    early_stopping = instantiate(cfg.callbacks.early_stopping)
    callbacks.append(early_stopping)
    model_checkpoint = instantiate(cfg.callbacks.model_checkpoint)
    callbacks.append(model_checkpoint)
    # # for _, cb_conf in cfg.callbacks.items():
    # #     callbacks.append(hydra.utils.instantiate(cb_conf))
    logger = instantiate(cfg.logger.wandb)
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=[logger])
    trainer.fit(model=autoencoder_pl, train_dataloaders=train_l, val_dataloaders=dev_l,ckpt_path=cfg.get("ckpt_path"))

    # # TEST
    if cfg.get("test"):
        trainer.test(autoencoder_pl, dataloaders=test_l)
    wandb.finish()

if __name__ == "__main__":
    main()