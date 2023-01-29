#!/usr/bin/env python

from nimrod.modules import Encoder, Decoder
from nimrod.models import AutoEncoder, AutoEncoderPL
from nimrod.data.datasets import MNISTDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader
import os
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

@hydra.main(version_base=None,config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    # MODEL
    enc = Encoder()
    dec = Decoder()
    autoencoder = AutoEncoder(enc, dec)
    autoencoder_pl = AutoEncoderPL(autoencoder)

    # DATA
    full_train = MNISTDataset('~/Data', train=True)
    test = MNISTDataset('~/Data', train=False)
    train, dev = full_train.train_dev_split(0.8)
    dev = MNISTDataset('~/Data', train=False)

    train_l = DataLoader(train)
    dev_l = DataLoader(dev)
    test_l = DataLoader(test)

    # TRAINING
    root_dir = os.path.dirname(__file__)
    devices = [0,1,2,3,4,5,6,7]
    early_stopping = instantiate(cfg.callbacks.early_stopping)
    model_checkpoint = instantiate(cfg.callbacks.model_checkpoint)
    callbacks = [early_stopping, model_checkpoint]

    trainer = Trainer(
        default_root_dir=root_dir,
        # limit_train_batches=100, max_epochs=1000,
        callbacks = callbacks,
        devices=devices, accelerator="gpu"
        )
    # trainer.fit(model=autoencoder_pl, train_dataloaders=train_l, val_dataloaders=dev_l)
    # trainer.test(autoencoder_pl, dataloaders=test_l)

if __name__ == "__main__":
    main()