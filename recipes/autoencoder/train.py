#!/usr/bin/env python
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
import wandb

@hydra.main(version_base="1.3",config_path="conf", config_name="train.yaml")
def main(cfg: DictConfig) -> None:

    # PARAMS
    print(OmegaConf.to_yaml(cfg))

    # SEED
    pl.seed_everything(cfg.seed, workers=True)

    # MODEL
    autoencoder_pl = instantiate(cfg.model)

    # DATA
    datamodule = instantiate(cfg.datamodules)

    # TRAIN
    callbacks = []
    for _, cb_conf in cfg.callbacks.items():
        callbacks.append(instantiate(cb_conf))
    logger = instantiate(cfg.logger)
    profiler = instantiate(cfg.profiler)
    trainer = instantiate(cfg.trainer, callbacks=callbacks, profiler=profiler, logger=[logger])

    if cfg.get("train"):
        # trainer.fit(model=autoencoder_pl, train_dataloaders=train_dl, val_dataloaders=dev_dl, ckpt_path=cfg.get("ckpt_path"))
        trainer.fit(autoencoder_pl, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # TEST
    if cfg.get("test"):
        # trainer.test(autoencoder_pl, dataloaders=test_dl)
        trainer.test(datamodule=datamodule)

    wandb.finish()

if __name__ == "__main__":
    main()