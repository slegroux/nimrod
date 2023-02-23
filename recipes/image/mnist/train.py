#!/usr/bin/env python
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
import wandb
from ray.tune.integration.pytorch_lightning import TuneReportCallback

@hydra.main(version_base="1.3",config_path="conf", config_name="train.yaml")
def main(cfg: DictConfig) -> None:

    # PARAMS
    print(OmegaConf.to_yaml(cfg))

    # SEED
    pl.seed_everything(cfg.seed, workers=True)

    # MODEL
    model = instantiate(cfg.model)

    # DATA
    datamodule = instantiate(cfg.datamodule)

    # TRAIN
    callbacks = []
    for _, cb_conf in cfg.callbacks.items():
        callbacks.append(instantiate(cb_conf))
    
    # ray tune callback
    metrics = {"loss": "val/avg_loss", "acc": "val/avg_accuracy"}
    tune_cb = TuneReportCallback(metrics, on="validation_end")
    callbacks.append(tune_cb)

    logger = instantiate(cfg.logger)

    # profiler = instantiate(cfg.profiler)
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=[logger]) #, profiler=profiler) #, logger=[logger])

    if cfg.get("train"):
        # trainer.fit(model=autoencoder_pl, train_dataloaders=train_dl, val_dataloaders=dev_dl, ckpt_path=cfg.get("ckpt_path"))
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # TEST
    if cfg.get("test"):
        # trainer.test(autoencoder_pl, dataloaders=test_dl)
        trainer.test(datamodule=datamodule, ckpt_path="best")

    wandb.finish()

if __name__ == "__main__":
    main()