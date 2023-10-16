#!/usr/bin/env python
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
import wandb

@hydra.main(version_base="1.3",config_path="conf", config_name="train.yaml")
def main(cfg: DictConfig) -> dict:

    # PARAMS
    # print(OmegaConf.to_yaml(cfg))

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

    logger = instantiate(cfg.logger)
    if isinstance(logger, pl.loggers.wandb.WandbLogger):
        # deal with hangs when hp optim multirun training 
        wandb.init(settings=wandb.Settings(start_method="thread"))
        # wandb requires dict not DictConfig
        hp = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        logger.experiment.config.update(hp)

    profiler = instantiate(cfg.profiler)
    trainer = instantiate(cfg.trainer, callbacks=callbacks, profiler=profiler, logger=[logger])

    if cfg.get("train"):
        # trainer.fit(model=autoencoder_pl, train_dataloaders=train_dl, val_dataloaders=dev_dl, ckpt_path=cfg.get("ckpt_path"))
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    train_metrics = trainer.callback_metrics

    # TEST
    if cfg.get("test"):
        # trainer.test(autoencoder_pl, dataloaders=test_dl)
        trainer.test(datamodule=datamodule, ckpt_path="best")
    test_metrics = trainer.callback_metrics

    wandb.finish()
    metric_dict = {**train_metrics, **test_metrics}
    return metric_dict[cfg.get("optimized_metric")]

if __name__ == "__main__":
    main()