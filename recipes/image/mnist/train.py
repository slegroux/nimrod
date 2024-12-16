#!/usr/bin/env python
import lightning as L
from lightning.pytorch.tuner import Tuner
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from nimrod.utils import set_seed
import json
import logging
log = logging.getLogger(__name__)

@hydra.main(version_base="1.3",config_path="conf", config_name="train_mlp.yaml")
def main(cfg: DictConfig) -> dict:

    # HPARAMS
    # print(OmegaConf.to_yaml(cfg))
    # convert hp to dict for logging & saving
    hp = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # SEED
    set_seed(cfg.seed)

    # MODEL
    model = instantiate(cfg.model)

    # DATA
    datamodule = instantiate(cfg.datamodule)

    # TRAIN
    callbacks = []
    for _, cb_conf in cfg.callbacks.items():
        callbacks.append(instantiate(cb_conf))

    loggers = []
    # logger.info("Instantiating logger <{}>".format(cfg.logger._target_))
    for log_conf in cfg.loggers:
        logger = instantiate(cfg[log_conf])
        # wandb logger special setup
        if isinstance(logger, L.pytorch.loggers.WandbLogger):
            # deal with hangs when hp optim multirun training 
            # wandb.init(settings=wandb.Settings(start_method="thread"))
            # wandb requires dict not DictConfig
            logger.experiment.config.update(hp["datamodule"], allow_val_change=True)
            logger.experiment.config.update(hp["model"], allow_val_change=True)
        loggers.append(logger)
    
    # print(json.dumps(hp, indent=4))
        
    # trainer
    profiler = instantiate(cfg.profiler)
    trainer = instantiate(cfg.trainer, callbacks=callbacks, profiler=profiler, logger=loggers)
    # trainer.logger.log_hyperparams(hp)

    # batch size & lr optimization
    tuner = Tuner(trainer)
    if cfg.get("bs_finder"):
        tuner.scale_batch_size(model, datamodule=datamodule, mode="power", init_val= 65536)
        if isinstance(logger, L.pytorch.loggers.WandbLogger):
            # bs is automatically updated in L datamodule but we need to manuallyupdate it in wandb
            logger.experiment.config.update({"batch_size": datamodule.hparams.batch_size}, allow_val_change=True)

    if cfg.get("lr_finder"):
        lr_finder = tuner.lr_find(model, datamodule=datamodule)
        print(lr_finder.results)
        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.show()
        new_lr = lr_finder.suggestion()
        model.hparams.lr = new_lr
        if isinstance(logger, L.pytorch.loggers.WandbLogger):
            logger.experiment.config.update({"lr": new_lr}, allow_val_change=True)


    if cfg.get("train"):
        # trainer.fit(model=autoencoder_pl, train_dataloaders=train_dl, val_dataloaders=dev_dl, ckpt_path=cfg.get("ckpt_path"))
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # # TEST
    if cfg.get("test"):
    #     # trainer.test(autoencoder_pl, dataloaders=test_dl)
        trainer.test(datamodule=datamodule, ckpt_path="best")

    # wandb.finish()

    # return trainer.callback_metrics[cfg.get("optimized_metric")].item()

if __name__ == "__main__":
    main()