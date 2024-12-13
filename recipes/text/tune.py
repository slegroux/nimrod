#!/usr/bin/env python
import lightning as L
from lightning.pytorch.tuner import Tuner
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from IPython import embed

@hydra.main(version_base="1.3",config_path="conf", config_name="train.yaml")
def main(cfg: DictConfig) -> dict:

    # HPARAMS
    # print(OmegaConf.to_yaml(cfg))
    # convert hp to dict for logging & saving
    hp = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # SEED
    L.seed_everything(cfg.seed) #, workers=True)

    # MODEL
    model = instantiate(cfg.model)

    # DATA
    datamodule = instantiate(cfg.datamodule)

    # CALLBACKS
    callbacks = []
    for _, cb_conf in cfg.callbacks.items():
        callbacks.append(instantiate(cb_conf))

    # LOGGERS
    loggers = []
    for log_conf in cfg.loggers:
        logger = instantiate(cfg[log_conf])
        # wandb logger special setup
        if isinstance(logger, L.pytorch.loggers.wandb.WandbLogger):
            # deal with hangs when hp optim multirun training 
            # wandb.init(settings=wandb.Settings(start_method="thread"))
            # wandb requires dict not DictConfig
            logger.experiment.config.update(hp)
        loggers.append(logger)
        
    # TRAINER
    profiler = instantiate(cfg.profiler)
    trainer = instantiate(cfg.trainer, callbacks=callbacks, profiler=profiler, logger=loggers)
    trainer.logger.log_hyperparams(hp)
    
    # TUNER
    tuner = Tuner(trainer)
    # tuner.scale_batch_size(model, datamodule=datamodule, mode="power")
    # print("suggested batch size: ", datamodule.hparams.batch_size)
    lr_finder = tuner.lr_find(model, datamodule=datamodule)        
    fig = lr_finder.plot(suggest=True)
    fig.savefig('lr_finder.png')

    new_lr = lr_finder.suggestion()
    print("suggested lr: ", new_lr)
    # model.hparams.lr = new_lr


if __name__ == "__main__":
    main()