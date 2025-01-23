#!/usr/bin/env python
import lightning as L
import torch
from lightning.pytorch.tuner import Tuner
from nimrod.utils import set_seed

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate, get_class

import wandb
import os
from IPython import embed
from pprint import pprint
from matplotlib import pyplot as plt
import multiprocessing

import logging
log = logging.getLogger(__name__)

# config_path = ["config", "config/image/model", "config/image/data"]

@hydra.main(version_base="1.3",config_path="config", config_name="train.yaml")
def main(cfg: DictConfig) -> dict:

    # HPARAMS
    # pprint(OmegaConf.to_yaml(cfg))
    # convert hp to dict for logging & saving
    hp = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # SEED
    if cfg.get("seed"):
        set_seed(cfg.seed)
    else:
        set_seed()

    # DATA
    log.info("Setup datamodule")
    datamodule = instantiate(cfg.data) #, num_workers=multiprocessing.cpu_count())
    datamodule.prepare_data()
    datamodule.setup()
    total_steps = len(datamodule.train_dataloader()) * cfg.trainer.max_epochs

    # OPTIMIZER
    log.info("Setup optimizer")
    optimizer = instantiate(cfg.optimizer)

    # SCHEDULER
    log.info("Setup scheduler")
    if get_class(cfg.scheduler._target_) == torch.optim.lr_scheduler.OneCycleLR:
        cfg.scheduler.total_steps = total_steps
    scheduler = instantiate(cfg.scheduler)

    # MODEL
    log.info("Setup model")
    model = instantiate(cfg.model, num_classes=datamodule.num_classes)(optimizer=optimizer, scheduler=scheduler)

    # CALLBACKS
    log.info("Setup callbacks")
    callbacks = []
    for _, cb_conf in cfg.callbacks.items():
        callbacks.append(instantiate(cb_conf))

    # LOGGERS
    log.info("Setup loggers")
    loggers = []

    if cfg.get("logger"):
        for log_ in cfg.logger:
            logger = instantiate(cfg['logger'][log_])
            # wandb logger special setup
            # if isinstance(logger, L.pytorch.loggers.wandb.WandbLogger):
                # deal with hangs when hp optim multirun training 
                # wandb.init(settings=wandb.Settings(start_method="thread"))
                # wandb requires dict not DictConfig
                # logger.experiment.config.update(hp)
                # logger.experiment.config.update(hp["data"], allow_val_change=True)
                # logger.experiment.config.update(hp["model"], allow_val_change=True)
                # logger.config = hp
            if hasattr(logger, 'log_hyperparams'):
                logger.log_hyperparams(hp)
            loggers.append(logger)
        
    # TRAINER
    log.info("Setup profiler")
    profiler = None
    if cfg.get("profiler"):
        profiler = instantiate(cfg.profiler)

    log.info("Setup trainer")
    trainer = instantiate(cfg.trainer, callbacks=callbacks, profiler=profiler, logger=loggers)

    # trainer.logger.log_hyperparams(hp)
    
    # TUNER
    tuner = Tuner(trainer)
    if cfg.get("tune_batch_size"):
        tuner.scale_batch_size(model, datamodule=datamodule, mode="power")

    if cfg.get("tune_lr"):
        log.info("Tuning learning rate")
        lr_finder = tuner.lr_find(
            model,
            datamodule=datamodule,
            min_lr=1e-6,
            max_lr=1.0,
            num_training=100,  # number of iterations to test
            )

        fig = lr_finder.plot(suggest=True)
        # plt.show()
        fig_name= os.path.join(cfg.paths.output_dir, 'lr_finder.png')
        fig.savefig(fig_name)
        log.info(f"lr_finder plot saved to {fig_name}")
        log.info(f"Suggested learning rate: {lr_finder.suggestion()}")
        if get_class(cfg.scheduler._target_) == torch.optim.lr_scheduler.OneCycleLR:
            cfg.scheduler.max_lr = lr_finder.suggestion()

    # # new_lr = lr_finder.suggestion()
    # # model.hparams.lr = new_lr

    # TRAIN
    if cfg.get("train"):
        log.info("Training model")
        trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader(), ckpt_path=cfg.get("ckpt_path"))

    # TEST
    if cfg.get("test"):
        log.info("Testing model")
        trainer.test(datamodule=datamodule, ckpt_path="best")
        log.info(f"Best ckpt path: {trainer.checkpoint_callback.best_model_path}")


    # # return trainer.callback_metrics[cfg.get("optimized_metric")].item()

if __name__ == "__main__":
    main()