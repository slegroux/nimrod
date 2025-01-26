#!/usr/bin/env python

import torch
from lightning.pytorch.tuner import Tuner
import wandb
from nimrod.utils import set_seed
from nimrod.models.core import lr_finder
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate, get_class
from rich import print
import logging
log = logging.getLogger(__name__)

# config_path = ["config", "config/image/model", "config/image/data"]

@hydra.main(version_base="1.3",config_path="config", config_name="train.yaml")
def main(cfg: DictConfig) -> dict:

    # HPARAMS
    # print(OmegaConf.to_yaml(cfg))
    # convert hp to dict for logging & saving
    hp = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    print(hp)

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
    log.info(f"Total steps: {total_steps}")

    # MODEL
    log.info("Setup partial model")
    model = instantiate(cfg.model, num_classes=datamodule.num_classes)#(optimizer=optimizer, scheduler=None)

    # OPTIMIZER
    log.info("Setup optimizer")
    optimizer = instantiate(cfg.optimizer)

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
            model_name = model.func.__name__
            # from IPython import embed; embed()
            run_name = f"{model_name}-bs:{datamodule.batch_size}-epochs:{cfg.trainer.max_epochs}"
            if log_ == 'wandb':
                cfg['logger'][log_]['name'] = run_name
                cfg['logger'][log_]['group'] = model_name

                logger = instantiate(cfg['logger'][log_])   
                # deal with hangs when hp optim multirun training 
                # wandb.init(settings=wandb.Settings(start_method="thread"))
                # wandb requires dict not DictConfig
                logger.experiment.config.update(hp, allow_val_change=True)
                # logger.experiment.config.update(hp["data"], allow_val_change=True)
                # logger.experiment.config.update(hp["model"], allow_val_change=True)
                logger.config = hp
            # if hasattr(logger, 'log_hyperparams'):
            #     logger.log_hyperparams(hp)
            else:
                logger = instantiate(cfg['logger'][log_])
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
        # TODO: should i pass same tuner to it?
        suggested_lr = lr_finder(model, datamodule, plot=cfg.get("plot_lr_tuning"))
        # fig_name= os.path.join(cfg.paths.output_dir, 'lr_finder.png')
        log.info(f"Suggested learning rate: {suggested_lr}")

        if get_class(cfg.scheduler._target_) == torch.optim.lr_scheduler.OneCycleLR:
            cfg.scheduler.max_lr = suggested_lr

    # new_lr = lr_finder.suggestion()

    # SCHEDULER
    log.info("Setup scheduler")
    if get_class(cfg.scheduler._target_) == torch.optim.lr_scheduler.OneCycleLR:
        cfg.scheduler.total_steps = total_steps

    scheduler = instantiate(cfg.scheduler)
    model = model(optimizer=optimizer, scheduler=scheduler)

    # TRAIN
    if cfg.get("train"):
        log.info("Training model")
        if cfg.get("ckpt_path"):
            log.info("Resuming training from ckpt")
            trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader(), ckpt_path=cfg.get("ckpt_path"))
        else:
            trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())
        
    log.info(f"Best ckpt path: {trainer.checkpoint_callback.best_model_path}")
    
    # TEST
    if cfg.get("test"):
        log.info("Testing model")
        trainer.test(datamodule=datamodule, ckpt_path="best")


    wandb.finish()
    # # return trainer.callback_metrics[cfg.get("optimized_metric")].item()

if __name__ == "__main__":
    main()