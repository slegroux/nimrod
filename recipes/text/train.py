#!/usr/bin/env python
import lightning as L
from lightning.pytorch.tuner import Tuner
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
import wandb
from IPython import embed

@hydra.main(version_base="1.3",config_path="conf", config_name="train.yaml")
def main(cfg: DictConfig) -> dict:

    # HPARAMS
    # print(OmegaConf.to_yaml(cfg))
    # convert hp to dict for logging & saving

    hp = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    print(hp)

    # SEED
    L.seed_everything(cfg.seed, workers=True)

    # MODEL
    model = instantiate(cfg.model)

    # DATA
    datamodule = instantiate(cfg.data)

    # CALLBACKS
    callbacks = []
    for _, cb_conf in cfg.callbacks.items():
        callbacks.append(instantiate(cb_conf))

    # LOGGERS
    loggers = []
    if cfg.get("logger"):
        for _, log_conf in cfg.logger.items():
            if "_target_" in log_conf:
                logger = instantiate(log_conf)
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

    if loggers:
        for logger in trainer.loggers:
            logger.log_hyperparams(hp)

            # trainer.logger.log_hyperparams(hp)
    
    # TRAIN
    if cfg.get("train"):
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    # TEST
    if cfg.get("test"):
        ckpt_path = trainer.checkpoint_callback.best_model_path
        print("best ckpt: ", ckpt_path)
        trainer.test(datamodule=datamodule, ckpt_path="best")

    test_metrics = trainer.callback_metrics

    wandb.finish()


    # return trainer.callback_metrics[cfg.get("optimized_metric")].item()

if __name__ == "__main__":
    main()