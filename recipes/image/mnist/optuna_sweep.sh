#!/usr/bin/env bash

# optuna hp tuning
# python train.py --multirun project='mnist-hp' trainer.max_epochs=5
# python train.py --multirun project='mnist-hp' trainer.max_epochs=10 datamodule.batch_size=64,128,256 'model.lr=tag(log, interval(0.001, 1))' 'model.dropout=interval(0,1)'
python train.py --multirun project='mnist-hp' \
    logger.group='mac-sweepin-all' logger.log_model='all' \
    trainer.max_epochs=2 \
    model.dropout=0.31 model.n_h=128

# show optuna training params
# python train.py hydra/sweeper=optuna --cfg hydra #-p hydra.sweeper