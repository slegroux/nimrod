#!/usr/bin/env bash

# optuna hp tuning
python train.py --multirun project='mnist-hp' trainer.max_epochs=5

# show optuna training params
# python train.py hydra/sweeper=optuna --cfg hydra #-p hydra.sweeper