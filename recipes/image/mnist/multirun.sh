#!/usr/bin/env bash

python train.py --multirun model.mlp.n_h=16,64,256 logger.group="multirun" trainer.max_epochs=5
