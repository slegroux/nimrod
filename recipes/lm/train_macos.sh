#!/usr/bin/env bash

python train.py \
    datamodule.num_workers=8 datamodule.persistent_workers=True \
    model.lr=1e-3 \
    trainer.max_epochs=10 trainer.devices=1 trainer.accelerator='mps'