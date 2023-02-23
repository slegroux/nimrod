#!/usr/bin/env bash
# run on 1 batch for debugging
python train.py trainer.fast_dev_run=True test=False trainer.devices=1 