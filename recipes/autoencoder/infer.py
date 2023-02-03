#!/usr/bin/env python

from nimrod.models.autoencoders import AutoEncoderPL, AutoEncoder
from nimrod.modules import Encoder, Decoder
from nimrod.data.datasets import MNISTDataset
from matplotlib import pyplot as plt
import torch
import os
import argparse
from hydra.utils import instantiate
from omegaconf import DictConfig
import hydra
from time import perf_counter
import onnxruntime
import numpy as np

def infer_rand_samples(ds, model,data_device=None):
    start = perf_counter()
    imgs = []
    model.eval()
    for index in torch.randint(0,100,(5,)):
        x = ds[index][0].flatten().unsqueeze(0)
        if data_device:
            x = x.to(data_device)
        imgs.append(x)
        with torch.no_grad():
            y_hat = model(x)
        imgs.append(y_hat)
    stop = perf_counter()
    print(f"{stop-start}")
    return imgs

def evaluate(cfg: DictConfig):
    run_id = "hp"
    ckpt_path = os.path.join("logs/runs/", run_id, "checkpoints", "last.ckpt")
    autoencoder = instantiate(cfg.model.autoencoder)
    model = AutoEncoderPL.load_from_checkpoint(ckpt_path, autoencoder=autoencoder)
    
    ds = MNISTDataset(train=False)
    # CPU
    print('CPU:')
    imgs = infer_rand_samples(ds, model.cpu(), data_device='cpu')
    ds.show_grid([img.cpu() for img in imgs], save_path="cpu.png")
    # GPU
    print('GPU:')
    imgs = infer_rand_samples(ds, model.cuda(), data_device='cuda')
    ds.show_grid([img.cpu() for img in imgs], save_path="gpu.png")

    # JIT CPU
    cpu_jit_model = model.cpu().to_torchscript()
    torch.jit.save(cpu_jit_model, "jit_cpu.pt")
    # cpu_jit_model = torch.jit.load("jit_cpu.pt")
    print('JIT CPU:')
    imgs = infer_rand_samples(ds, cpu_jit_model, data_device='cpu')
    ds.show_grid(imgs, save_path="jit_cpu.png")

    # JIT GPU
    gpu_jit_model = model.cuda().to_torchscript()
    torch.jit.save(cpu_jit_model, "jit_gpu.pt")
    print('JIT GPU:')
    imgs = infer_rand_samples(ds, gpu_jit_model, data_device='cuda')
    ds.show_grid([img.cpu() for img in imgs], save_path="jit_gpu.png")

    # ONNX
    input_sample = torch.randn((1, 28*28))
    model.to_onnx("model.onnx", input_sample, export_params=True)

    ort_session = onnxruntime.InferenceSession("model.onnx")
    input_name = ort_session.get_inputs()[0].name
    x = ds[0][0].flatten().unsqueeze(0)
    ort_inputs = {input_name: x.numpy().astype(np.float32)}
    start = perf_counter()
    ort_outs = ort_session.run(None, ort_inputs)
    stop = perf_counter()
    print(stop-start)
    imgs = [x, torch.Tensor(ort_outs)]
    ds.show_grid(imgs, save_path='onnx.png')

    # from IPython import embed; embed()    
    # BATCH INFERENCE
    # dl = instantiate(cfg.dataloaders.test, dataset=ds)
    # b = next(iter(dl))
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

@hydra.main(version_base="1.3", config_path="conf", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)

if __name__ == "__main__":
    main()