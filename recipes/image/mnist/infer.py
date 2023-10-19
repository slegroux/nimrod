#!/usr/bin/env python
from nimrod.models.mlp import MLP_PL
from nimrod.models.mlp import MLP
from omegaconf import OmegaConf
from hydra.utils import instantiate 
import pytorch_lightning as pl
import torch
import wandb
from pathlib import Path
import random
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

# instantiate from configs
cfg = OmegaConf.load('conf/train.yaml')
model = instantiate(cfg.model)
# default train.yaml is setup for ...training! change num_workers for predict
datamodule = instantiate(cfg.datamodule, num_workers=0, persistent_workers=False)
datamodule.prepare_data()
datamodule.setup()

# local checkpoints
PATH = "ckpt/epoch=0-step=875.ckpt"
model = MLP_PL.load_from_checkpoint(PATH).to(torch.device('cpu'))

# WANDB to retrive checkpoints from cloud
run = wandb.init()
artifact = run.use_artifact('slegroux/MNIST-HP/model-0hfq6cko:v0', type='model')
artifact_dir = artifact.download()
model = MLP_PL.load_from_checkpoint(Path(artifact_dir) / "model.ckpt").to(torch.device('cpu'))

# artifact_dir = "artifacts/model-girtmnkf:v0"
# model = MLP_PL.load_from_checkpoint(Path(artifact_dir) / "model.ckpt").to(torch.device('cpu'))

# fake batch data 
n_batch, n_channel, w, h = 2, 1, 28,28
x = torch.rand((n_batch, n_channel, w*h))

def predict(x, model):
    model.eval()
    with torch.no_grad():
        # model forward method calls mlp which is (B,C,W*H) unlike datamodule which is (B,C,W,H)
        y_hat = model(x).argmax(dim=2)
    return y_hat

print(predict(x, model))

# image from dataset
for idx in random.sample(range(0,10),4):
    x, y = datamodule.data_test[idx][0].view(1,1,28*28), datamodule.data_test[idx][1]
    print(predict(x, model).item(), y)
    # plt.imshow(x.numpy().reshape(28,28),cmap='gray')
    # plt.title(str(y))
    # plt.savefig(str(y))
    # plt.show()

# # image from file
image = Image.open('../../../data/image/MNIST/97.jpg').convert('RGB')
# # img = torchvision.io.read_image('test.png')
image.show()
tf = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(),transforms.Resize((28,28))])
x = tf(image).view(1,1, 28*28)
print(x.shape)
print(predict(x, model).item())

# batch prediction with trainer predict
trainer = pl.Trainer()
preds = trainer.predict(model, datamodule.test_dataloader())
