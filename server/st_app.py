#!/usr/bin/env python

import streamlit as st
st.set_page_config(page_title="Digit Recognition Tutorial", page_icon=":shark:",)
import os
import time
import glob
import os
from PIL import Image
from omegaconf import OmegaConf
import json
import time
from dotenv import load_dotenv
import wandb
from hydra.utils import instantiate 
from nimrod.models.mlp import MLP_PL
import torch
import torchvision.transforms as transforms
from pathlib import Path
from matplotlib import pyplot as plt

load_dotenv()

def predict(x, model):
    model.eval()
    with torch.no_grad():
        # model forward method calls mlp which is (B,C,W*H) unlike datamodule which is (B,C,W,H)
        y_hat = model(x).argmax(dim=2)
    return y_hat

def process_image(image):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(),transforms.Resize((28,28))])
    x = tf(image).view(1,1, 28*28)
    return x

def main():
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.header("Digit recognizer")
    st.markdown(hide_st_style, unsafe_allow_html=True)
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    model = load_model()

    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.image(image, width=250)
        x = process_image(image)
        predictions = predict(x, model).item()
        st.write("**Recognized digit:**", predictions)
        
@st.cache_resource()
def load_model():
    cfg = OmegaConf.load('../recipes/image/mnist/conf/train.yaml')
    model = instantiate(cfg.model)
    run = wandb.init()
    artifact = run.use_artifact('slegroux/MNIST-HP/model-0hfq6cko:v0', type='model')
    artifact_dir = artifact.download()
    wandb.finish()
    model = MLP_PL.load_from_checkpoint(Path(artifact_dir) / "model.ckpt").to(torch.device('cpu'))
    return model

if __name__ == "__main__":
 
    main()