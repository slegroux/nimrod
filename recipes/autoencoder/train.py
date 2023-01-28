from nimrod.modules import Encoder, Decoder
from nimrod.models import AutoEncoder, AutoEncoderPL
from nimrod.data.datasets import MNISTDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import os

def main():
    # MODEL
    enc = Encoder()
    dec = Decoder()
    autoencoder = AutoEncoder(enc, dec)
    autoencoder_pl = AutoEncoderPL(autoencoder)

    # DATA
    # train = MNIST(os.getcwd(), download=True, train=True, transform=transforms.ToTensor())
    # dev = MNIST(os.getcwd(), download=True, train=False, transform=transforms.ToTensor())
    # dev = MNISTDataset('~/Data', train=False)
    train = MNISTDataset('~/Data', train=True)

    train_loader = DataLoader(train)
    # dev_loader = DataLoader(dev)

    # TRAINING
    trainer = pl.Trainer(devices=[7], accelerator="gpu")
    trainer.fit(model=autoencoder_pl, train_dataloaders=train_loader)

if __name__ == "__main__":
    main()