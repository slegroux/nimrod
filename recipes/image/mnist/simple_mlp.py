import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from nimrod.models.mlp import MLP, MLP_PL
from omegaconf import OmegaConf
from hydra.utils import instantiate
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger

print(f"PyTorch version: {torch.__version__}")

# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

# Set the device
device = "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device)
print(f"Using device: {device}")

# # Load the MNIST dataset
# train_dataset = MNIST("./data", train=True, download=True, transform=ToTensor())
# test_dataset = MNIST("./data", train=False, download=True, transform=ToTensor())

# # Define the data loaders
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Data module
cfg = OmegaConf.load('../../../config/data/image/mnist.yaml')
datamodule = instantiate(cfg.datamodule)
datamodule.prepare_data()
datamodule.setup()

train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()

# Define the model
model = MLP(n_in=28*28, n_h=64, n_out=10).to(device)

# mode = "pytorch"
mode = "lightning"

if mode == "pytorch":
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 1
    for epoch in range(n_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.view(-1, 28*28)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            # print(loss.item())
            optimizer.step()

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                # model expects input (B,H*W)
                images = images.view(-1, 28*28).to(device)
                images = images.to(device)
                labels = labels.to(device)
                # Pass the input through the model
                outputs = model(images)
                # Get the predicted labels
                _, predicted = torch.max(outputs.data, 1)

                # Update the total and correct counts
                total += labels.size(0)
                correct += (predicted == labels).sum()

            # Print the accuracy
            print(f"Epoch {epoch + 1}: Accuracy = {100 * correct / total:.2f}%")

if mode == "lightning":
    wandb_logger = WandbLogger(project="MNIST")
    trainer = pl.Trainer(accelerator=device.type, devices=1, max_epochs=5, logger=wandb_logger)    
    model_pl = MLP_PL(model)
    trainer.fit(model_pl, train_loader, val_loader)