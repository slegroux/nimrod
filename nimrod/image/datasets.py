"""Image datasets"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/image.datasets.ipynb.

# %% auto 0
__all__ = ['logger', 'ImageDataset', 'MNISTDataset', 'MNISTDataModule']

# %% ../../nbs/image.datasets.ipynb 3
import torch
import torch.utils.data as data
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import torchvision
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from lightning import LightningDataModule

import pandas as pd
from matplotlib import pyplot as plt

import os

from omegaconf import OmegaConf
from hydra.utils import instantiate

from typing import Any, Dict, Optional, Tuple, List
from ..data.core import DataModule
from ..utils import set_seed

import logging


# %% ../../nbs/image.datasets.ipynb 4
set_seed(42)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
plt.set_loglevel('INFO')

# %% ../../nbs/image.datasets.ipynb 6
class ImageDataset(Dataset):
    " Base class for image datasets providing visualization of (image, label) samples"

    def __init__(self):
        logger.info("ImageDataset: init")
        super().__init__()

    def show_idx(self,
            index:int # Index of the (image,label) sample to visualize
        ):
        "display image from data point index of a image dataset"
        X, y = self.__getitem__(index)
        plt.figure(figsize = (1, 1))
        plt.imshow(X.numpy().reshape(28,28),cmap='gray')
        plt.title(f"Label: {int(y)}")
        plt.show()

    @staticmethod
    def show_grid(
            imgs: List[torch.Tensor], # python list of images dim (C,H,W)
            save_path=None, # path where image can be saved
            dims:Tuple[int,int] = (28,28)
        ):
        "display list of mnist-like images (C,H,W)"
        if not isinstance(imgs, list):
            imgs = [imgs]
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            axs[0, i].imshow(img.numpy().reshape(dims[0],dims[1]))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if save_path:
            plt.savefig(save_path)

    def show_random(
            self,
            n:int=3, # number of images to display
            dims:Tuple[int,int] = (28,28)
        ):
        "display grid of random images"
        indices = torch.randint(0,len(self), (n,))
        images = []
        for index in indices:
            X, y = self.__getitem__(index)
            X = X.reshape(dims[0],dims[1])
            images.append(X)
        self.show_grid(images)
        

# %% ../../nbs/image.datasets.ipynb 10
class MNISTDataset(ImageDataset):
    "MNIST digit dataset"

    def __init__(
        self,
        data_dir:str='../data/image', # path where data is saved
        train = True, # train or test dataset
        transform:torchvision.transforms.transforms=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        # TODO: add noramlization?
        # torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(0.1307,), (0.3081,))])

    ):
        logger.info("MNISTDataset: init")
        abs_data_dir = os.path.abspath(data_dir)
        logger.info(f"Data directory: {abs_data_dir}")
        os.makedirs(data_dir, exist_ok=True)
        super().__init__()
        
        try:
            self.ds = MNIST(
                data_dir,
                train = train,
                transform=transform, 
                download=True
            )
            logger.info(f"MNIST dataset loaded with total {len(self.ds)} samples")
        except Exception as e:
            logger.error(f"Error loading MNIST dataset: {e}")
            raise

    def __len__(self) -> int: # length of dataset
        return len(self.ds)
    
    def __getitem__(self, idx # index into the dataset
                    ) -> tuple[torch.FloatTensor, int]: # Y image data, x digit number
        x = self.ds[idx][0]
        y = self.ds[idx][1]
        return x, y
    
    def train_dev_split(
            self,
            ratio:float, # percentage of train/dev split,
        ) -> tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]: # train and set mnnist datasets

        train_set_size = int(len(self.ds) * ratio)
        valid_set_size = len(self.ds) - train_set_size

        # split the train set into two
        train_set, valid_set = data.random_split(self.ds, [train_set_size, valid_set_size])
        # TODO: cast to ImageDataset to allow for drawing
        # train_set, valid_set = Dataset(train_set),j Dataset(valid_set)
        return train_set, valid_set



# %% ../../nbs/image.datasets.ipynb 17
class MNISTDataModule(DataModule, LightningDataModule):
    def __init__(self,
                 data_dir: str | os.PathLike = "~/Data/", # path to source data dir
                 train_val_test_split:List[float] = [0.8, 0.1, 0.1], # train val test %
                 batch_size: int = 64, # size of compute batch
                 num_workers: int = 0, # num_workers equal 0 means that it’s the main process that will do the data loading when needed, num_workers equal 1 is the same as any n, but you’ll only have a single worker, so it might be slow
                 pin_memory: bool = False, # If you load your samples in the Dataset on CPU and would like to push it during training to the GPU, you can speed up the host to device transfer by enabling pin_memory. This lets your DataLoader allocate the samples in page-locked memory, which speeds-up the transfer
                 persistent_workers: bool = False
                 ):

        logger.info("Init MNIST DataModule")
        super().__init__(train_val_test_split, batch_size, num_workers, pin_memory, persistent_workers)
        self.save_hyperparameters()

    @property
    def num_classes(self) -> int: # num of classes in dataset
        return 10

    def prepare_data(self) -> None:
        """Download data if needed + format with MNISTDataset
        """
        # train set
        MNISTDataset(self.hparams.data_dir, train=True)
        # test set
        MNISTDataset(self.hparams.data_dir, train=False)

    def setup(self, stage: Optional[str] = None) -> None:
        # stage: {fit,validate,test,predict}\n",
        # concat train & test mnist dataset and randomly generate train, eval, test sets
        if not self.data_train or not self.data_val or not self.data_test:
            # ((B, H, W), int)
            trainset = MNISTDataset(self.hparams.data_dir, train=True, transform=self.transforms)
            testset = MNISTDataset(self.hparams.data_dir, train=False, transform=self.transforms)
            dataset = ConcatDataset(datasets=[trainset, testset])
            # TODO: keep test set untouched
            lengths = [int(split * len(dataset)) for split in self.hparams.train_val_test_split]
            self.data_train, self.data_val, self.data_test = random_split(dataset=dataset, lengths=lengths)
        
