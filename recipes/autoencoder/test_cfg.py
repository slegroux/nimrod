import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
config = OmegaConf.load("cfg/callbacks.yml")
early_stopping = EarlyStopping(**config.callbacks.early_stopping)
# model_checkpoint = ModelCheckpoint(**config.callbacks.model_checkpoint)
config = OmegaConf.load("cfg/callbacks_hydra.yml")
early_stopping2 = hydra.utils.instantiate(config.callbacks.early_stopping)
print(early_stopping==early_stopping2)
print(early_stopping)
print(early_stopping2)