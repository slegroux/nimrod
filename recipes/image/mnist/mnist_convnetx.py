from omegaconf import OmegaConf
from hydra.utils import instantiate

N_EPOCHS = 5
lr_found = 0.012

cfg = OmegaConf.load('../../../config/data/image/mnist.yaml')
cfg.data_dir = "../../../data/image"
cfg.batch_size = 512
cfg.num_workers = 0
dm = instantiate(cfg)
dm.prepare_data()
dm.setup()
total_train_steps = len(dm.train_dataloader()) * N_EPOCHS # number of batches in training data * epochs

# cfg = OmegaConf.load('../../../config/optimizer/adam.yaml')
# optimizer = instantiate(cfg)

# # cfg = OmegaConf.load('../../../config/scheduler/reduce_lr_on_plateau.yaml')
# cfg = OmegaConf.load('../../../config/scheduler/one_cycle_lr.yaml')
# scheduler = instantiate(cfg)(optimizer=optimizer, total_steps=total_train_steps)

cfg = OmegaConf.load('../../../config/model/image/convnetx_adam.yaml')
# cfg.scheduler.total_steps = total_train_steps
mdl = instantiate(cfg)

cfg = OmegaConf.load('../../../config/trainer/debug.yaml')
trainer = instantiate(cfg)(max_epochs=N_EPOCHS)

trainer.fit(mdl, datamodule=dm)