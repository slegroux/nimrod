from omegaconf import OmegaConf
from hydra.utils import instantiate

N_EPOCHS = 2
lr_found = 0.012

# cfg = OmegaConf.load('../../../config/data/image/mnist.yaml')
# cfg.data_dir = "../../../data/image"
# cfg.batch_size = 1024
# cfg.num_workers = 0
# dm = instantiate(cfg)
# dm.prepare_data()
# dm.setup()
# total_train_steps = len(dm.train_dataloader()) * N_EPOCHS # number of batches in training data * epochs
total_train_steps = 100

cfg = OmegaConf.load('../../../config/optimizer/adam_w.yaml')
optimizer = instantiate(cfg)

# cfg = OmegaConf.load('../../../config/scheduler/reduce_lr_on_plateau.yaml')
# scheduler = instantiate(cfg)

cfg = OmegaConf.load('../../../config/scheduler/step_lr.yaml') #, '../../../config/scheduler/one_cycle_lr.yaml')
# cfg.total_steps = total_train_steps
scheduler = instantiate(cfg) #, total_steps=total_train_steps)
print(scheduler)

# cfg = OmegaConf.load('../../../config/model/image/mlp_.yaml')
# nnet = instantiate(cfg)

cfg = OmegaConf.load('../../../config/model/image/mlp.yaml')
model = instantiate(cfg)(optimizer=optimizer, scheduler=scheduler)

cfg = OmegaConf.load('../../../config/trainer/default.yaml')
trainer = instantiate(cfg,max_epochs=N_EPOCHS, default_root_dir='.')

# trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())