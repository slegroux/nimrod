from omegaconf import OmegaConf
from hydra.utils import instantiate
from nimrod.utils import get_device, set_seed
import torch
import torch.nn as nn
import logging
from typing import Callable
from matplotlib import pyplot as plt
from fastprogress.fastprogress import master_bar, progress_bar

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
set_seed()

MAX_EPOCHS = 5

def config():
    logger.info("data preparation")
    cfg = OmegaConf.load('../../../config/data/image/mnist.yaml')
    cfg.data_dir = "../../../data/image"
    cfg.batch_size = 64
    cfg.num_workers = 0
    dm = instantiate(cfg)
    dm.prepare_data()
    dm.setup()
    total_train_steps = len(dm.train_dataloader()) * MAX_EPOCHS # number of batches in training data * epochs

    logger.info("load model")
    cfg = OmegaConf.load('../../../config/model/image/convnet.yaml')
    mdl = instantiate(cfg.batchnorm)

    logger.info("optimizer")
    cfg = OmegaConf.load('../../../config/optimizer/adam.yaml')
    optimizer = instantiate(cfg)
    optimizer = optimizer(params=mdl.parameters())

    logger.info("scheduler")
    # cfg = OmegaConf.load('../../../config/scheduler/reduce_lr_on_plateau.yaml')
    cfg = OmegaConf.load('../../../config/scheduler/one_cycle_lr.yaml')
    scheduler = instantiate(cfg)(optimizer=optimizer, total_steps=total_train_steps)

    logger.info("trainer")
    cfg = OmegaConf.load('../../../config/trainer/debug.yaml')
    trainer = instantiate(cfg) #, callbacks=[scheduler], optimizer=optimizer)

    return dm, mdl, optimizer, scheduler, trainer

def train_epoch(
        model,
        criterion,
        optimizer,
        scheduler,
        total_steps=1000,
        train_dl=None,
        device='cpu',
        ):
    model.train()
    lr_step = []
    loss_step = []
    for images, labels in train_dl:

        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)        
        loss.backward()
        optimizer.step()
        scheduler.step()    
        # current_lr = scheduler.get_last_lr()[0]
        current_lr = optimizer.param_groups[0]['lr']
        lr_step.append(current_lr)
        loss_step.append(loss.item())

    return loss_step, lr_step


def val_epoch(
        model,
        criterion,
        val_dl=None,
        device='cpu',
        ):
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        loss_total = 0
        for images, labels in val_dl:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            eval_loss = criterion(outputs, labels)
            loss_total += eval_loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = (correct / total)
        
    return loss_total/len(val_dl), accuracy



if __name__ == '__main__':
    dm, mdl, optimizer, scheduler, trainer = config()
    device = get_device()
    mdl.to(device)
    criterion = nn.CrossEntropyLoss()
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()

    lrs = []
    train_losses_steps = [] # per step
    train_losses = [] # per epoch
    val_losses = [] # per epoch


    for epoch in range(MAX_EPOCHS):
        train_loss_steps, lr_steps = train_epoch(
            model=mdl,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            total_steps=len(train_dl) * MAX_EPOCHS,
            train_dl=train_dl,
            device=device,
        )
        train_losses_steps.extend(train_loss_steps)
        train_losses.append(sum(train_loss_steps)/float(len(train_dl)))
        lrs.extend(lr_steps)

        val_loss_epoch, val_accuracy = val_epoch(
            model=mdl,
            criterion=criterion,
            val_dl=val_dl,
            device=device,

        )
        val_losses.append(val_loss_epoch)

        logger.info(
            f"Epoch {epoch}: \
            Train Loss = {train_losses[-1]:.4f}, \
            Val Loss = {val_loss_epoch:.4f}, \
            Accuracy = {100*val_accuracy:.2f}"
            )
        # trainer.save_checkpoint(f"epoch_{epoch + 1}.ckpt")


    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.title("training losses")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.plot(train_losses, label='train loss', color='blue', alpha=0.5)
    plt.plot(val_losses, label='val loss', color='red', marker='o')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(212)
    plt.title("learning rate schedule")
    plt.ylabel('lr')
    plt.xlabel('step')
    plt.plot(lrs)
    plt.tight_layout()
    plt.savefig("mnist_convnet.png")
    plt.show()