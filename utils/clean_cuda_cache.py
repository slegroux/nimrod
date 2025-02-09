import torch
import logging
from nimrod.utils import get_device
logger = logging.getLogger(__name__)

device = get_device()

if device == "mps":
    logger.info("Cleaning MPS cache")
    with torch.no_grad():
        torch.mps.empty_cache()
else:
    logger.info("Cleaning CUDA cache")
    with torch.no_grad():
        torch.cuda.empty_cache()