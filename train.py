import os

import torch

from configs import get_srgan_config
from gans.SRGAN import SRGAN_G, SRGAN_D

config = {}

device = "cuda:0" if torch.cuda.is_available() else "cpu"

for epoch in range(config.epoch_count):
    pass



