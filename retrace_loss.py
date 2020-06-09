from collections import namedtuple
import random
import time


import torch
import torch.nn as nn
import numpy as np

# Named tuple for a single step within a trajectory


class RetraceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, trajectory, task, actor, critic, gamma=0.95):


        raise NotImplementedError


