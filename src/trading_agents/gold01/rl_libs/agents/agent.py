import torch
from torch import nn

class Agent(nn.Module):
    def __init__(self):
        super().__init__()

    def selectAction(self, state):
        raise NotImplementedError('Not implemented.')
