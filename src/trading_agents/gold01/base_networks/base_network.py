import torch
from torch import nn

class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def returnOutFeaturesNumber(self):
        raise NotImplementedError('Not implemented.')
