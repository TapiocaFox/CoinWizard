import torch
from torch import nn
from .base_network import BaseNetwork

class GafCnnBaseNetwork(BaseNetwork):
    def __init__(self):
        super().__init__()

    def returnOutFeaturesNumber(self):
        raise NotImplementedError('Not implemented.')
