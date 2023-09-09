# method='summation' or 'difference'
from .pipeline import Pipeline
import copy
import torch

# GAF stands for Gramian Angular Field.
class TransformGafPipeline(Pipeline):
    def __init__(self, features_to_channels=True):
        super().__init__(torch.Tensor, torch.Tensor)
        self.features_to_channels = features_to_channels

    def setupInSpecs(self, in_specs):
        self.in_specs = in_specs
        self.device = self.in_specs['device']
        self.out_specs = copy.deepcopy(self.in_specs)

    def generateOutSpecs(self):
        return self.out_specs

    def reset(self):
        raise NotImplementedError('Not implemented.')

    def _gasf(self, x_cos, x_sin):
        return torch.einsum('af, bf -> abf', x_cos, x_cos) - torch.einsum('af, bf -> abf', x_sin, x_sin)

    def _gadf(self, x_cos, x_sin):
        return torch.einsum('af, bf -> abf', x_sin, x_cos) - torch.einsum('af, bf -> abf', x_cos, x_sin)

    def _process(self, in_data, attachment_dict):
        x_cos = in_data
        x_sin = torch.sqrt(torch.clamp(1 - in_data**2, 0, 1))

        # GASF
        x_s = self._gasf(x_cos, x_sin)
        # GADF
        x_d = self._gadf(x_cos, x_sin)

        x = torch.cat((x_s, x_d), -1)

        if self.features_to_channels:
            x = x.permute((2, 0, 1))

        return x, attachment_dict
